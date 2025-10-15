"""Module defining utility functions for agents."""

import torch
import torch.nn as nn
import math
import re
import numpy as np
from time import time

EPS = 1e-5

class TanhTransform(torch.distributions.transforms.Transform):
    """Implementation of the Tanh transformation."""

    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may
        # degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):

        return 2.0 * (math.log(2.0) - x - torch.nn.functional.softplus(-2.0 * x))


class SquashedNormal(
    torch.distributions.transformed_distribution.TransformedDistribution
):
    """Implementation of the Squashed Normal distribution."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = torch.distributions.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class TruncatedNormal(torch.distributions.Normal):
    """Implementation of the Truncated Normal distribution."""

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(  # pylint: disable=W0237
        self, clip=None, sample_shape=torch.Size()
    ) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = torch.distributions.utils._standard_normal(  # pylint: disable=W0212
            shape, dtype=self.loc.dtype, device=self.loc.device
        )
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step) -> float:
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2


def reparameterise(x, clamp=("hard", -5, 2), params=False):
    """
    The reparameterisation trick.
    Construct a Gaussian from x, taken to parameterise
    the mean and log standard deviation.
    """
    mean, log_std = torch.split(x, int(x.shape[-1] / 2), dim=-1)

    if clamp[0] == "hard":  # This is used by default for the SAC policy.
        log_std = torch.clamp(log_std, clamp[1], clamp[2])
    elif clamp[0] == "soft":  # This is used by default for the PETS model.
        log_std = clamp[1] + torch.nn.functional.softplus(log_std - clamp[1])
        log_std = clamp[2] - torch.nn.functional.softplus(clamp[2] - log_std)
    return (
        (mean, log_std)
        if params
        else torch.distributions.Normal(mean, torch.exp(log_std))
    )


def squashed_gaussian(x, sample=True):
    """
    For continuous spaces. Interpret pi as the mean
    and log standard deviation of a Gaussian,
    then generate an action by sampling from that
    distribution and applying tanh squashing.
    """
    gaussian = reparameterise(x)
    if sample:
        action_unsquashed = (
            gaussian.rsample()
        )  # rsample() required to allow differentiation.
    else:
        action_unsquashed = gaussian.mean
    action = torch.tanh(action_unsquashed)
    # Compute log_prob from Gaussian, then apply correction for tanh squashing.
    log_prob = gaussian.log_prob(action_unsquashed).sum(axis=-1)
    log_prob -= (
        2
        * (
            np.log(2)
            - action_unsquashed
            - torch.nn.functional.softplus(-2 * action_unsquashed)
        )
    ).sum(axis=-1)
    return action, log_prob.unsqueeze(-1)

def weight_init(m) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            # if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
                
# Global variable to track the last saved time
last_saved_time = 0  # Initialize the last saved time

def store_tensor(tensor: torch.Tensor, name:str='', tensor_save_interval:int=1200):
    """Store a tensor to disk for further analysis.

    Args:
        tensor (torch.Tensor): The tensor to store.
        name (str, optional): Names for this tensor. Defaults to ''.
        tensor_save_interval (int, optional): At which time interval we want to store a new tensor. Defaults to 1200.
    """
    global last_saved_time
    current_time = time.time()  # Get current time in seconds since epoch

    # Check if 20 minutes have passed since the last save
    if current_time - last_saved_time >= tensor_save_interval:
        last_saved_time = current_time
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f'tensor/tensor_{name}_{timestamp}.pt'
        torch.save(tensor, filename)