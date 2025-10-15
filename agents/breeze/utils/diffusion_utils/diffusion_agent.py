import copy
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

class BaseAgent(nn.Module):
    def __init__(
        self, 
        policy: torch.nn.Module,
        v_model: torch.nn.Module, 
        gamma: float = 0.99,
        utd: int = 2,
        start_steps: int = int(25e3),
        ema: float = 1e-3,
    ):       
        super().__init__()

        self.policy = policy
        self.v_model = v_model
        self.policy_target = copy.deepcopy(policy)
        self.v_model_target = copy.deepcopy(v_model)
        
        self.start_steps = start_steps
        self.utd = utd
        self.gamma = gamma
        self.device = policy.device
        self.ema = ema
            
    def ema_update_policy(self):
        for param, target_param in zip(self.policy.parameters(), self.policy_target.parameters()):
            if param.data is not None and target_param.data is not None:
                target_param.data.copy_(
                    self.ema * param.data + (1 - self.ema) * target_param.data
                )
                     
    def next_state(self, state):
        """
        get the action during evaluation
        """
        pass
    
    def load(self, ckpt_path):
        pass
    
    def policy_loss(self):
        pass
    
    def v_loss(self):
        pass

def extract(a, x_shape):
    '''
    align the dimention of alphas_cumprod_t to x_shape
    
    a: alphas_cumprod_t, B
    x_shape: B x F x F x F
    output: alphas_cumprod_t B x 1 x 1 x 1]
    '''
    b, *_ = a.shape
    return a.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def vp_beta_schedule(timesteps):
    """Discret VP noise schedule
    """
    t = torch.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.
    b_min = 0.1

    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas       


SCHEDULE = {
    'linear': linear_beta_schedule,
    'cosine': cosine_beta_schedule,
    'sigmoid': sigmoid_beta_schedule,
    'vp': vp_beta_schedule
}
                     
class DiffusionAgent(BaseAgent):
    def __init__(
        self, 
        policy: torch.nn.Module, 
        schedule: str = 'cosine',
        num_timesteps: int = 5,
        ema: float = 1e-3,
        device: str = 'cpu'
    ):
        super().__init__(policy, None, None, ema=ema)
        
        self.device = policy.device
        if schedule not in SCHEDULE.keys():
            raise ValueError(
                f"Invalid schedule '{schedule}'. Expected one of: {list(SCHEDULE.keys())}"
            )
        
        self.schedule = SCHEDULE[schedule]
        
        self.num_timesteps = num_timesteps
        self.betas = self.schedule(self.num_timesteps).to(self.device)
        self.alphas = (1 - self.betas).to(self.device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(self.device)

        self.to(device)
        
    def forward(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        cond: Optional[torch.Tensor] = None, 
        z: Optional[torch.Tensor] = None, 
        from_target: bool = False
    ) -> torch.Tensor:
        """
        predict the noise
        """
        if from_target:
            return self.policy_target(xt, t, cond, z)
        return self.policy(xt, t, cond, z)
    
    def predict_noise(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        predict the noise
        """
        noise_pred = self.policy(xt, t, cond, z)
        return noise_pred
    
    def policy_loss(
        self, 
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        calculate ddpm loss
        '''
        batch_size = x0.shape[0]
        
        noise = torch.randn_like(x0, device=self.device)
        t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
        
        xt = self.q_sample(x0, t, noise)
        
        noise_pred = self.predict_noise(xt, t, cond)
        loss = (((noise_pred - noise) ** 2).sum(axis = -1)).mean()
        
        return loss
    
    def policy_loss_with_weight(
        self, 
        weight: torch.Tensor, 
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''
        calculate ddpm loss
        '''
        batch_size = x0.shape[0]
        
        noise = torch.randn_like(x0, device=self.device)
        t = torch.randint(0, self.num_timesteps, (batch_size, ), device=self.device)
        
        xt = self.q_sample(x0, t, noise)
        
        noise_pred = self.predict_noise(xt, t, cond, z)
        
        return (((noise_pred - noise) ** 2).sum(axis = -1) * weight).mean()
            
    def q_sample(
        self, 
        x0: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        sample noisy xt from x0, q(xt|x0), forward process
        """
        alphas_cumprod_t = self.alphas_cumprod[t]
        xt = x0 * extract(torch.sqrt(alphas_cumprod_t), x0.shape) \
            + noise * extract(torch.sqrt(1 - alphas_cumprod_t), x0.shape)
        return xt
    
    def p_sample(
        self, 
        xt: torch.Tensor, 
        t: torch.Tensor, 
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        clip_sample: bool = False,
        ddpm_temperature: float = 1., 
        from_target: bool = False
    ) -> torch.Tensor:
        """
        sample xt-1 from xt, p(xt-1|xt)
        """
        noise_pred = self.forward(xt, t, cond, z, from_target=from_target)
        
        alpha1 = 1 / torch.sqrt(self.alphas[t])
        alpha2 = (1 - self.alphas[t]) / (torch.sqrt(1 - self.alphas_cumprod[t]))
        
        xtm1 = alpha1 * (xt - alpha2 * noise_pred)
        
        noise = torch.randn_like(xtm1, device=self.device) * ddpm_temperature
        xtm1 = xtm1 + (t > 0) * (torch.sqrt(self.betas[t]) * noise)
        
        if clip_sample:
            xtm1 = torch.clip(xtm1, -1., 1.)
        return xtm1
    
    def ddpm_sampler(
        self, 
        shape: Tuple, 
        cond: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        from_target: bool = False
    ) -> torch.Tensor:
        """
        sample x0 from xT, reverse process
        """
        x = torch.randn(shape, device=self.device)

        if len(shape) == 3:
            cond = cond.unsqueeze(1).repeat_interleave(shape[1], dim=1)
            z = z.unsqueeze(1).repeat_interleave(shape[1], dim=1)
            
            for t in reversed(range(self.num_timesteps)):
                x = self.p_sample(
                    xt=x, 
                    t=torch.full((shape[0], shape[1], 1), t, device=self.device), 
                    cond=cond, 
                    z=z, 
                    from_target=from_target
                )
        else:
            cond = cond.repeat(x.shape[0], 1)
            z = z.repeat(x.shape[0], 1)

            for t in reversed(range(self.num_timesteps)):
                x = self.p_sample(
                    xt=x, 
                    t=torch.full((shape[0], 1), t, device=self.device), 
                    cond=cond, 
                    z=z, 
                    from_target=from_target
                )
        return x

    def get_action(
        self, 
        state: torch.Tensor, 
        z: torch.Tensor,
        num: int = 1, 
        batch_input: bool = False, 
        from_target: bool = True
    ) -> torch.Tensor:
        if batch_input:
            return self.ddpm_sampler(
                (state.shape[0], num, self.policy.output_dim), 
                cond=state, 
                z=z, 
                from_target=from_target
            )
        
        return self.ddpm_sampler(
            (num, self.policy.output_dim), 
            cond=state, 
            z=z, 
            from_target=from_target
        )

       