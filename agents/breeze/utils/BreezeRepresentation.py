import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Tuple
from agents.utils import weight_init
from agents.breeze.utils.base import (
    BackwardTransformer,
    AttentionForwardRepresentation
)

class AttentionBackwardRepresentation(torch.nn.Module):
    """Backward representation network."""

    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        device: torch.device,
    ):
        super().__init__()

        self.B = BackwardTransformer(
            observation_length=observation_length,
            z_dimension=z_dimension,
            hidden_dimension=backward_hidden_dimension,
            hidden_layers=backward_hidden_layers,
            device=device,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Estimates routes to observation via backwards model."""

        return self.B(observation)

class AttentionForwardBackwardRepresentation(torch.nn.Module):
    """Combined Forward-backward representation network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        orthonormalisation_coefficient: float,
        discount: float,
        device: torch.device,
    ):
        super().__init__()

        self.forward_representation = AttentionForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            device=device,
        )

        self.backward_representation = AttentionBackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
        )

        self.forward_representation_target = AttentionForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            device=device,
        )

        self.backward_representation_target = AttentionBackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
        )
            
        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device


class BreezeRepresentation(AttentionForwardBackwardRepresentation):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        forward_config: dict,
        backward_config: dict,
        z_dimension: int,
        orthonormalisation_coefficient: float,
        device: torch.device,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=forward_config["preprocessor"][
                "hidden_dimension"
            ],
            preprocessor_feature_space_dimension=forward_config["preprocessor"][
                "output_dimension"
            ],
            preprocessor_hidden_layers=forward_config["preprocessor"]["hidden_layers"],
            preprocessor_activation=forward_config["preprocessor"]["activation"],
            forward_hidden_dimension=forward_config["hidden_dimension"],
            backward_hidden_dimension=backward_config["hidden_dimension"],
            backward_hidden_layers=backward_config["hidden_layers"],
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            z_dimension=z_dimension,
            discount=None,
            device=device,
        )
        
        self.apply(weight_init)
        self._load_state()
        self.optimizer, self.scheduler = self._build_optimizer(forward_config, backward_config)

    def _load_state(self):
        """Load the state of the target networks.
        """
        self.forward_representation_target.load_state_dict(
            self.forward_representation.state_dict()
        )
        self.backward_representation_target.load_state_dict(
            self.backward_representation.state_dict()
        )

    def _build_optimizer(
        self,
        forward_config: dict,
        backward_config: dict,
    ) -> Tuple[
        torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    ]:
        """Build the optimizer for the model.
        Args:
            forward_backward_config (ForwardBackwardConfig): Parameters for the forward and backward representations

        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts]: build the common optimizer for F, B, and O, and the scheduler
        """

        critic_learning_rate = forward_config["critic_learning_rate"]
        backward_lr = (
            critic_learning_rate * backward_config["learning_rate_coefficient"]
        )
        base_params = [
            {"params": self.forward_representation.parameters()},
            {
                "params": self.backward_representation.parameters(),
                "lr": backward_lr,
            },
        ]

        optimizer = torch.optim.AdamW(
            base_params,
            lr=critic_learning_rate,
            weight_decay=0.03,
            betas=(0.9, 0.99),
            amsgrad=False,
        ) 

        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-5
        )  # NOTE: added from original code

        return optimizer, scheduler
