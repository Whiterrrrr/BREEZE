import torch
from agents.breeze.utils.base import V_net
from agents.breeze.utils.diffusion_utils import *

class Actor:
    """Wrapping class that builds an actor model from its parameter calling the appropriate actor class."""

    def __new__(
        cls,
        observation_length,
        action_length,
        z_dimension,
        actor_config,
        device,
    ):
        policy = IDQLDiffusion(
            input_dim=action_length,
            output_dim=action_length,
            cond_dim=observation_length,
            z_dim=z_dimension,
            time_embeding=actor_config['time_embeding'],
            hidden_dim=actor_config["hidden_dimension"],
            device=device,
        )
        actor = DiffusionAgent(
            policy=policy,
            schedule=actor_config['beta'],
            num_timesteps=actor_config['ts'],
            ema=actor_config['ema'],
            device=device
        )

        actor.optimizer = Actor._build_optimizer(actor, actor_config["learning_rate"])
        return actor

    @staticmethod
    def _build_optimizer(actor, learning_rate) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=learning_rate,
        )
        return optimizer

class Vfunc:
    """Wrapping class that builds an Value model."""

    def __new__(
        cls,
        input_dimension,
        v_config,
        device,
    ):
        v = V_net(
            input_dimension=input_dimension,
            hidden_dimension=v_config['hidden_dimension'],
            hidden_layers=v_config['hidden_layers'],
            activation=v_config['activation'],
            device=device
        )

        v.optimizer = Vfunc._build_optimizer(v, v_config["learning_rate"])
        return v

    @staticmethod
    def _build_optimizer(v, learning_rate) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(
            v.parameters(),
            lr=learning_rate,
            weight_decay=0.03,
            betas=(0.9, 0.99),
            amsgrad=False
        )
        return optimizer