from agents.fb.agent import FB
import math
from typing import Tuple, Dict
import torch
import numpy as np
import torch.nn.functional as F
from agents.utils import EPS
from agents.base import Batch
from agents.breeze.utils.losses import *
from agents.breeze.utils.BreezeRepresentation import BreezeRepresentation
from agents.breeze.utils.model import Actor, Vfunc


class BREEZE(FB):

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        forward_config: dict,
        backward_config: dict,
        actor_config: dict,
        v_config: dict,
        code_options_parameters: dict,
        orthonormalisation_coefficient: float,
        z_dimension: int,  # 50
        batch_size: int,
        z_mix_ratio: float,
        tau: float,
        device: torch.device,
        discount = 0.98,
        s_mean: torch.Tensor = torch.tensor([0]),
        a_mean: torch.Tensor = torch.tensor([0]),
        s_std: torch.Tensor = torch.tensor([1]),
        a_std: torch.Tensor = torch.tensor([1]),
        name: str = 'breeze'
    ):
        super(FB, self).__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name
        )
        self.a_mean, self.a_std = a_mean, a_std
        self.s_mean, self.s_std = s_mean, s_std
        self.encoder = torch.nn.Identity()
        self.augmentation = torch.nn.Identity()
        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self.discount = discount
        self._z_dimension = z_dimension
        self.std_dev_schedule = actor_config["std_dev_schedule"]
        self.off_diagonal = ~torch.eye(batch_size, device=device, dtype=torch.bool)  # future states =/= s_{t+1}
        self.future_weight = code_options_parameters['future_weight']
        self.guide_weight = code_options_parameters['guide_weight']
        self.expectile = code_options_parameters["expectile"]
        self.freg_coef = code_options_parameters['freg_coef']
        self.ktrain = code_options_parameters['ktrain']
        self.keval = code_options_parameters['keval']

        self.FB = BreezeRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            forward_config=forward_config,
            backward_config=backward_config,
            z_dimension=z_dimension,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            device=device,
        )

        self.actor = Actor(
            observation_length=observation_length,
            action_length=action_length,
            z_dimension=z_dimension,
            actor_config=actor_config,
            device=device,
        )

        self.V = Vfunc(
            input_dimension=observation_length + z_dimension,
            v_config=v_config,
            device=device,
        )

    @torch.no_grad()
    def act(
        self,
        observation: Dict[str, np.ndarray],
        task: np.array,
        step: int,
        sample: bool = False,
    ) -> Tuple[np.array, float]:
        """
        Used at test time to perform zero-shot rollouts.
        Takes observation array from environment, encodes, and selects
        action from actor.
        Args:
            observation: observation array of shape [observation_length]
            task: task array of shape [z_dimension]
            step: current step in env
            sample: whether to sample action from actor distribution
        Returns:
            action: action array of shape [action_length]
            std_dev: current actor standard deviation
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        observation_norm = (observation - self.s_mean) / (self.s_std + EPS)
        h = self.encoder(observation_norm)
        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)

        with torch.no_grad():
            action_norm = self.actor.get_action(h, z, num=self.keval, from_target=True)
            Q = self.predict_q(
                observation=observation_norm.repeat(self.keval, 1),
                z=z.repeat(self.keval, 1),
                action=action_norm, 
            )
            action_norm = action_norm[torch.argmax(Q)].unsqueeze(0)
                
        action = action_norm * (self.a_std + EPS) + self.a_mean
        return action.detach().cpu().numpy(), None

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
        Returns:
            metrics: dictionary of metrics for logging
        """
        zs = self.sample_z(size=self.batch_size)
        perm = torch.randperm(self.batch_size)
        backward_input = batch.observations[perm]

        # Mix z vectors with backward representations
        mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
        with torch.no_grad():
            mix_zs = self.FB.backward_representation(
                backward_input[mix_indices]
            ).detach()
            mix_zs = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                mix_zs, dim=1
            )

        zs[mix_indices] = mix_zs
        actor_zs = zs.clone().requires_grad_(True)
        actor_observations = batch.observations.clone().requires_grad_(True)
        actor_actions = batch.actions.clone().requires_grad_(True) # NOTE: actions used for IQL loss computation

        FB_metrics = self.update_fb(
            observations=batch.observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            next_actions=batch.next_actions,
            discounts=torch.ones_like(batch.discounts).to(batch.discounts.device) * self.discount,
            zs=zs,
        )
        actor_metrics = self.update_actor(
            observation=actor_observations,
            action=actor_actions,
            z=actor_zs,
        )  

        self.FB.scheduler.step()
        current_lr = self.FB.scheduler.get_last_lr()[0]

        self.update_representation_target_networks()
        self.actor.ema_update_policy()

        metrics = {
            **FB_metrics,
            **actor_metrics,
        }
        metrics["current_lr"] = current_lr
        return metrics

    def update_fb(
        self,
        observations: torch.Tensor,  # [batch_size, observation_length]
        actions: torch.Tensor,  # [batch_size, action_length]
        next_observations: torch.Tensor,  # [batch_size, observation_length]
        next_actions: torch.Tensor,
        discounts: torch.Tensor,  # [batch_size]
        zs: torch.Tensor,  # [batch_size, z_dimension]
    ) -> Dict[str, float]:
        """
        Updates the forward-backward representation network.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """

        total_loss, metrics = self._update_fb_inner(
            observations,
            actions,
            next_observations,
            next_actions, 
            discounts,
            zs,
        )

        # Optimize FB networks
        self.FB.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        
        # Gradient clipping
        for param in self.FB.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.FB.optimizer.step()

        metrics = {
            **metrics,
            "loss/forward_backward_total_loss": total_loss,
        }
        return metrics

    def _update_fb_inner(
        self,
        observations: torch.Tensor,  # [batch_size, observation_length]
        actions: torch.Tensor,  # [batch_size, action_length]
        next_observations: torch.Tensor,  # [batch_size, observation_length]
        next_actions: torch.Tensor, 
        discounts: torch.Tensor,  # [batch_size]
        zs: torch.Tensor,  # [batch_size, z_dimension]
    ) -> Tuple[int, Dict[str, float]]:
        """
        Loss computation common to FB and all child classes. All equation references
        are to the appendix of the FB paper (Touati et. al (2022)). TODO modify these references if paper published
        The loss contains several components:
            1. Forward-backward representation loss: a Bellman update on the successor
                measure (equation 24, Appendix B)
            2. Orthonormalisation loss: constrains backward function such that the
                measure of state s from state s = 1 (equation 26, Appendix B)
            3. Regularization loss: additional loss term to regularize the representation
            Note: Q loss (Equation 9) is not implemented.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            total_loss: total loss for FB
            metrics: dictionary of metrics for logging
        """
        with torch.no_grad():
            mix_indices = np.where(np.random.rand(self.batch_size) < self.future_weight)[0]
            input_observation, input_zs = next_observations[mix_indices], zs[mix_indices]

            diffusion_output = self.actor.get_action(
                input_observation, input_zs, num=self.ktrain, batch_input=True, from_target=True
            ).squeeze()

            # Rejection Sampling: Select best actions using Q-values
            if self.ktrain > 1:
                Q_sample = self.predict_q(
                    observation=input_observation.unsqueeze(1).repeat(1, self.ktrain, 1).view(-1, input_observation.shape[-1]),
                    z=input_zs.unsqueeze(1).repeat(1, self.ktrain, 1).view(-1, input_zs.shape[-1]),
                    action=diffusion_output.view(-1, diffusion_output.shape[-1])
                )
                _, max_indices = torch.max(Q_sample.view(input_observation.shape[0], self.ktrain), dim=1)
                next_actions[mix_indices] = diffusion_output[torch.arange(input_observation.shape[0]), max_indices]
            else:
                next_actions[mix_indices] = diffusion_output
            
            # Compute next state values
            V_next = self.V(next_observations, zs).squeeze()
                
        with torch.no_grad():
            target_M = self.compute_target_matrices(next_observations, zs, next_actions.detach())

        B_next = self.FB.backward_representation(next_observations)
        F1, F2 = self.FB.forward_representation(observations, actions, zs)

        M1_next = torch.einsum("sd, td -> st", F1, B_next)
        M2_next = torch.einsum("sd, td -> st", F2, B_next)

        fb_loss, fb_diag_loss, fb_off_diag_loss = forward_backward_loss(
            M1_next,
            M2_next,
            target_M,
            discounts,
            self.off_diagonal,
        )

        # Value function and Q-learning losses
        cov = torch.matmul(B_next.T, B_next) / B_next.shape[0]
        inv_cov = torch.inverse(cov)
        implicit_reward = (torch.matmul(B_next, inv_cov) * zs).sum(dim=1)
        target_Q = implicit_reward + self.discount * V_next.detach()
            
        Q1, Q2 = torch.einsum("sd, sd -> s", F1, zs), torch.einsum("sd, sd -> s", F2, zs)
        Q = torch.min(Q1, Q2)
        V = self.V(observations, zs).squeeze()
        
        q_loss = sum(F.mse_loss(q, target_Q) for q in [Q1, Q2]) / 2
        v_loss = asymmetric_l2_loss(Q.detach() - V, self.expectile)

        # Update value function
        self.V.optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.V.optimizer.step()
        
        regularization_loss = q_loss * self.freg_coef

        # Orthonormality loss 
        ortho_loss, ortho_loss_diag, ortho_loss_off_diag = orthogonality_loss(
            B_next, self.off_diagonal, self.FB.orthonormalisation_coefficient, self._device
        )

        total_loss = fb_loss + ortho_loss + regularization_loss

        metrics = {
            "loss/forward_backward_total_loss": total_loss,
            "loss/forward_backward_fb_loss": fb_loss,
            "loss/forward_backward_fb_diag_loss": fb_diag_loss,
            "loss/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
            "loss/ortho_diag_loss": ortho_loss_diag,
            "loss/ortho_off_diag_loss": ortho_loss_off_diag,
            "train/F1": F1.mean().item(),
            "train/B_next": B_next.mean().item(),
            "value/M1_next": M1_next.mean().item(),
            "loss/regularization_loss": regularization_loss,
            "value/V": V.mean().item(),
            "value/Q": Q.mean().item(),
            "loss/V_loss": v_loss.mean().item(),
        }
        return (
            total_loss,
            metrics,
        )

    def update_actor(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, float]:
        """Update actor network using weighted regression.
        
        Args:
            observation: Batch of observations [batch_size, obs_dim]
            action: Batch of actions [batch_size, action_dim]  
            z: Batch of task embeddings [batch_size, z_dim]
            
        Returns:
            metrics: Actor training metrics
        """
        Q = self.predict_q(observation, z, action)  
        V = self.V(observation, z).squeeze()
        
        # Compute weights
        weight = torch.exp(self.guide_weight * (Q - V)).clamp(max=100)
        actor_loss = self.actor.policy_loss_with_weight(weight, action, observation, z)
                
        # Optimize actor
        self.actor.optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        
        # Gradient clipping
        for param in self.actor.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
                
        self.actor.optimizer.step()

        metrics = {
            "loss/actor_loss": actor_loss.item(),
            "value/actor_Q": Q.mean().item(),
            "value/actor_V": V.mean().item(),
            "value/weight": weight.mean().item(),
        }

        return metrics

    def predict_q(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict Q-values for state-action-task tuples.
        
        Args:
            observation: Observations [N, obs_dim]
            z: Task embeddings [N, z_dim]
            action: Actions [N, action_dim]
            
        Returns:
            Q-values [N] using double Q-learning
        """
        F1, F2 = self.FB.forward_representation(
            observation=observation, z=z, action=action
        )
        
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)

        return torch.min(Q1, Q2)

    def compute_target_matrices(
        self,
        next_observations: torch.Tensor,
        zs: torch.Tensor,
        next_actions: torch.Tensor,
    ) -> {torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor}:
        """
        Compute all target matrices
        Args:
            next_observations: tensor of shape [N, observation_length]
            observations_rand: tensor of shape [N, observation_length]
            zs: tensor of shape [N, z_dimension]
            next_actions: tensor of shape [N, action_length]
            discounts: tensor of shape [N]
        """
        target_B = self.FB.backward_representation_target(
            observation=next_observations
        )
        target_F1, target_F2 = self.FB.forward_representation_target(
            observation=next_observations, z=zs, action=next_actions
        )

        target_M1 = torch.einsum("sd, td -> st", target_F1, target_B)
        target_M2 = torch.einsum("sd, td -> st", target_F2, target_B)

        return torch.min(target_M1, target_M2)

    def update_representation_target_networks(self) -> None:
        """Perform soft updates of target networks."""
        self.soft_update_params(
            network=self.FB.forward_representation,
            target_network=self.FB.forward_representation_target,
            tau=self._tau,
        )
        
        self.soft_update_params(
            network=self.FB.backward_representation,
            target_network=self.FB.backward_representation_target,
            tau=self._tau,
        )