import torch
from typing import Tuple

def forward_backward_loss(
    M1_next: torch.Tensor,
    M2_next: torch.Tensor,
    target_M: torch.Tensor,
    discounts: torch.Tensor,
    off_diagonal: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the loss for the forward and backward models
    """
    # diagonal/off diagonal method mentioned here:
    # https://github.com/facebookresearch/controllable_agent/issues/4
    off_diag_loss = 0.5 * sum(
        (M - discounts * target_M)[off_diagonal].pow(2).mean()
        for M in [M1_next, M2_next]
    )
    diag_loss = -sum(M.diag().mean() for M in [M1_next, M2_next])
    loss = diag_loss + off_diag_loss

    return loss, diag_loss, off_diag_loss


def orthogonality_loss(
    B_next: torch.Tensor, off_diagonal: torch.Tensor, orthogonality_loss_coefficient: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enable a fixed be value to facilitate convergence
    """
    covariance = torch.matmul(B_next, B_next.T)
    ortho_loss_diag = -2 * covariance.diag().mean()
    ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
    ortho_loss = orthogonality_loss_coefficient * (
        ortho_loss_diag + ortho_loss_off_diag
    )
    return ortho_loss, ortho_loss_diag, ortho_loss_off_diag


def asymmetric_l2_loss(u: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    IQL regularization loss
    Compute the asymmetric L2 tau loss of the gap between prediction and target u.
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def sql_loss(M1_rand: torch.Tensor, M2_rand: torch.Tensor, O: torch.Tensor, alpha: int) -> torch.Tensor:
    """
    SQL loss from the paper "OFFLINE RL WITH NO OOD ACTIONS: IN-SAMPLE  LEARNING VIA IMPLICIT VALUE REGULARIZATION"
    """
    sp_term = (torch.min(M1_rand, M2_rand).detach() - O) / (2 * alpha) + 1.0
    sp_weight = torch.where(sp_term > 0, torch.tensor(1.0), torch.tensor(0.0))
    loss = (sp_weight * (sp_term**2) + O / alpha).mean()

    return loss


def eql_loss(M1_rand: torch.Tensor, M2_rand: torch.Tensor, O: torch.Tensor, alpha:int) -> torch.Tensor:
    """
    EQL loss from the paper "OFFLINE RL WITH NO OOD ACTIONS: IN-SAMPLE LEARNING VIA IMPLICIT VALUE REGULARIZATION"
    """
    sp_term = (torch.min(M1_rand, M2_rand).detach() - O) / alpha
    sp_term = torch.minimum(sp_term, torch.tensor(5.0))
    max_sp_term = torch.max(sp_term, dim=0).values
    max_sp_term = torch.where(max_sp_term < -1.0, torch.tensor(-1.0), max_sp_term)
    max_sp_term = max_sp_term.detach()
    loss = (
        torch.exp(sp_term - max_sp_term) + torch.exp(-max_sp_term) * O / alpha
    ).mean()

    return loss
