import math
from typing import Optional

import torch
import torch.nn as nn
from .mlp import MLP, MLPResNet

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size
            
    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.output_size // 2
        f = math.log(10000) / (half_dim - 1)
        f = torch.exp(torch.arange(half_dim, device=device) * -f)
        f = x * f[None, :]
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f

# learned positional embeds
class LearnedPosEmb(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.output_size = output_size
        self.kernel = nn.Parameter(torch.randn(output_size // 2, input_size) * 0.2)
            
    def forward(self, x: torch.Tensor):
        f = 2 * torch.pi * x @ self.kernel.T
        f = torch.cat([f.cos(), f.sin()], axis=-1)
        return f       

TIMEEMBED = {"fixed": SinusoidalPosEmb, "learned": LearnedPosEmb}

class IDQLDiffusion(nn.Module):
    """
    Diffusion model implementation for IDQL (Implicit Diffusion Q-Learning).
    
    Reference: 
        IDQL: Implicit Diffusion Q-Learning - arXiv:2304.10573
    """
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        cond_dim: int = 0,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        time_dim: int = 64,
        ac_fn: str = 'mish',
        time_embeding: str = 'fixed',
        device: str = 'cpu',
        z_dim: Optional[torch.Tensor] = None
    ):
        super(IDQLDiffusion, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        
        # time embedding
        if time_embeding not in TIMEEMBED.keys():
            raise ValueError(
                f"Invalid time_embedding '{time_embeding}'. Expected one of: {list(TIMEEMBED.keys())}"
            )
        
        self.time_process = TIMEEMBED[time_embeding](1, time_dim)
        self.time_encoder = MLP(time_dim, [128], 128, ac_fn='mish')
        
        # decoder
        decoder_input_dim = input_dim + 128 + cond_dim
        self.decoder = MLPResNet(
            num_blocks=num_blocks, 
            input_dim=decoder_input_dim, 
            hidden_dim=hidden_dim, 
            output_size=output_dim, 
            ac_fn=ac_fn, 
            use_layernorm=True, 
            dropout_rate=0.1, 
            condition_dim=z_dim
        )
        
        self.device = device
        self.to(device)
              
    def forward(
        self,
        x_t: torch.Tensor,
        time: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the diffusion model.
        """
        # Process time embedding
        if x_t.dim() == 3:
            time_embedding = self.time_process(time)
        else:
            time_embedding = self.time_process(time.view(-1, 1))
            
        time_embedding = self.time_encoder(time_embedding)
        
        # Concatenate conditioning if provided
        if condition is not None:
            x_t = torch.cat([x_t, condition], dim=-1)
        
        # Prepare input for decoder
        decoder_input = torch.cat([time_embedding, x_t], dim=-1)

        noise_pred = self.decoder(decoder_input, z)
        return noise_pred      