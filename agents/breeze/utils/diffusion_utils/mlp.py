from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

AC_FN ={'relu': F.relu, 'mish': F.mish, 'gelu': F.gelu}

class MLP(nn.Module):
    def __init__(
        self, 
        input_size : int, 
        hidden_sizes : list, 
        output_size : int, 
        ac_fn: str = 'relu', 
        use_layernorm: bool = False, 
        dropout_rate: float = 0.
    ):
        super().__init__()             
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout_rate
        
        # initialize layers
        self.layers = nn.ModuleList()
        self.layernorms = nn.ModuleList() if use_layernorm else None
        self.ac_fn = AC_FN[ac_fn]
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
                
        if self.use_layernorm:
            self.layernorms.append(nn.LayerNorm(input_size))
                
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

            
    def forward(self, x: torch.Tensor):
        if self.use_layernorm:
            x = self.layernorms[-1](x)
        
        for layer in self.layers[:-1]:
            x = layer(x)
            if self.dropout_rate > 0:
                x = self.dropout(x)
            x = self.ac_fn(x)

        x = self.layers[-1](x)
        return x
    

class MLPResNetBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim : int, 
        ac_fn: str ='relu', 
        use_layernorm: bool = False, 
        dropout_rate: int = 0.1, 
        condition_dim: Optional[torch.Tensor] = None,
    ):

        super(MLPResNetBlock, self).__init__()

        self.use_layernorm = use_layernorm
        self.dropout = nn.Dropout(dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ac_fn = AC_FN[ac_fn]
        self.dense2 = nn.Linear(hidden_dim * 4, hidden_dim)
        
        self.condition_dim = condition_dim
        if condition_dim is not None:
            self.film_gamma = nn.Linear(condition_dim, hidden_dim * 4)
            self.film_beta = nn.Linear(condition_dim, hidden_dim * 4)
            nn.init.zeros_(self.film_gamma.weight)
            nn.init.zeros_(self.film_gamma.bias)
            nn.init.zeros_(self.film_beta.weight)
            nn.init.zeros_(self.film_beta.bias)
            
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        identity = x
        
        out = self.dropout(x)
        if self.use_layernorm:
            out = self.norm1(out)
        out = self.dense1(out)
        out = self.ac_fn(out)
        
        if self.condition_dim is not None:
            assert condition is not None, "give condition"
            gamma = self.film_gamma(condition)
            beta = self.film_beta(condition)
            out = gamma * out + beta
            
        out = self.dense2(out)
        return identity + out
    
class MLPResNet(nn.Module):
    def __init__(
            self, 
            num_blocks : int, 
            input_dim : int, 
            hidden_dim : int, 
            output_size : int, 
            ac_fn: str = 'relu', 
            use_layernorm: bool = True, 
            dropout_rate: float = 0.1, 
            condition_dim: Optional[torch.Tensor] = None,
        ):

        super(MLPResNet, self).__init__()
        
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.ac_fn = AC_FN[ac_fn]
        self.dense2 = nn.Linear(hidden_dim, output_size)
        self.mlp_res_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.mlp_res_blocks.append(
                MLPResNetBlock(hidden_dim, ac_fn, use_layernorm, dropout_rate, condition_dim)
            )
            
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None):
        out = self.dense1(x)
        for mlp_res_block in self.mlp_res_blocks:
            out = mlp_res_block(out, condition=condition)
        out = self.ac_fn(out)
        return self.dense2(out)