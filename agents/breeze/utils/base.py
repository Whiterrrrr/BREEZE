import abc
from typing import Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
from agents.base import AbstractMLP
from agents.utils import weight_init
from agents.fb.base import AbstractPreprocessor

class V_net(AbstractMLP):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
        layernorm = True
    ):
        super().__init__(
            input_dimension=input_dimension,
            output_dimension=1,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            layernorm=layernorm
        )
        self.apply(weight_init)

    def forward(self, observation: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        V = self.trunk(torch.cat([observation, z], dim=-1))
        return V 


class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = True):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if affine else 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim = -1) * self.gamma * self.scale
    

class SelfAttention(nn.Module):
    def __init__(self, z_dim: int):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(z_dim, z_dim)
        self.key = nn.Linear(z_dim, z_dim)
        self.value = nn.Linear(z_dim, z_dim)
        self.z_dim = z_dim
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.query(x)
        K = self.key(x)   
        V = self.value(x) 

        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.z_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.bmm(attention_weights, V)
        return output
    
    
class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):

        super().__init__()

        inner_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = RMSNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x))


class ForwardRepresentation2(nn.Module):

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
        device: torch.device,
    ):
        super().__init__()
        
        self.z_dimension = z_dimension
        self.device = device
        
        # Initialize preprocessors
        self.obs_action_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=action_length,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )

        self.obs_z_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=z_dimension,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )

        self.feedforward_1 = FeedForward(preprocessor_feature_space_dimension)
        self.self_attention_1 = SelfAttention(preprocessor_feature_space_dimension)
        self.linear_11 = nn.Linear(
            preprocessor_feature_space_dimension * 2, 
            forward_hidden_dimension
        )
        self.linear_12 = nn.Linear(forward_hidden_dimension, z_dimension)
        
        self.feedforward_2 = FeedForward(preprocessor_feature_space_dimension)
        self.self_attention_2 = SelfAttention(preprocessor_feature_space_dimension)
        self.linear_21 = nn.Linear(
            preprocessor_feature_space_dimension * 2,
            forward_hidden_dimension
        )
        self.linear_22 = nn.Linear(forward_hidden_dimension, z_dimension)
        
        # Regularization components
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm_1 = nn.LayerNorm(preprocessor_feature_space_dimension)
        self.layer_norm_2 = nn.LayerNorm(preprocessor_feature_space_dimension)

        self.to(device)

    def forward(
        self, 
        observation: torch.Tensor, 
        action: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.
        
        Args:
            observation: Input observation tensor
            action: Optional action tensor
            z: Optional latent z tensor  
            
        Returns:
            Tuple of two output tensors from the dual processing path
        """

        # Process observation-action pairs
        obs_action_input = torch.cat([observation, action], dim=-1)
        obs_action_embedding = self.obs_action_preprocessor(
            obs_action_input
        ).unsqueeze(1)


        # Process observation-z pairs
        obs_z_input = torch.cat([observation, z], dim=-1)
        obs_z_embedding = self.obs_z_preprocessor(obs_z_input).unsqueeze(1)

        # Combine embeddings for processing
        combined_embeddings = torch.cat(
            [obs_z_embedding, obs_action_embedding], 
            dim=1
        )

        # First processing block
        attended_1 = self.self_attention_1(combined_embeddings)
        residual_1 = attended_1 + self.feedforward_1(attended_1)
        normalized_1 = self.layer_norm_1(self.dropout(residual_1))
        
        # Second processing block  
        attended_2 = self.self_attention_2(normalized_1)
        residual_2 = attended_2 + self.feedforward_2(attended_2)
        normalized_2 = self.layer_norm_2(self.dropout(residual_2))
        
        # Flatten and project to output space
        flattened_features = normalized_2.flatten(start_dim=1)
        
        # Dual output pathways
        F1 = self.linear_12(self.linear_11(flattened_features))
        F2 = self.linear_22(self.linear_21(flattened_features))
        
        return F1, F2


class TransformerFull(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transformer = torch.nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class AbstractFullTransformer(torch.nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for full transformer networks."""

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        d_model: int,
        nhead: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        device: torch.device,
        dropout: float = 0.1,
        preprocessor: bool = False,
    ):
        super().__init__()
        self._input_dimension = input_dimension
        self._output_dimension = output_dimension
        self._d_model = d_model
        self.device = device
        self._preprocessor = preprocessor

        # Input projection
        self.input_proj = torch.nn.Linear(input_dimension, d_model)
        
        # Positional encodings for encoder and decoder
        self.pos_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.GELU()
        )
        self.pos_decoder = torch.nn.Sequential(
            torch.nn.Linear(1, d_model),
            torch.nn.GELU()
        )
        
        # Full Transformer
        self.transformer = TransformerFull(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = torch.nn.Linear(d_model, output_dimension)
        self.to(device)


class BackwardTransformer(AbstractFullTransformer):
    """Backwards model implemented with full Transformer architecture."""
    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        d_model: int = 256,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(
            input_dimension=observation_length,
            output_dimension=z_dimension,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=hidden_layers,
            num_decoder_layers=hidden_layers,
            dim_feedforward=hidden_dimension,
            device=device,
            dropout=dropout,
        )
        self._z_dimension = z_dimension
        self.dropout = torch.nn.Dropout(dropout)
        
        # Learnable query for the decoder
        self.query_embed = torch.nn.Parameter(torch.randn(1, d_model))
        self.to(device)

    def forward(self, observation: torch.Tensor, position_encoding: bool = False) -> torch.Tensor:
        """
        Takes observation and processes it through full transformer architecture.
        Args:
            observation: state tensor of shape [batch_dim, observation_length]
        Returns:
            z: embedded tensor of shape [batch_dim, z_dimension]
        """
        batch_size = observation.shape[0]
        
        if position_encoding:
            # Create position encodings for encoder
            src_positions = torch.arange(observation.shape[1], dtype=torch.float32)\
                .expand(batch_size, -1).unsqueeze(-1).to(self.device)
            src_pos_encoding = self.pos_encoder(src_positions)

            # Project and add positional encoding for encoder input
            memory = self.input_proj(observation)
            memory = memory.unsqueeze(1)
            memory = memory + src_pos_encoding
        else:
            x = observation.unsqueeze(1).to(self.device)
            memory = self.input_proj(x)
        
        # Create decoder query
        query = self.query_embed.expand(batch_size, -1, -1)
        
        # Create position encoding for decoder
        tgt_positions = torch.zeros(batch_size, 1, 1, dtype=torch.float32).to(self.device)
        tgt_pos_encoding = self.pos_decoder(tgt_positions)
        query = query + tgt_pos_encoding

        # Generate target mask for decoder
        tgt_mask = torch.zeros((1, 1), dtype=torch.float32).to(self.device)
        
        # Pass through transformer
        output = self.transformer.transformer(
            src=memory,
            tgt=query,
            tgt_mask=tgt_mask
        )
        
        # Project to output dimension
        z = self.output_proj(output.squeeze(1))

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = torch.sqrt(
            torch.tensor(self._z_dimension, dtype=torch.float32, device=self.device)
        ) * torch.nn.functional.normalize(z, dim=1)

        return z