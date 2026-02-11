import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import numpy as np
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def get_sinusoidal_positional_embeddings(num_positions, d_model):
    position = torch.arange(num_positions).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(0)
    pos_embedding = torch.zeros(num_positions, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    return pos_embedding.unsqueeze(0)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        assert dim % 2 == 0, "Dimension must be even"
    def forward(self, t):
        t = t * 100.0
        t = t.unsqueeze(-1)
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        embedding = embedding.squeeze(1)
        return embedding

class SpatialTransformerlayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        mlp_ratio = 1.0
        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d_model, num_heads=1, qkv_bias=True)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

    def forward(self, x, c):
        # x: (B*T, N, C), c: (B*T, C)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Chattraffic(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.channel = 1
        self.H = 96
        self.W = args.num_nodes
        emb_size = getattr(args, 'chattraffic_emb_size', 128)
        n_layers = getattr(args, 'chattraffic_n_layers', 2)
        
        self.emb_size = emb_size
        self.input_proj = nn.Linear(1, emb_size)
        
        # spatial pos embed: only N positions instead of T*N
        pos_embed = get_sinusoidal_positional_embeddings(self.W, emb_size)
        self.pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)
        
        # temporal pos embed
        time_pos_embed = get_sinusoidal_positional_embeddings(self.H, emb_size)
        self.time_pos_embed = torch.nn.Parameter(time_pos_embed, requires_grad=False)
        
        self.ln = nn.LayerNorm(emb_size)
        self.output_proj = nn.Linear(emb_size, 1)

        self.time_emb = TimeEmbedding(dim=emb_size)
        self.layers = nn.ModuleList([SpatialTransformerlayer(d_model=emb_size) for _ in range(n_layers)])

        self.fc = nn.Linear(768, emb_size)
        self.initialize_weights()

    def forward(self, input: torch.Tensor, t: torch.Tensor, text_input):
        # input: (B, T, N)
        B, T, N = input.shape
        
        # (B, T, N) -> (B, T, N, 1) -> (B, T, N, C)
        x = self.input_proj(input.unsqueeze(-1))
        
        # add spatial pos embed: (1, N, C) broadcast to (B, T, N, C)
        x = x + self.pos_embed.unsqueeze(1)
        
        # add temporal pos embed: (1, T, C) -> (B, T, N, C)
        x = x + self.time_pos_embed.unsqueeze(2)
        
        # diffusion time embedding
        c = self.time_emb(t)  # (B, C)
        if text_input is not None:
            c = c + self.fc(text_input)
        
        # expand c for each timestep: (B, C) -> (B*T, C)
        c = c.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
        
        # reshape for spatial attention: (B, T, N, C) -> (B*T, N, C)
        x = x.reshape(B * T, N, self.emb_size)
        
        # spatial attention (only across N nodes, not T*N)
        for layer in self.layers:
            x = layer(x, c)
        
        # reshape back: (B*T, N, C) -> (B, T, N, C)
        x = x.reshape(B, T, N, self.emb_size)
        
        x = self.ln(x)
        x = self.output_proj(x).squeeze(-1)  # (B, T, N)

        return x

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
