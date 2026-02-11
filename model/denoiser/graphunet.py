import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import numpy as np
import os
import math
import torch.nn.functional as F
class linear(nn.Module):
    """Linear layer."""

    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_sinusoidal_positional_embeddings(num_positions, d_model):
    position = torch.arange(num_positions).unsqueeze(1)  # shape: (num_positions, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)).unsqueeze(
        0)  # shape: (1, d_model/2)

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
        t = t.float()
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

################################################
#               Embedding Layers               #
################################################

class LatentEmbedding(nn.Module):
    def __init__(self, embed_dim: int=64):
        super().__init__()
        self.dim = embed_dim
        self.embedding2d = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(6, 6),
            stride=(6, 6),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        B, _, M, N = x.shape
        x = self.embedding2d(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class InverseLatentEmbedding(nn.Module):
    def __init__(self, embed_dim: int=64):
        super().__init__()
        self.dim = embed_dim
        self.inv_embedding2d = nn.ConvTranspose2d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=(6, 6),
            stride=(6, 6),
        )
        self.fc1 = nn.Linear(60, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        B, K, C = x.shape
        x = x.transpose(1, 2).reshape(B, self.dim, 1, K)
        x = self.inv_embedding2d(x)
        x = x.squeeze(1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


#################################################################################
#                                 Core Model                                #
#################################################################################

class Transformerlayer(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        mlp_ratio = 4.0
        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d_model, num_heads=4, qkv_bias=True)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DIT(nn.Module):
    def __init__(self,args):
        super().__init__()
        # patchify
        self.channel=1
        self.H = 96 #6 # 30
        self.W = args.num_nodes
        
        emb_size=768 #64
        self.patch_size=1
        self.patch_count=int((self.H/self.patch_size)*(self.W/self.patch_size))
        self.conv=nn.Conv2d(in_channels=self.channel,out_channels=self.channel*self.patch_size**2,kernel_size=self.patch_size,padding=0,stride=self.patch_size)
        self.patch_emb=nn.Linear(in_features=self.channel*self.patch_size**2,out_features=emb_size)
        pos_embed = get_sinusoidal_positional_embeddings(self.patch_count, emb_size)
        self.pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)
        self.ln = nn.LayerNorm(emb_size)
        self.linear_emb_to_patch = nn.Linear(emb_size, self.channel * self.patch_size ** 2)


        self.time_emb = TimeEmbedding(dim=emb_size)
        # pos_embed = get_sinusoidal_positional_embeddings(6,64)
        # self.pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)

        self.layers = nn.ModuleList([Transformerlayer() for _ in range(2)])
        self.unpatch = InverseLatentEmbedding(embed_dim=emb_size)



        self.initialize_weights()



    def forward(self, input: torch.Tensor, t: torch.Tensor, text_input):
        """
                x: (B, M, N) tensor of input latent (batch, latent num:4, latent dim:64)
                t: (B,) tensor of diffusion timesteps
                text_input:
                """
        # x = input.permute(0, 2, 1)
        # x = x + self.pos_embed
        x = input.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (batch,new_channel,patch_count,patch_count)
        x = x.permute(0, 2, 3, 1)  # (batch,patch_count,patch_count,new_channel)
        # print(f"Before view: x.shape = {x.shape}, patch_count = {self.patch_count}")
        x = x.view(x.size(0), self.patch_count, x.size(3))  # (batch,patch_count**2,new_channel)
        x = self.patch_emb(x)  # (batch,patch_count**2,emb_size)
        x = x + self.pos_embed  # (batch,patch_count**2,emb_size)

        t = self.time_emb(t)

        c = t
        if text_input is not None:
            c = t + text_input
        for layer in self.layers:
            x = layer(x, c)

        x = self.ln(x)
        x = self.linear_emb_to_patch(x)
        x = x.view(x.size(0), int(self.H/self.patch_size), int(self.W/self.patch_size), self.channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5)  # (batch,channel,patch_count(H),patch_count(W),patch_size(H),patch_size(W))
        x = x.permute(0, 1, 2, 4, 3, 5)  # (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x = x.reshape(x.size(0), self.channel, self.H,
                      self.W)  # (batch,channel,img_size,img_size)
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)


        return x
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)





def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")
class nconv(nn.Module):
    """Graph conv operation."""

    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, act, p, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order + 1) * in_dim
        self.mlp = linear(c_in, out_dim)
        self.dropout = p
        self.order = order
        self.act = act

    def forward(self, x, support):
        # thanks for GraphWaveNet!! haha
        out = [x]
        for a in support:
            x1 = self.nconv(x, a.to(x.device))
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a.to(x.device))
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        
        if self.act is not None:
            h = self.act(h)
        
        return h



class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze(-1)
        scores = self.sigmoid(weights)
        if scores.dim() >2:
            scores = scores.mean(dim=1)
        return top_k_graph(scores, g, h, self.k)

def top_k_graph(scores, g, h, k):
    """Top-k graph pooling following torche reference implementation"""
    num_nodes = g.shape[0]
    
    k_nodes = max(2, min(int(k * num_nodes), num_nodes))
    
    batch_size, time_steps, nodes, channels = h.shape
    
    values, idx = torch.topk(scores, k_nodes, dim=1)
    
    idx_common = idx[0]
    new_h = h[:, :, idx_common, :]
    
    values = values[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    values = values.expand(batch_size, time_steps, -1, channels)
    new_h = torch.mul(new_h, values)
    
    if isinstance(g, np.ndarray):
        g = torch.FloatTensor(g).to(h.device)
    
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx_common, :][:, idx_common]
    g = norm_g(un_g)
    
    return g, new_h, idx_common

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g

class STDownsample(nn.Module):
    """
    A downsampling layer witorch an optional convolution.
    :param channels: channels in torche inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if torche signal is 1D, 2D, or 3D. If 3D, torchen
                 downsampling occurs in torche inner-two dimensions.
    """

    def __init__(self, channels, use_conv=True, dims=2, out_channels=None,padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = (2,1)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
            self.gcn_layers = nn.ModuleList([GCN(channels, channels, nn.SiLU(), 0.1) for _ in range(4)])
            self.pool = Pool(0.9, channels, 0.1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x, g):
        assert x.shape[1] == self.channels
        h = x
        for layer in self.gcn_layers:
            h = h + layer(h, [g])
        h = h.transpose(1,3)
        g, h, idx = self.pool(g, h)
        h = h.permute(0,3,1,2)
        h = self.op(h)
        return h, g, idx


class STUpsample(nn.Module):
    def __init__(self, channels, num_nodes, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.num_nodes = num_nodes
        self.final_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x, idx, graph):
        B, C, T, N = x.shape
        T_full = T * 2
        x = F.interpolate(x, size=(T_full, N), mode="bilinear", align_corners=False)
        
        x_ = torch.zeros(B, C, T_full, self.num_nodes, device=x.device, dtype=x.dtype)
        x_[:,:,:,idx] = x
        
        x_ = self.final_conv(x_)
        x_ = x_ + self.refine(x_)
        x_ = x_.transpose(2, 3)
        
        return x_




class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.d_model = d_model
        mlp_hidden = int(d_model * mlp_ratio)
        
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        
        self.gcn = GCN(d_model, d_model, None, 0.0, order=2)
        self.attn = Attention(d_model, num_heads=num_heads, qkv_bias=True)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden, act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model, bias=True)
        )

    def forward(self, x, g, c):
        B, C, N, T = x.shape
        mod = self.adaLN_modulation(c)
        shift_g, scale_g, gate_g, shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(9, dim=-1)
        
        h = x.permute(0, 2, 3, 1).reshape(B * N, T, C)
        h = modulate(self.norm1(h), shift_g.repeat(N, 1), scale_g.repeat(N, 1))
        h = h.reshape(B, N, T, C).permute(0, 3, 1, 2)
        h = self.gcn(h, [g])
        x = x + gate_g.unsqueeze(-1).unsqueeze(-1) * h
        
        h = x.permute(0, 2, 3, 1).reshape(B * N, T, C)
        h = modulate(self.norm2(h), shift_a.repeat(N, 1), scale_a.repeat(N, 1))
        h = self.attn(h)
        h = h.reshape(B, N, T, C).permute(0, 3, 1, 2)
        x = x + gate_a.unsqueeze(-1).unsqueeze(-1) * h
        
        h = x.permute(0, 2, 3, 1).reshape(B * N, T, C)
        h = modulate(self.norm3(h), shift_m.repeat(N, 1), scale_m.repeat(N, 1))
        h = self.mlp(h)
        h = h.reshape(B, N, T, C).permute(0, 3, 1, 2)
        x = x + gate_m.unsqueeze(-1).unsqueeze(-1) * h
        
        return x


class GraphUnet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.adj_path = os.path.join(f'./Data/{args.dataset_name}/graph', f'{args.year}_adj.npz')
        self.adj = torch.FloatTensor(np.load(self.adj_path)['x']).to(args.device)

        self.H = 24
        self.W = args.num_nodes
        emb_size = 64
        self.emb_size = emb_size
        self.num_nodes = args.num_nodes
        
        self.time_emb = TimeEmbedding(dim=emb_size)
        self.text_proj = nn.Linear(768, emb_size)
        
        self.input_proj = nn.Linear(1, emb_size)
        pos_embed = get_sinusoidal_positional_embeddings(self.H, emb_size)
        self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        
        self.enc1 = GraphTransformerBlock(emb_size, num_heads=4)
        self.pool1 = Pool(0.9, emb_size, 0.0)
        self.down1 = nn.Conv1d(emb_size, emb_size, kernel_size=2, stride=2)
        
        self.enc2 = GraphTransformerBlock(emb_size, num_heads=4)
        self.pool2 = Pool(0.9, emb_size, 0.0)
        self.down2 = nn.Conv1d(emb_size, emb_size, kernel_size=2, stride=2)
        
        self.bottleneck = GraphTransformerBlock(emb_size, num_heads=4)
        
        self.up2 = nn.ConvTranspose1d(emb_size, emb_size, kernel_size=2, stride=2)
        self.unpool2_gcn = nn.ModuleList([GCN(emb_size, emb_size, nn.SiLU(), 0.0) for _ in range(2)])
        self.dec_proj2 = nn.Linear(emb_size * 2, emb_size)
        self.dec2 = GraphTransformerBlock(emb_size, num_heads=4)
        
        self.up1 = nn.ConvTranspose1d(emb_size, emb_size, kernel_size=2, stride=2)
        self.unpool1_gcn = nn.ModuleList([GCN(emb_size, emb_size, nn.SiLU(), 0.0) for _ in range(2)])
        self.dec_proj1 = nn.Linear(emb_size * 2, emb_size)
        self.dec1 = GraphTransformerBlock(emb_size, num_heads=4)
        
        self.final_ln = nn.LayerNorm(emb_size, elementwise_affine=False, eps=1e-6)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_size, 2 * emb_size, bias=True)
        )
        self.output_proj = nn.Linear(emb_size, 1)
        
        self.initialize_weights()
        
    def _unpool(self, x, idx, full_n, g, gcn_layers):
        B, C, N_p, T = x.shape
        out = torch.zeros(B, C, full_n, T, device=x.device, dtype=x.dtype)
        out[:, :, idx, :] = x
        for gcn in gcn_layers:
            out = out + gcn(out, [g])
        return out

    def forward(self, input: torch.Tensor, t: torch.Tensor, text_input):
        c = self.time_emb(t)
        if text_input is not None:
            c = c + self.text_proj(text_input)
            
        x = self.input_proj(input.unsqueeze(-1))
        B, T, N, C = x.shape
        x = x + self.pos_embed.unsqueeze(2)
        x = x.permute(0, 3, 2, 1)
        
        h0 = x
        
        x = self.enc1(x, self.adj, c)
        skip1 = x
        x_p = x.permute(0, 3, 2, 1)
        g1, x_pooled, idx1 = self.pool1(self.adj, x_p)
        x = x_pooled.permute(0, 3, 2, 1)
        B, C, N1, T1 = x.shape
        x = x.reshape(B * N1, C, T1)
        x = self.down1(x)
        x = x.reshape(B, N1, C, -1).permute(0, 2, 1, 3)
        
        x = self.enc2(x, g1, c)
        skip2 = x
        x_p = x.permute(0, 3, 2, 1)
        g2, x_pooled, idx2 = self.pool2(g1, x_p)
        x = x_pooled.permute(0, 3, 2, 1)
        B, C, N2, T2 = x.shape
        x = x.reshape(B * N2, C, T2)
        x = self.down2(x)
        x = x.reshape(B, N2, C, -1).permute(0, 2, 1, 3)
        
        x = self.bottleneck(x, g2, c)
        
        B, C, N2, T_b = x.shape
        x = x.reshape(B * N2, C, T_b)
        x = self.up2(x)
        T_new = x.shape[-1]
        x = x.reshape(B, N2, C, T_new).permute(0, 2, 1, 3)
        if T_new != skip2.shape[-1]:
            x = F.interpolate(x, size=(N2, skip2.shape[-1]), mode='bilinear', align_corners=False)
        x = self._unpool(x, idx2, N1, g1, self.unpool2_gcn)
        x = torch.cat([x, skip2], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.dec_proj2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dec2(x, g1, c)
        
        B, C, N1, T1 = x.shape
        x = x.reshape(B * N1, C, T1)
        x = self.up1(x)
        T_new = x.shape[-1]
        x = x.reshape(B, N1, C, T_new).permute(0, 2, 1, 3)
        if T_new != skip1.shape[-1]:
            x = F.interpolate(x, size=(N1, skip1.shape[-1]), mode='bilinear', align_corners=False)
        x = self._unpool(x, idx1, self.num_nodes, self.adj, self.unpool1_gcn)
        x = torch.cat([x, skip1], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = self.dec_proj1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dec1(x, self.adj, c)
        
        x = x + h0
        
        x = x.permute(0, 3, 2, 1)
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_ln(x) * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
        x = self.output_proj(x).squeeze(-1)
        
        return x

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        
        for block in [self.enc1, self.enc2, self.bottleneck, self.dec1, self.dec2]:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)