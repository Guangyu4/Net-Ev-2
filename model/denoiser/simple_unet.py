import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        assert dim % 2 == 0
    def forward(self, t):
        t = t.float()
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 4 * out_ch)
        )
        if in_ch != out_ch:
            self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, c):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        shift1, scale1, shift2, scale2 = self.adaLN(c).chunk(4, dim=-1)
        h = h * (1 + scale1.unsqueeze(-1)) + shift1.unsqueeze(-1)
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = h * (1 + scale2.unsqueeze(-1)) + shift2.unsqueeze(-1)
        h = self.act(h)
        
        return h + self.skip(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.conv_block = ConvBlock(in_ch, out_ch, emb_dim)
        self.down = nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2)

    def forward(self, x, c):
        h = self.conv_block(x, c)
        return self.down(h), h

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv_block = ConvBlock(in_ch + out_ch, out_ch, emb_dim)

    def forward(self, x, skip, c):
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x, c)

class SimpleUNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_nodes = args.num_nodes
        emb_size = 64
        
        self.time_emb = TimeEmbedding(dim=emb_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.SiLU(),
            nn.Linear(emb_size * 4, emb_size)
        )
        self.text_proj = nn.Linear(768, emb_size)
        
        ch = 64
        self.input_proj = nn.Conv1d(self.num_nodes, ch, kernel_size=1)
        
        self.down1 = DownBlock(ch, ch, emb_size)
        self.down2 = DownBlock(ch, ch * 2, emb_size)
        
        self.bottleneck = ConvBlock(ch * 2, ch * 2, emb_size)
        
        self.up2 = UpBlock(ch * 2, ch * 2, emb_size)
        self.up1 = UpBlock(ch * 2, ch, emb_size)
        
        self.output_proj = nn.Conv1d(ch, self.num_nodes, kernel_size=1)
        
        self.final_norm = nn.LayerNorm(self.num_nodes)
        self.final_adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_size, 2 * self.num_nodes)
        )
        
        self.initialize_weights()
    
    def forward(self, input: torch.Tensor, t: torch.Tensor, text_input):
        B, T, N = input.shape
        
        c = self.time_mlp(self.time_emb(t))
        if text_input is not None:
            c = c + self.text_proj(text_input)
        
        x = input.permute(0, 2, 1)
        x = self.input_proj(x)
        
        x, skip1 = self.down1(x, c)
        x, skip2 = self.down2(x, c)
        
        x = self.bottleneck(x, c)
        
        x = self.up2(x, skip2, c)
        x = self.up1(x, skip1, c)
        
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)
        
        shift, scale = self.final_adaLN(c).chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        
        return x
    
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.constant_(self.output_proj.weight, 0)
        nn.init.constant_(self.output_proj.bias, 0)
        nn.init.constant_(self.final_adaLN[-1].weight, 0)
        nn.init.constant_(self.final_adaLN[-1].bias, 0)
