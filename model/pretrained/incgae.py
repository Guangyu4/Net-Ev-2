import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model.pretrained.core import BaseModel

import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, args, hidden_dim, nlayers, mlp_ratio, num_heads=4, dropout=0, out_dim=1):
        super().__init__()
        self.d_model = hidden_dim
        self.start_proj = nn.Linear(1, hidden_dim)
        self.end_proj = nn.Linear(hidden_dim, out_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout, batch_first=True)
        # encoder_layers2 = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.transformer_encoder2 = TransformerEncoder(encoder_layers2, nlayers)

    def forward(self, src):
        org = src.clone()
        src = self.start_proj(src)
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src=src.contiguous()
        src = src.view(B*N, L, D)
        # src = src.transpose(0, 1)
        # output = self.transformer_encoder(src, mask=None)
        # output = output.transpose(0, 1).view(B, N, L, D)
        output = self.transformer_encoder(src, mask=None)
        # output = self.transformer_encoder2(output, mask=None)
        output = output.view(B, N, L, D)
        output = self.end_proj(output)
        # output = output.view(B, N, L, 1)
        return output
    

class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.mean = torch.clamp(self.mean, -10.0, 10.0)
        self.logvar = torch.clamp(self.logvar, -10.0, 2.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        noise = torch.randn_like(self.mean)
        noise = torch.clamp(noise, -3.0, 3.0)
        x = self.mean + self.std * noise
        x = torch.clamp(x, -10.0, 10.0)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean

class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        self._pre_vq_conv = nn.Conv1d(in_channels=num_hiddens, out_channels=embedding_dim, kernel_size=1, stride=1)

    def forward(self, inputs):
        x = inputs.transpose(1, 2)
        x = self._conv_1(x)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        x = self._residual_stack(x)
        x = self._pre_vq_conv(x)
        before = x
        x = F.interpolate(x, size=30, mode='linear', align_corners=True)
        return x, before


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=705):
        super(Decoder, self).__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                out_channels=num_hiddens,
                                                kernel_size=4,
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens,
                                                out_channels=out_channels,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs, length):
        x = F.interpolate(inputs, size=int(length / 4), mode='linear', align_corners=True)
        after = x
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)
        x = x.transpose(1, 2)
        x = F.interpolate(x.transpose(1, 2), size=length, mode='linear', align_corners=True).transpose(1, 2)
        return x, after

# class DoubleTransformerLayers(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.transformer_encoder1 = TransformerLayers(args, 12, args.num_residual_layers, 24)
#         self.transformer_encoder2 = TransformerLayers(args, 12, args.num_residual_layers, 24)
#     def forward(self, x):
#         x = self.transformer_encoder1(x)
#         x = self.transformer_encoder2(x)
#         return x

class INCEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_residual_layers = args.num_residual_layers
        self.num_nodes = args.num_nodes
        hidden_dim = 32
        self.transformer_encoder = TransformerLayers(args, hidden_dim, num_residual_layers, 4, num_heads=4, out_dim=2)
        self.time_compress = nn.Sequential(
            nn.Conv1d(args.num_nodes, args.num_nodes, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.Conv1d(args.num_nodes, args.num_nodes, kernel_size=1),
        )
    
    def forward(self, x, sample=True):
        x = torch.clamp(x, -100.0, 100.0)
        x = self.transformer_encoder(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
        posterior = DiagonalGaussianDistribution(x)
        if sample and self.training:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = z.squeeze(-1)
        z = self.time_compress(z)
        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        before = z
        return z, before


class INCDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        num_residual_layers = args.num_residual_layers
        self.num_nodes = args.num_nodes
        hidden_dim = 32
        self.time_decompress = nn.Sequential(
            nn.ConvTranspose1d(args.num_nodes, args.num_nodes, kernel_size=4, stride=4, padding=0),
            nn.GELU(),
            nn.Conv1d(args.num_nodes, args.num_nodes, kernel_size=1),
        )
        self.transformer_decoder = TransformerLayers(args, hidden_dim, num_residual_layers, 4, num_heads=4, out_dim=1)
        
    def forward(self, z, length=96):
        after = z.clone()
        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        z = self.time_decompress(z)
        z = z.unsqueeze(-1)
        z = self.transformer_decoder(z)
        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
        return z, after


class incgae(BaseModel):
    def __init__(self, args):
        super().__init__()

        
        self.encoder = INCEncoder(args)
        self.decoder = INCDecoder(args)

    def shared_eval(self, batch, optimizer, mode, nodeid, timeid):
        if mode == 'train':
            self._skip_step = False
            optimizer.zero_grad()
            tem_mask = self.get_tem_mask(batch)
            spa_mask = self.get_spa_mask(batch)
            
            batch_masked = batch * tem_mask
            z, _ = self.encoder(batch_masked.unsqueeze(-1).transpose(1,2))
            data_recon_tem, _ = self.decoder(z)
            data_recon_tem = data_recon_tem.squeeze(-1).transpose(1,2)
            
            data_recon_masked = data_recon_tem * spa_mask
            z2, _ = self.encoder(data_recon_masked.unsqueeze(-1).transpose(1,2))
            data_recon_spa, _ = self.decoder(z2)
            data_recon_spa = data_recon_spa.squeeze(-1).transpose(1,2)

            data_recon = data_recon_tem + data_recon_spa
            recon_error = F.mse_loss(data_recon, batch)
            
            loss = recon_error
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                self._skip_step = True
                return loss, recon_error, data_recon, z
            loss.backward()
            bad_grad = False
            for p in self.parameters():
                if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                    bad_grad = True
                    break
            if bad_grad:
                optimizer.zero_grad()
                self._skip_step = True
                return loss, recon_error, data_recon, z
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()
        elif mode == 'val' or mode == 'test':
            with torch.no_grad():
                z, _ = self.encoder(batch.unsqueeze(-1).transpose(1,2))
                data_recon, _ = self.decoder(z)
                data_recon = data_recon.squeeze(-1).transpose(1,2)
                recon_error = F.mse_loss(data_recon, batch)
                loss = recon_error
        return loss, recon_error, data_recon, z

    def forward(self, x):
        original_length = x.shape[1]
        z = self.encoder(x.unsqueeze(-1).transpose(1,2))
        print("Encoder Output Shape", z.shape)
        data_recon = self.decoder(z).squeeze(-1).transpose(1,2)
        return data_recon
    def get_tem_mask(self, x):
        B, L, N = x.shape
        mask_ratio = 0.05
        mask_length = int(L * mask_ratio)
        
        mask = torch.ones(B, L, N, device=x.device)
        mask_indices = torch.randperm(L)[:mask_length]
        mask[:, mask_indices, :] = 0
        return mask
    def get_spa_mask(self, x):
        B, L, N = x.shape
        mask_ratio = 0.05
        mask_length = int(N * mask_ratio)
        
        mask = torch.ones(B, L, N, device=x.device)
        mask_indices = torch.randperm(N)[:mask_length]
        mask[:, :, mask_indices] = 0
        return mask
    def get_inc_mask(self, x, nodeid, timeid):
        B, L, N = x.shape
        mask = torch.ones(B, L, N, device=x.device)
        nodeid = [item for sublist in nodeid for item in sublist if item != -1]
        mask[:, :, nodeid] = 0
        mask[:, :, :max(timeid)] = 0
        return mask
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='SD', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_training_updates', type=int, default=2000, help='number of training updates/epochs')
    parser.add_argument('--save_path', type=str,default='results/saved_pretrained_models/', help='denoiser model save path')
    # Model-specific parameters
    parser.add_argument('--general_seed', type=int, default=42, help='seed for random number generation')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for the optimizer')
    parser.add_argument('--block_hidden_size', type=int, default=32, help='hidden size of the blocks in the network')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers in the model')
    parser.add_argument('--res_hidden_size', type=int, default=64, help='hidden size of the residual layers')
    parser.add_argument('--embedding_dim', type=int, default=16, help='dimension of the embeddings')
    parser.add_argument('--num_embeddings', type=int, default=128, help='number of embeddings in the VQ-VAE')
    parser.add_argument('--compression_factor', type=int, default=4, help='compression factor')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost used in the loss function')
    parser.add_argument('--mix_train', type=bool, default=False, help='whether to use mixture training')
    args = parser.parse_args()
    model = incgae(args)
    print(model.shared_eval(torch.randn(4,96,705),torch.optim.Adam(model.parameters(), lr=args.learning_rate),'train'))