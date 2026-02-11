import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model.pretrained.core import BaseModel

import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, mlp_ratio, num_heads=1, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        self.start_proj = nn.Linear(1, hidden_dim)
        self.end_proj = nn.Linear(hidden_dim, 1)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

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
        output = output.view(B, N, L, D)
        output = self.end_proj(output)+org
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


class vqvae(BaseModel):
    def __init__(self, args):
        super().__init__()
        num_residual_layers = args.num_residual_layers
        # self.encoder = Encoder(705, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
        self.encoder = TransformerLayers(1, num_residual_layers, 16)
        self.decoder = TransformerLayers(1, num_residual_layers, 16)
    def shared_eval(self, batch, optimizer, mode):
        if mode == 'train':
            optimizer.zero_grad()
            z = self.encoder(batch.unsqueeze(-1).transpose(1,2)) #(B,T,N)
            data_recon = self.decoder(z).squeeze(-1).transpose(1,2)
            
            recon_error = F.mse_loss(data_recon, batch)
            loss = recon_error
            loss.backward()
            optimizer.step()
        elif mode == 'val' or mode == 'test':
            with torch.no_grad():
                z = self.encoder(batch.unsqueeze(-1).transpose(1,2))
                data_recon = self.decoder(z).squeeze(-1).transpose(1,2)
                recon_error = F.mse_loss(data_recon, batch)
                # cross_loss = F.mse_loss(before, after)
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
        mask_ratio = 0.75
        mask_length = int(L * mask_ratio)
        
        mask = torch.ones(B, L, N, device=x.device)
        mask_indices = torch.randperm(L)[:mask_length]
        mask[:, mask_indices, :] = 0
        return mask
    def get_spa_mask(self, x):
        B, L, N = x.shape
        mask_ratio = 0.75
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
    parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--save_path', type=str,default='results/saved_pretrained_models/', help='denoiser model save path')
    # Model-specific parameters
    parser.add_argument('--general_seed', type=int, default=42, help='seed for random number generation')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for the optimizer')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='number of residual layers in the model')
    args = parser.parse_args()
    model = vqvae(args)
    print(model.shared_eval(torch.randn(4,96,705),torch.optim.Adam(model.parameters(), lr=args.learning_rate),'train'))