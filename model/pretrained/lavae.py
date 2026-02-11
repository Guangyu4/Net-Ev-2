import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from model.pretrained.core import BaseModel

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
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim, train_mode):
        super(Encoder, self).__init__()
        self.train_mode = train_mode
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=in_channels,
                                             num_hiddens=in_channels,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=embedding_dim)
        self._pre_vq_conv = nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1, stride=1)
        
        if self.train_mode == 'glaencoder':
            self.encoder_proj = nn.Linear(1820,705) # use sd backbone so there are 705 nodes
        elif self.train_mode == 'gbaencoder':
            self.encoder_proj = nn.Linear(2294,1820) # use gla backbone so there are 1820 nodes

 
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
        # x = F.interpolate(x, size=30, mode='linear', align_corners=True)
        if self.train_mode in ['glaencoder', 'gbaencoder']:
            x = x.transpose(1, 2)
            x = self.encoder_proj(x)
            x = x.transpose(1, 2)
        return x, before


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=705, train_mode='normal'):
        super(Decoder, self).__init__()
        self.train_mode = train_mode
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

        if self.train_mode == 'glaencoder':
            self.decoder_proj = nn.Linear(705,1820) # use sd backbone so there are 705 nodes
        elif self.train_mode == 'gbaencoder':
            self.decoder_proj = nn.Linear(1820,2294) # use gla backbone so there are 1820 nodes

    def forward(self, inputs, length):
        # x = F.interpolate(inputs, size=int(length / 4), mode='linear', align_corners=True)
        after = inputs
        x = inputs
        
        if self.train_mode in ['glaencoder', 'gbaencoder']:
            x = x.transpose(1, 2)
            x = self.decoder_proj(x)
            x = x.transpose(1, 2)

        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        x = self._conv_trans_2(x)
        x = x.transpose(1, 2)
        # x = F.interpolate(x.transpose(1, 2), size=length, mode='linear', align_corners=True).transpose(1, 2)
        return x, after


class lavae(BaseModel):
    def __init__(self, args):
        super().__init__()
        num_hiddens = args.block_hidden_size
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.res_hidden_size
        embedding_dim = args.embedding_dim
        self.train_mode = args.train_mode
        self.encoder = Encoder(args.num_nodes, num_hiddens, num_residual_layers, num_residual_hiddens, args.num_nodes, train_mode=self.train_mode)
        self.decoder = Decoder(args.num_nodes, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=args.num_nodes, train_mode=self.train_mode)

    def shared_eval(self, batch, optimizer, mode):
        if mode == 'train':
            optimizer.zero_grad()
            z, before = self.encoder(batch)
            data_recon, after = self.decoder(z,length=batch.shape[1])
            recon_error = F.mse_loss(data_recon, batch)
            # cross_loss = F.mse_loss(before, after)
            loss = recon_error #+ cross_loss
            loss.backward()
            optimizer.step()
        elif mode == 'val' or mode == 'test':
            with torch.no_grad():
                z, before = self.encoder(batch)
                data_recon, after = self.decoder(z,length=batch.shape[1])
                recon_error = F.mse_loss(data_recon, batch)
                # cross_loss = F.mse_loss(before, after)    
                loss = recon_error #+ cross_loss
        return loss, recon_error, data_recon, z

    def forward(self, x):
        original_length = x.shape[1]
        z, before = self.encoder(x)
        print("Encoder Output Shape", z.shape)
        data_recon, after = self.decoder(z, length=original_length)
        return data_recon


class vqvae(BaseModel):
    def __init__(self, args):
        super().__init__()
        num_hiddens = args.block_hidden_size
        num_residual_layers = args.num_residual_layers
        num_residual_hiddens = args.res_hidden_size
        embedding_dim = args.embedding_dim
        self.encoder = Encoder(args.num_nodes, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)
        self.decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens, out_channels=args.num_nodes)

    def shared_eval(self, batch, optimizer, mode):
        if mode == 'train':
            optimizer.zero_grad()
            z, before = self.encoder(batch)
            data_recon, after = self.decoder(z,length=batch.shape[1])
            recon_error = F.mse_loss(data_recon, batch)
            cross_loss = F.mse_loss(before, after)
            loss = recon_error + cross_loss
            loss.backward()
            optimizer.step()
        elif mode == 'val' or mode == 'test':
            with torch.no_grad():
                z, before = self.encoder(batch)
                data_recon, after = self.decoder(z,length=batch.shape[1])
                recon_error = F.mse_loss(data_recon, batch)
                cross_loss = F.mse_loss(before, after)
                loss = recon_error + cross_loss
        return loss, recon_error, data_recon, z

    def forward(self, x):
        original_length = x.shape[1]
        z, before = self.encoder(x)
        print("Encoder Output Shape", z.shape)
        data_recon, after = self.decoder(z, length=original_length)
        return data_recon


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
    model = vqvae(args)
    print(model.shared_eval(torch.randn(4,96,705),torch.optim.Adam(model.parameters(), lr=args.learning_rate),'train')[2].shape)
