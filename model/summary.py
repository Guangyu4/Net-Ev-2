import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from T2S.model.pretrained.lavae import lavae

# from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='GLA', help='dataset name')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_training_updates', type=int, default=2000, help='number of training updates/epochs')
parser.add_argument('--save_path', type=str,default='results/saved_pretrained_models/', help='denoiser model save path')
# Model-specific parameters
parser.add_argument('--general_seed', type=int, default=42, help='seed for random number generation')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for the optimizer')
parser.add_argument('--block_hidden_size', type=int, default=64, help='hidden size of the blocks in the network')
parser.add_argument('--num_residual_layers', type=int, default=64, help='number of residual layers in the model')
parser.add_argument('--res_hidden_size', type=int, default=64, help='hidden size of the residual layers')
parser.add_argument('--embedding_dim', type=int, default=64, help='dimension of the embeddings')
parser.add_argument('--num_embeddings', type=int, default=64, help='number of embeddings in the VQ-VAE')
parser.add_argument('--compression_factor', type=int, default=4, help='compression factor')
parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost used in the loss function')
parser.add_argument('--mix_train', type=bool, default=False, help='whether to use mixture training')
parser.add_argument('--train_mode', type=str, default='none', help='train mode: glaencoder or gbaencoder')
parser.add_argument('--num_nodes', type=int, default=1820, help='number of nodes')
parser.add_argument('--total_step', type=int, default=100, help='total step')
parser.add_argument('--device', type=str, default='cuda', help='device')    
parser.add_argument('--year', type=str, default='2017', help='year')
args = parser.parse_args()
# model = lavae(args)
# model = model.cuda()
# summary(model, (96, 705))






from T2S.model.denoiser.trendtransformer import TrendTransformer

model = TrendTransformer(args)
model = model.cuda()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"DiffusionTS parameters: {total_params:,}")


from T2S.model.denoiser.verbalts import VerbalTS

model = VerbalTS(args)
model = model.cuda()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"VerbalTS parameters: {total_params:,}")


from T2S.model.denoiser.chattraffic import Chattraffic
from T2S.model.pretrained.vqvae import vqvae
model = Chattraffic(args)
model = model.cuda()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model = vqvae(args)
model = model.cuda()

vae_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"Chattraffic parameters: {total_params:,}+{vae_params:,}")

from T2S.model.denoiser.transformer import Transformer
from T2S.model.pretrained.lavae import lavae

model = Transformer(args)
model = model.cuda()


total_params = sum(p.numel() for p in model.parameters())

model = lavae(args)
model = model.cuda()

vae_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"T2S parameters: {total_params:,}+{vae_params:,}")


from T2S.model.denoiser.graphunet import GraphUnet
from T2S.model.pretrained.incgae import incgae

model = GraphUnet(args)
model = model.cuda()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

model = incgae(args)
model = model.cuda()

incgae_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"GraphUnet parameters: {total_params:,}+{incgae_params:,}")