# -------------------------------- Pretrained Model -------------------------------- #
from .pretrained.incgae import incgae
from .pretrained.vqvae import vqvae
from .pretrained.lavae import lavae
from .pretrained.mae import mae
from .pretrained.incgae_nomask import incgae_nomask
from .pretrained.incgae_eventmask import incgae_eventmask
from .denoiser.graphunet import GraphUnet
# -------------------------------- Denoiser -------------------------------- #
from .denoiser.transformer import Transformer
from .denoiser.mlp import MLP
from .denoiser.trendtransformer import TrendTransformer
from .denoiser.chattraffic import Chattraffic
from .denoiser.verbalts import VerbalTS
from .denoiser.simple_unet import SimpleUNet
# -------------------------------- Backbone -------------------------------- #
from .backbone.rectified_flow import RectifiedFlow
from .backbone.DDPM import DDPM
from .backbone.Diffusion_TS import Diffusion_TS

def get_pretrained_model(args):
    args.pretrained_model = args.pretrained_model.upper()
    if args.pretrained_model == 'INCGMAE':
        return incgae(args)
    elif args.pretrained_model == 'VQVAE':
        return vqvae(args)
    elif args.pretrained_model == 'LAVAE':
        return lavae(args)
    elif args.pretrained_model == 'MAE':
        return mae(args)
    elif args.pretrained_model == 'NONE':
        return 
    elif args.pretrained_model == 'INCGMAE_NOMASK':
        return incgae_nomask(args)
    elif args.pretrained_model == 'INCGMAE_EVENTMASK':
        return incgae_eventmask(args)
    else:
        raise ValueError(f"No pretrained model found")

def get_denoiser_model(args):
    args.denoiser = args.denoiser.upper()
    if args.denoiser == 'DIT':
        return Transformer(args)
    elif args.denoiser == 'MLP':
        return MLP()
    elif args.denoiser == 'DIFFUSION_TS':
        return TrendTransformer(args)
    elif args.denoiser == 'CHATTRAFFIC':
        return Chattraffic(args)
    elif args.denoiser == 'GRAPHUNET':
        return GraphUnet(args)
    elif args.denoiser == 'VERBALTS':
        return VerbalTS(args)
    elif args.denoiser == 'SIMPLEUNET':
        return SimpleUNet(args)
    else:
        raise ValueError(f"No denoiser model found")

def get_denoiser_backbone(args):
    args.backbone = args.backbone.upper()
    if args.backbone == 'FLOWMATCHING':
        return RectifiedFlow()
    elif args.backbone in ('DDPM','INCDDPM'):
        return DDPM(args.total_step, args.device)
    elif args.backbone == 'DIFFUSION_TS':
        return Diffusion_TS(args.total_step, args.device)
    elif args.backbone == 'VERBALTS':
        return Diffusion_TS(args.total_step, args.device)
    else:
        raise ValueError(f"No denoiser backbone found")
