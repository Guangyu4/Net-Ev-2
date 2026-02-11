from .STPRunner import STPRunner
from .INCPRunner import INCPRunner
from .DDPMRunner import DDPMRunner
from .FlowMatchingRunner import FlowMatchingRunner
from .DiffusionTSRunner import DiffusionTSRunner
from .GraphRunner import GraphRunner
from .VerbalTSRunner import VerbalTSRunner
def get_pretrained_runner(args):
    args.pretrained_model = args.pretrained_model.upper()
    if args.pretrained_model in ('VQVAE','LAVAE','MAE'):
        return STPRunner(args)
    elif args.pretrained_model in ['INCGMAE']:
        return INCPRunner(args)
    else:
        raise ValueError(f"No pretrained runner found")


def get_diffusion_runner(args):
    args.backbone = args.backbone.upper()
    if args.backbone == 'FLOWMATCHING':
        return FlowMatchingRunner(args)
    elif args.backbone in ('DDPM'):
        return DDPMRunner(args)
    elif args.backbone == 'DIFFUSION_TS':
        return DiffusionTSRunner(args)
    elif args.backbone == 'INCDDPM':
        return GraphRunner(args)
    elif args.backbone == 'VERBALTS':
        return VerbalTSRunner(args)
    else:
        raise ValueError(f"No diffusion runner found")