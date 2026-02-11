from .stdataset import STDataset
from torch.utils.data import DataLoader
from .incgdataset import INCGDataset
def get_pretrained_dataset(args, period):
    args.pretrained_model = args.pretrained_model.upper()
    if args.pretrained_model in ('VQVAE','LAVAE','MAE','NONE'):
        dataset = STDataset(name=args.dataset_name, data_root=f'./Data/{args.dataset_name}', period=period,args=args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.pretrained_model in ('INCGMAE'):
        dataset = INCGDataset(name=args.dataset_name, data_root=f'./Data/{args.dataset_name}', period=period,args=args)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    else:
        raise ValueError(f"No dataset found")
    return dataset, dataloader
