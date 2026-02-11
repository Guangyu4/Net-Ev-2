import argparse
import os
import torch
import numpy as np
import math
import gc
from torch.optim import AdamW, lr_scheduler
from model import get_denoiser_model, get_denoiser_backbone, get_pretrained_model
from model.denoiser.mlp import MLP
from model.denoiser.transformer import Transformer
from model.denoiser.trendtransformer import TrendTransformer
from model.denoiser.graphunet import GraphUnet
from model.denoiser.verbalts import VerbalTS
from model.pretrained.bert import BertEmbedding
from model.backbone.rectified_flow import RectifiedFlow
from model.backbone.DDPM import DDPM
from model.backbone.Diffusion_TS import Diffusion_TS
from datafactory import get_pretrained_dataset
from datafactory.dataloader import loader_provider
from libs.utils import seed_everything, print_log
from runners import get_diffusion_runner, get_pretrained_runner
from datetime import datetime
from tqdm import tqdm
import time
from scipy.linalg import sqrtm
from torch import Tensor
from torch_geometric.data import TemporalData
from evaluate.ts2vec import initialize_ts2vec

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

MODEL_CONFIGS = {
    'T2S': {'pretrained_model': 'lavae', 'denoiser': 'DiT', 'backbone': 'flowmatching'},
    'chattraffic': {'pretrained_model': 'vqvae', 'denoiser': 'Chattraffic', 'backbone': 'ddpm'},
    'Diffusion_TS': {'pretrained_model': 'none', 'denoiser': 'Diffusion_TS', 'backbone': 'diffusion_ts'},
    'NetEv2': {'pretrained_model': 'incgmae', 'denoiser': 'GraphUnet', 'backbone': 'incddpm'},
    'VerbalTS': {'pretrained_model': 'none', 'denoiser': 'VerbalTS', 'backbone': 'verbalts'},
    'W/O masking': {'pretrained_model': 'vqvae', 'denoiser': 'GraphUnet', 'backbone': 'incddpm'},
    'W/O InCGMasking': {'pretrained_model': 'mae', 'denoiser': 'GraphUnet', 'backbone': 'incddpm'},
    'W/O GraphUnet': {'pretrained_model': 'incgmae', 'denoiser': 'Chattraffic', 'backbone': 'incddpm'},
    'W/O STSampling': {'pretrained_model': 'incgmae', 'denoiser': 'GraphUnet', 'backbone': 'incddpm'}
}

def get_args():
    parser = argparse.ArgumentParser(description="Train T2S model")
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint path')
    parser.add_argument('--model_name', type=str, default='NetEv2', help='model name')
    parser.add_argument('--dataset_name', type=str, default='GLA', help='dataset name')
    parser.add_argument('--year', type=str, default='2017', help='year')
    parser.add_argument('--num_nodes', type=int, default=8196, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1, help='training epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=1, help='pretrain epochs')
    parser.add_argument('--save_path', type=str, default='./results/', help='model save path')
    parser.add_argument('--mix_train', type=bool, default=False, help='mixture train or not')
    parser.add_argument('--block_hidden_size', type=int, default=16, help='hidden size')
    parser.add_argument('--num_residual_layers', type=int, default=1, help='number of residual layers')
    parser.add_argument('--res_hidden_size', type=int, default=32, help='hidden size of residual layers')
    parser.add_argument('--embedding_dim', type=int, default=8, help='dimension of embeddings')
    parser.add_argument('--num_embeddings', type=int, default=32, help='number of embeddings')
    parser.add_argument('--compression_factor', type=int, default=4, help='compression factor')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost')
    parser.add_argument('--usepretrainedvae', default=False, help='pretrained vae')
    parser.add_argument('--total_step', type=int, default=100, help='sampling steps')
    parser.add_argument('--train_mode', type=str, default='expansion', help='train mode')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='pretrain learning rate')
    parser.add_argument('--general_seed', type=int, default=42, help='seed')
    parser.add_argument('--incg', type=bool, default=True, help='whether to use incg')
    parser.add_argument('--cfg_scale', type=float, default=2, help='CFG Scale for inference')
    parser.add_argument('--checkpoint_id', type=int, default=0, help='checkpoint id for inference')
    parser.add_argument('--run_multi', type=bool, default=False, help='run multi times for CRPS,MAP,MRR,NDCG')
    parser.add_argument('--chattraffic_emb_size', type=int, default=8, help='chattraffic embedding size')
    parser.add_argument('--chattraffic_n_layers', type=int, default=1, help='chattraffic transformer layers')
    
    args = parser.parse_args()
    config = MODEL_CONFIGS.get(args.model_name, MODEL_CONFIGS['NetEv2'])
    args.pretrained_model = config['pretrained_model']
    args.denoiser = config['denoiser']
    args.backbone = config['backbone']
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mix_train:
        args.data_length = 0
    args.main_save_path = os.path.join(args.save_path, args.model_name, args.year, 'checkpoints', '{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name))
    args.generation_save_path = os.path.join(args.save_path, args.model_name, args.year, 'generation', '{}_{}_{}_{}_{}'.format(args.backbone, args.denoiser, args.dataset_name, args.cfg_scale, args.total_step))
    return args

def pretrain(args, log):
    print_log(f"\n{'='*50}", log=log)
    print_log(f"Phase 1: Pretraining {args.pretrained_model}...", log=log)
    print_log(f"{'='*50}\n", log=log)
    
    model = get_pretrained_model(args).to(args.device)
    optimizer = model.configure_optimizers(lr=args.learning_rate)
    
    runner = get_pretrained_runner(args)
    _, train_loader = get_pretrained_dataset(args, period='train')
    _, test_loader = get_pretrained_dataset(args, period='test')
    
    start_time = time.time()
    start_time_formatted = datetime.fromtimestamp(start_time).strftime("%Y-%j-%H:%M:%S.%f")[:-4]
    print_log(f"Pretraining start time: {start_time_formatted}", log=log)
    
    args.num_epochs = args.pretrain_epochs
    runner.train_no_save(args, model, optimizer, train_loader, test_loader, log=log)
    
    end_time = time.time()
    end_time_formatted = datetime.fromtimestamp(end_time).strftime("%Y-%j-%H:%M:%S.%f")[:-4]
    print_log(f"Pretraining end time: {end_time_formatted}", log=log)
    print_log(f"Pretraining total time: {end_time - start_time:.2f} seconds", log=log)
    
    return model

def train_main(args, pretrained_model, log):
    print_log(f"\n{'='*50}", log=log)
    print_log(f"Phase 2: Training main model {args.model_name}...", log=log)
    print_log(f"{'='*50}\n", log=log)
    
    os.makedirs(args.main_save_path, exist_ok=True)
    dataset, dataloader = get_pretrained_dataset(args, period='train')
    
    model = get_denoiser_model(args)
    backbone = get_denoiser_backbone(args)
    
    if pretrained_model is not None:
        model.encoder = pretrained_model.encoder
        model.decoder = pretrained_model.decoder
    model = model.to(args.device)
    
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if hasattr(model, 'encoder'):
        total_params += sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    if hasattr(model, 'decoder'):
        total_params += sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print_log(f"Total learnable parameters: {total_params}", log=log)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=len(dataloader) * args.epochs)
    
    runner = get_diffusion_runner(args)
    
    start_time = time.time()
    print_log(f"Training config:: epoch: {args.epochs}\tsave_path: {args.main_save_path}\tdevice: {args.device}", log=log)
    
    args.save_path = args.main_save_path
    model = runner.train(args, model, backbone, optimizer, scheduler, dataloader, log)
    
    end_time = time.time()
    print_log(f"Main training total time: {end_time - start_time:.2f} seconds", log=log)
    
    return model

def infer(args, model, log):
    print_log(f"\n{'='*50}", log=log)
    print_log(f"Phase 3: Inference...", log=log)
    print_log(f"{'='*50}\n", log=log)
    
    step = args.total_step
    cfg_scale = args.cfg_scale
    device = args.device
    generation_save_path_result = args.generation_save_path_result
    
    print_log(f"Inference config:: Step: {step}\t CFG Scale: {cfg_scale}", log=log)
    os.makedirs(generation_save_path_result, exist_ok=True)
    
    _, dataloader = loader_provider(args, period='test')
    print_log(f"Test dataset length: {len(dataloader)}", log=log)
    
    model.eval()
    
    backbone = get_denoiser_backbone(args)
    if args.backbone == 'flowmatching':
        rf = backbone
    else:
        ddpm = backbone
    
    x_1_list = []
    x_t_list = []
    bert_embedder = BertEmbedding()
    
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            print_log(f"Generating {batch}th Batch TS...", log=log)
            
            if len(data) == 3:
                y, x_1, nodelist = data
            else:
                y, x_1 = data
            x_1 = x_1.float().to(device)
            embedding = bert_embedder.get_embedding(y)
            
            if args.model_name == 'NetEv2':
                x_1_encoded, _ = model.encoder(x_1.unsqueeze(-1).transpose(1, 2))
                x_1_encoded = x_1_encoded.transpose(1, 2)
                x_t = torch.randn_like(x_1_encoded).float().to(device)
            elif args.model_name == 'T2S':
                x_t, _ = model.encoder(x_1)
                x_t = torch.randn_like(x_t).float().to(device)
            elif args.model_name == 'chattraffic' and hasattr(model, 'encoder'):
                x_1_encoded = model.encoder(x_1.unsqueeze(-1)).squeeze(-1)
                x_t = torch.randn_like(x_1_encoded).float().to(device)
            else:
                x_t = torch.randn_like(x_1).float().to(device)
            
            for j in tqdm(range(step), desc=f"Batch {batch}"):
                if args.backbone == 'flowmatching':
                    t = torch.round(torch.full((x_t.shape[0],), j * 1.0 / step, device=device) * step) / step
                    pred_uncond = model(input=x_t, t=t, text_input=None)
                    pred_cond = model(input=x_t, t=t, text_input=embedding)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t = rf.euler(x_t, pred, 1.0 / step)
                elif args.backbone == 'incddpm':
                    t = torch.full((x_t.size(0),), math.floor(step-1-j), dtype=torch.long, device=device)
                    pred_uncond = model(input=x_t, t=t, text_input=None)
                    pred_cond = model(input=x_t, t=t, text_input=embedding)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t = ddpm.p_sample(x_t, pred, t)
                elif args.backbone == 'diffusion_ts':
                    t = torch.full((x_t.size(0),), math.floor(step-1-j), dtype=torch.long, device=device)
                    pred_uncond = model(input=x_t, t=t, text_input=None)
                    pred_cond = model(input=x_t, t=t, text_input=embedding)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t = ddpm.p_sample(x_t, pred, t)
                elif args.backbone == 'verbalts':
                    t = torch.full((x_t.size(0),), math.floor(step-1-j), dtype=torch.long, device=device)
                    x_raw = x_t.unsqueeze(1).permute(0, 1, 3, 2)
                    tp = torch.arange(x_raw.shape[3]).unsqueeze(0).repeat(x_raw.shape[0], 1).to(device)
                    pred_uncond, _ = model(x_raw=x_raw, tp=tp, attr_emb_raw=None, diffusion_step=t)
                    pred_cond, _ = model(x_raw=x_raw, tp=tp, attr_emb_raw=embedding, diffusion_step=t)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t = ddpm.p_sample(x_t, pred, t)
                elif args.backbone == 'ddpm':
                    t = torch.full((x_t.size(0),), math.floor(step-1-j), dtype=torch.long, device=device)
                    pred_uncond = model(input=x_t, t=t, text_input=None)
                    pred_cond = model(input=x_t, t=t, text_input=embedding)
                    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                    x_t = ddpm.p_sample(x_t, pred, t)
            
            if args.model_name == 'T2S' and hasattr(model, 'decoder'):
                x_t, _ = model.decoder(x_t, length=x_1.shape[1])
            elif args.model_name == 'NetEv2' and hasattr(model, 'decoder'):
                x_t, _ = model.decoder(x_t.transpose(1, 2))
                x_t = x_t.squeeze(-1).transpose(1, 2)
            elif args.model_name == 'chattraffic' and hasattr(model, 'decoder'):
                x_t = model.decoder(x_t.unsqueeze(-1)).squeeze(-1)
            
            x_1_np = x_1.detach().cpu().numpy()
            x_t_np = x_t.detach().cpu().numpy()
            if x_1_np.ndim == 2:
                x_1_np = x_1_np[np.newaxis, :, :]
                x_t_np = x_t_np[np.newaxis, :, :]
            x_1_list.append(x_1_np)
            x_t_list.append(x_t_np)
            print_log(f"MAE: {np.abs(x_1_np - x_t_np).mean():.4f}", log=log)
    
    x_1_array = np.concatenate(x_1_list, axis=0)
    x_t_array = np.concatenate(x_t_list, axis=0)
    
    x_1 = x_1_array[:, :, np.newaxis, :] if x_1_array.ndim == 3 else x_1_array
    x_t = x_t_array[:, :, np.newaxis, :] if x_t_array.ndim == 3 else x_t_array
    np.save(os.path.join(generation_save_path_result, 'x_1.npy'), x_1)
    np.save(os.path.join(generation_save_path_result, 'x_t.npy'), x_t)
    
    print_log(f"Results saved to {generation_save_path_result}", log=log)
    
    del x_1_list, x_t_list, x_1_array, x_t_array, bert_embedder
    torch.cuda.empty_cache()
    gc.collect()
    
    return x_1, x_t

def need_pretrain(pretrained_model):
    return pretrained_model != 'none'

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_mae(ori_data, gen_data):
    return np.mean(np.abs(ori_data - gen_data))

def calculate_mse(ori_data, gen_data):
    return np.mean((ori_data - gen_data) ** 2)

def calculate_wape(ori_data, gen_data):
    abs_err = np.sum(np.abs(ori_data - gen_data), axis=(1, 2))
    abs_ori = np.sum(np.abs(ori_data), axis=(1, 2))
    wape = np.where(abs_ori != 0, abs_err / abs_ori, np.nan)
    return np.nanmean(wape)

def calculate_ed(ori_data, gen_data):
    diff = ori_data - gen_data
    ed_per_series = np.sqrt(np.sum(diff ** 2, axis=1))
    return np.mean(ed_per_series)

def cosine_similarity_per_timestep(a, b):
    # a, b: (N, T) - N nodes, T timesteps
    # compute cosine similarity for each timestep (across all nodes), then average
    a = np.array(a)
    b = np.array(b)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
    # a, b: (N, T), compute similarity per timestep
    dot_product = np.sum(a * b, axis=0)  # (T,)
    norm_a = np.linalg.norm(a, axis=0)   # (T,)
    norm_b = np.linalg.norm(b, axis=0)   # (T,)
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = dot_product / (norm_a * norm_b + 1e-8)
    similarity = np.nan_to_num(similarity)
    return np.mean(similarity)

def calculate_mrr(ori_data, gen_data_list, k=5, threshold=0.5):
    # ori_data: (batch, N, T)
    # gen_data_list: list of (batch, N, T)
    n_batch_size = ori_data.shape[0]
    n_generations = len(gen_data_list)
    k = min(k, n_generations)
    mrr_scores = np.zeros(n_batch_size)
    for batch_idx in range(n_batch_size):
        similarities = []
        for gen_idx in range(n_generations):
            real_sequence = ori_data[batch_idx]      # (N, T)
            generated_sequence = gen_data_list[gen_idx][batch_idx]  # (N, T)
            similarity = cosine_similarity_per_timestep(real_sequence, generated_sequence)
            similarities.append(similarity)
        sorted_indices = np.argsort(similarities)[::-1][:k]
        rank = None
        for r, idx in enumerate(sorted_indices):
            if similarities[idx] > threshold:
                rank = r + 1
                break
        mrr_scores[batch_idx] = 1.0 / rank if rank is not None else 0.0
    return np.mean(mrr_scores)

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    X_flat = X.reshape(X.shape[0], -1)
    Y_flat = Y.reshape(Y.shape[0], -1)
    m, n = X_flat.shape[0], Y_flat.shape[0]
    if kernel == 'rbf':
        XX = np.sum(X_flat**2, axis=1, keepdims=True)
        YY = np.sum(Y_flat**2, axis=1, keepdims=True)
        dist_XX = XX + XX.T - 2 * np.dot(X_flat, X_flat.T)
        dist_YY = YY.T + YY - 2 * np.dot(Y_flat, Y_flat.T)
        dist_XY = XX + YY.T - 2 * np.dot(X_flat, Y_flat.T)
        max_dist = 700 / gamma
        dist_XX = np.clip(dist_XX, 0, max_dist)
        dist_YY = np.clip(dist_YY, 0, max_dist)
        dist_XY = np.clip(dist_XY, 0, max_dist)
        K_XX = np.exp(-gamma * dist_XX)
        K_YY = np.exp(-gamma * dist_YY)
        K_XY = np.exp(-gamma * dist_XY)
        mmd_squared = (np.sum(K_XX) - np.trace(K_XX)) / (m * (m - 1)) + \
                      (np.sum(K_YY) - np.trace(K_YY)) / (n * (n - 1)) - \
                      2 * np.sum(K_XY) / (m * n)
        return np.sqrt(max(mmd_squared, 0))

def create_graph(data, adj):
    edge_indices = np.where(adj > 0)
    src_nodes = edge_indices[0]
    dst_nodes = edge_indices[1]
    edge_weights = adj[edge_indices]
    num_edges = len(src_nodes)
    num_timesteps = data.shape[2]
    graphlist = []
    for i in range(data.shape[0]):
        temdata = data[i]
        src_expanded = np.repeat(src_nodes, num_timesteps)
        dst_expanded = np.repeat(dst_nodes, num_timesteps)
        time_expanded = np.tile(np.arange(num_timesteps), num_edges)
        edge_weights_expanded = np.repeat(edge_weights, num_timesteps)
        src_features = temdata[src_expanded, time_expanded]
        dst_features = temdata[dst_expanded, time_expanded]
        msg_array = np.column_stack([src_features, dst_features, edge_weights_expanded])
        graph = TemporalData(
            src=Tensor(src_expanded),
            dst=Tensor(dst_expanded),
            t=Tensor(time_expanded),
            msg=Tensor(msg_array),
        )
        graphlist.append(graph)
    return graphlist

def evaluate(args, x_1, x_t_list, log):
    print_log(f"\n{'='*50}", log=log)
    print_log(f"Phase 4: Evaluation...", log=log)
    print_log(f"{'='*50}\n", log=log)
    
    device = args.device
    results = {}
    
    x_t = x_t_list[0]
    
    x_1_eval = x_1[:, :, 0, :] if x_1.ndim == 4 else x_1[:, :, 0] if x_1.ndim == 3 else x_1
    x_t_eval = x_t[:, :, 0, :] if x_t.ndim == 4 else x_t[:, :, 0] if x_t.ndim == 3 else x_t
    
    if x_1_eval.ndim == 2:
        x_1_eval = x_1_eval[:, :, np.newaxis]
        x_t_eval = x_t_eval[:, :, np.newaxis]
    
    x_1_t = np.transpose(x_1_eval, (0, 2, 1))
    x_t_t = np.transpose(x_t_eval, (0, 2, 1))
    
    x_t_list_processed = []
    for x_t_i in x_t_list:
        x_t_i_eval = x_t_i[:, :, 0, :] if x_t_i.ndim == 4 else x_t_i[:, :, 0] if x_t_i.ndim == 3 else x_t_i
        if x_t_i_eval.ndim == 2:
            x_t_i_eval = x_t_i_eval[:, :, np.newaxis]
        x_t_list_processed.append(np.transpose(x_t_i_eval, (0, 2, 1)))
    
    dataset_name = args.dataset_name.split('_')[0]
    adjpath = f'./Data/{dataset_name}/graph/{args.year}_adj.npz'
    use_jl = os.path.exists(adjpath)
    
    if use_jl:
        try:
            from jlmetric.metric import JLEvaluator
            adj = np.load(adjpath)['x']
            Evaluator = JLEvaluator(device=torch.device(device))
            bs = 4
        except ImportError:
            print_log("jlmetric not installed, skipping JL evaluation", log=log)
            use_jl = False
    else:
        print_log(f"Adjacency matrix not found at {adjpath}, skipping JL evaluation", log=log)
    
    for data_length in [24, 48, 96]:
        print_log(f"\n--- Evaluating with data_length={data_length} ---", log=log)
        
        ori = x_1_t[:, :, :data_length]
        gen = x_t_t[:, :, :data_length]
        
        ori_t = np.transpose(ori, (0, 2, 1))
        gen_t = np.transpose(gen, (0, 2, 1))
        
        mae = calculate_mae(ori_t, gen_t)
        mse = calculate_mse(ori_t, gen_t)
        wape = calculate_wape(ori_t, gen_t)
        ed = calculate_ed(ori, gen)
        
        gen_list_for_mrr = [x[:, :, :data_length] for x in x_t_list_processed]
        mrr = calculate_mrr(ori, gen_list_for_mrr, k=5, threshold=0.5)
        
        try:
            fid_model = initialize_ts2vec(ori, device)
            ori_repr = fid_model.encode(ori, encoding_window='full_series')
            gen_repr = fid_model.encode(gen, encoding_window='full_series')
            cfid = calculate_fid(ori_repr, gen_repr)
        except Exception as e:
            print_log(f"C-FID calculation failed: {e}", log=log)
            cfid = float('nan')
        
        jl_metric = float('nan')
        mmd = float('nan')
        if use_jl:
            refemb = []
            genemb = []
            sim = []
            for i in tqdm(range(0, ori.shape[0], bs), desc=f"JL-{data_length}"):
                end_idx = min(i + bs, ori.shape[0])
                if end_idx - i < bs:
                    break
                orggraphdata = create_graph(ori[i:end_idx], adj)
                gengraphdata = create_graph(gen[i:end_idx], adj)
                reference_embeddings, generated_embeddings, similarities = Evaluator.batch_eval(orggraphdata, gengraphdata)
                refemb.append(reference_embeddings.detach().cpu().numpy())
                genemb.append(generated_embeddings.detach().cpu().numpy())
                sim.append(similarities.detach().cpu().numpy())
                del orggraphdata, gengraphdata
            
            if len(sim) > 0:
                sim_flat = np.concatenate([s.flatten() for s in sim])
                jl_metric = np.mean(sim_flat)
                refemb_array = np.concatenate(refemb, axis=0)
                genemb_array = np.concatenate(genemb, axis=0)
                mmd = compute_mmd(refemb_array, genemb_array, kernel='rbf', gamma=0.001)
                np.savez_compressed(
                    os.path.join(args.generation_save_path_result, f'jl_metric_{data_length}.npz'),
                    refemb=refemb_array, genemb=genemb_array, sim=sim_flat
                )
        
        results[f'L{data_length}'] = {
            'MAE': mae, 'MSE': mse, 'WAPE': wape, 'ED': ed, 'C-FID': cfid, 'MRR@5': mrr, 'JL': jl_metric, 'MMD': mmd
        }
        
        print_log(f"MAE: {mae:.4f} | MSE: {mse:.4f} | WAPE: {wape:.4f} | ED: {ed:.4f} | C-FID: {cfid:.4f} | MRR@5: {mrr:.4f} | JL: {jl_metric:.4f} | MMD: {mmd:.4f}", log=log)
        print_log(f"{mae:.2f} & {mse:.2f} & {wape:.2f} & {ed:.2f} & {cfid:.2f} & {mrr:.2f} & {jl_metric:.2f} & {mmd:.2f}", log=log)
    
    return results

if __name__ == '__main__':
    seed_everything(42)
    args = get_args()
    
    log_path = f"./logs/{args.model_name}/{args.year}"
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f"{args.model_name}-{args.denoiser}-{args.backbone}-{args.dataset_name}.log")
    log = open(log_file, "a")
    
    print_log(f"Model: {args.model_name}", log=log)
    print_log(f"Pretrained model: {args.pretrained_model}", log=log)
    print_log(f"Denoiser: {args.denoiser}", log=log)
    print_log(f"Backbone: {args.backbone}", log=log)
    print_log(f"Dataset: {args.dataset_name}", log=log)
    
    total_start = time.time()
    
    pretrained_model = None
    if need_pretrain(args.pretrained_model):
        pretrained_model = pretrain(args, log)
    else:
        print_log(f"Model {args.model_name} does not require pretraining.", log=log)
    
    model = train_main(args, pretrained_model, log)
    
    args.generation_save_path_result = args.generation_save_path
    x_t_list = []
    for run_index in range(5):
        print_log(f"\n--- Inference run {run_index + 1}/5 ---", log=log)
        x_1, x_t = infer(args, model, log)
        x_t_list.append(x_t)
        np.save(os.path.join(args.generation_save_path_result, f'x_t_run{run_index}.npy'), x_t)
    
    evaluate(args, x_1, x_t_list, log)
    
    total_end = time.time()
    print_log(f"\nTotal pipeline time: {total_end - total_start:.2f} seconds", log=log)
    log.close()
