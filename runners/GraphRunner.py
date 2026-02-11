
import torch
import numpy as np
import os
import math
from tqdm import tqdm

from .AbstractRunner import AbstractRunner
from libs.utils import print_log
from model.pretrained.bert import BertEmbedding

class GraphRunner(AbstractRunner):
    def __init__(self, args):
        super().__init__

    def train_one_epoch(self):
        pass

    def eval_model(self):
        pass

    def predict(self):
        pass

    def test_model(self):
        pass

    def model_summary(self):
        pass
    def get_inc_mask(self, x, nodeid, timeid):
        B, L, N = x.shape
        mask = torch.ones(B, L, N, device=x.device)
        nodeid = [item for sublist in nodeid for item in sublist if item != -1]
        mask[:, :, nodeid] = 0
        mask[:, :, :max(timeid)] = 0
        return mask
    def train(self, args, model, backbone, optimizer, scheduler, train_loader, log):
        loss_list = []
        start_epoch = 0
        # if from checkpoint:
        if args.checkpoint_path:
            checkpoint = torch.load(args.checkpoint_path, map_location=torch.device(args.device))
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            loss_list = checkpoint['loss_list']
        
        bert_embedder = BertEmbedding()
        best_train_loss = float('inf')
        best_model_state = None
        patience = 5
        patience_counter = 0
        
        for epoch in range(start_epoch, args.epochs):
            train_loss = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
            for batch, data in enumerate(pbar):
                y_text, x_1, nodeid, timeid = data
                y_text_embedding = bert_embedder.get_embedding(y_text)
                x_1 = x_1.float().to(args.device)

                x_1_encoded, _ = model.encoder(x_1.unsqueeze(-1).transpose(1, 2))
                x_1_encoded = x_1_encoded.transpose(1, 2)

                t = torch.floor(torch.rand(x_1_encoded.size(0)).to(args.device) * args.total_step).long()
                noise_gt = torch.randn_like(x_1_encoded).float().to(args.device)
                x_t, x0 = backbone.q_sample(x_1_encoded, t, noise_gt)

                optimizer.zero_grad()
                decide = torch.rand(1) < 0.3
                if decide:
                    y_text_embedding = None
                
                pred = model(input=x_t, t=t, text_input=y_text_embedding)

                loss = backbone.loss(pred, noise_gt)
                loss.backward()
                loss_list.append(loss.item())
                train_loss.append(loss.item())
                optimizer.step()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                scheduler.step()
            
            epoch_train_loss = np.mean(train_loss).item()
            print_log(f"Epoch: {epoch}, Training Loss: {epoch_train_loss:.4g}", log=log)
            
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                print_log(f"No improvement in training loss. Patience counter: {patience_counter}/{patience}", log=log)
            
            if epoch % 1 == 0 or epoch == args.epochs - 1:
                print(f'Saving model {epoch} to {args.save_path}...')
                save_dict = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                                    loss_list=loss_list)
                torch.save(save_dict, os.path.join(args.save_path, f'model_{epoch}.pth'))

        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f'Saving best model to {args.save_path}...')
            best_save_dict = dict(model=best_model_state, optimizer=optimizer.state_dict(), 
                                 epoch=args.epochs-1, loss_list=loss_list)
            torch.save(best_save_dict, os.path.join(args.save_path, 'best_model.pth'))
        return model