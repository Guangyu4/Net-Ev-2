
import torch
import numpy as np
import os
import math
from tqdm import tqdm

from .AbstractRunner import AbstractRunner
from libs.utils import print_log
from model.pretrained.bert import BertEmbedding

class DiffusionTSRunner(AbstractRunner):
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

    def train(self, args, model, backbone, optimizer, scheduler, train_loader, log=None):
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
                if len(data) == 3:
                    y_text, x_1, nodelist = data
                else:
                    y_text, x_1 = data
                y_text_embedding = bert_embedder.get_embedding(y_text)
                x_1 = x_1.float().to(args.device)

                t = torch.floor(torch.rand(x_1.size(0)).to(args.device) * args.total_step).long()
                noise_gt = torch.randn_like(x_1).float().to(args.device)
                x_t, x0 = backbone.q_sample(x_1, t, noise_gt)

                optimizer.zero_grad()
                decide = torch.rand(1) < 0.3
                if decide:
                    y_text_embedding = None
                pred = model(input=x_t, t=t, text_input=y_text_embedding)

                loss = backbone.loss(pred, x0)
                fft1 = torch.fft.fft(pred.transpose(1, 2), norm='forward')
                fft2 = torch.fft.fft(x0.transpose(1, 2), norm='forward')
                fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
                fourier_loss = backbone.loss(torch.real(fft1), torch.real(fft2))\
                                + backbone.loss(torch.imag(fft1), torch.imag(fft2))
                loss += 0.1 * fourier_loss  

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
            
            if epoch % 1000 == 0 or epoch == args.epochs - 1:
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
