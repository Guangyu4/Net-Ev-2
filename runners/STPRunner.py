import torch
import numpy as np
import os
from tqdm import tqdm

from .AbstractRunner import AbstractRunner
from libs.utils import print_log

class STPRunner(AbstractRunner):
    def __init__(self, args):
        super().__init__

    def train_one_epoch(self):
        pass

    def eval_model(self, model, test_loader, device, save_dir, log=None):
        model.eval()
        batch_maes = []
        batch_mses = []
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                y_text, batch_x = data[0], data[1]
                if batch_x == None:
                    continue
                real_sample = batch_x.float().to(device)
                loss, recon_error, reconstructed_sample, z = model.shared_eval(
                    real_sample, None, mode='test')
                
                batch_mae = torch.mean(torch.abs(real_sample - reconstructed_sample)).cpu().item()
                batch_mse = torch.mean((real_sample - reconstructed_sample) ** 2).cpu().item()
                batch_maes.append(batch_mae)
                batch_mses.append(batch_mse)
                # print_log(f"batch_mae: {batch_mae}, batch_mse: {batch_mse}", log=log)
                # print_log(f"Evaluating model...{j} batches evaluated", log=log)
        
        mae = np.mean(batch_maes)
        mse = np.mean(batch_mses)
        rmse = np.sqrt(mse)
        return mae, rmse
    def predict(self):
        pass

    def test_model(self):
        pass

    def model_summary(self):
        pass

    def train(self, args, model, optimizer, train_loader, test_loader, save_dir, log=None):
        
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        best_train_loss = float('inf')
        best_model_state = None
        patience = 5
        patience_counter = 0
        
        for epoch in range(int(args.num_epochs)):
            train_loss = []
            pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch}')
            for i, data in enumerate(pbar):
                y_text, batch_x = data[0], data[1]
                tensor_all_data_in_batch = batch_x.clone().detach().float().to(device)
                loss, recon_error, x_recon, z = \
                    model.shared_eval(tensor_all_data_in_batch, optimizer, 'train')
                train_loss.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            epoch_train_loss = np.mean(train_loss).item()
            mae, rmse = self.eval_model(model, test_loader, device, save_dir, log=log)
            print_log(f"Epoch: {epoch}, Training Loss: {epoch_train_loss:.4g}, Eval MAE: {mae:.4g}, Eval RMSE: {rmse:.4g}", log=log)
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                print_log(f"No improvement in training loss. Patience counter: {patience_counter}/{patience}", log=log)
            
            if patience_counter >= patience:
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        final_model_path = os.path.join(save_dir, 'final_model.pth')
        torch.save(model, final_model_path)
        print_log(f"Best model saved to: {final_model_path}\nTraining complete.", log=log)

    def train_no_save(self, args, model, optimizer, train_loader, test_loader, log=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        best_train_loss = float('inf')
        best_model_state = None
        patience = 5
        patience_counter = 0
        
        for epoch in range(int(args.num_epochs)):
            train_loss = []
            pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch}')
            for i, data in enumerate(pbar):
                y_text, batch_x = data[0], data[1]
                tensor_all_data_in_batch = batch_x.clone().detach().float().to(device)
                loss, recon_error, x_recon, z = \
                    model.shared_eval(tensor_all_data_in_batch, optimizer, 'train')
                train_loss.append(loss.item())
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            epoch_train_loss = np.mean(train_loss).item()
            mae, rmse = self.eval_model(model, test_loader, device, None, log=log)
            print_log(f"Epoch: {epoch}, Training Loss: {epoch_train_loss:.4g}, Eval MAE: {mae:.4g}, Eval RMSE: {rmse:.4g}", log=log)
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                print_log(f"No improvement in training loss. Patience counter: {patience_counter}/{patience}", log=log)
            
            if patience_counter >= patience:
                break
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        print_log(f"Pretraining complete. Model kept in memory.", log=log)