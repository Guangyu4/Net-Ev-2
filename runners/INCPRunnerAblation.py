import torch
import numpy as np
import os
from tqdm import tqdm

from .AbstractRunner import AbstractRunner
from libs.utils import print_log


class INCPRunnerAblation(AbstractRunner):
    """Ablation variant of INCPRunner that handles datasets returning 6 values"""
    
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
                if len(data) == 6:
                    y_text, batch_x, nodeid, timeid, event_type, weather_type = data
                elif len(data) == 5:
                    y_text, batch_x, nodelist, event_type, weather_type = data
                    nodeid = nodelist
                    timeid = [0] * len(event_type)
                else:
                    y_text, batch_x, nodeid, timeid = data
                
                if batch_x is None:
                    continue
                real_sample = batch_x.float().to(device)
                loss, recon_error, reconstructed_sample, z = model.shared_eval(
                    real_sample, None, mode='test', nodeid=nodeid, timeid=timeid)
                
                batch_mae = torch.mean(torch.abs(real_sample - reconstructed_sample)).cpu().item()
                batch_mse = torch.mean((real_sample - reconstructed_sample) ** 2).cpu().item()
                batch_maes.append(batch_mae)
                batch_mses.append(batch_mse)
        
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
        patience = args.num_epochs
        patience_counter = 0
        
        for epoch in range(int(args.num_epochs)):
            model.train()
            with torch.no_grad():
                param_norm_before = 0.0
                for p in model.parameters():
                    param_norm_before += p.norm().item()
            train_loss = []
            skipped = 0
            updated = 0
            pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch}')
            for i, data in enumerate(pbar):
                if len(data) == 6:
                    y_text, batch_x, nodeid, timeid, event_type, weather_type = data
                elif len(data) == 5:
                    y_text, batch_x, nodelist, event_type, weather_type = data
                    nodeid = nodelist
                    timeid = [0] * len(event_type)
                else:
                    y_text, batch_x, nodeid, timeid = data
                
                tensor_all_data_in_batch = batch_x.clone().detach().float().to(device)
                tensor_all_data_in_batch = torch.nan_to_num(tensor_all_data_in_batch, nan=0.0)
                loss, recon_error, x_recon, z = \
                    model.shared_eval(tensor_all_data_in_batch, optimizer, 'train', nodeid, timeid)
                if torch.isnan(loss) or torch.isinf(loss):
                    print_log(f"Warning: NaN/Inf loss in train at batch {i}, skipping...", log=log)
                    skipped += 1
                    continue
                if getattr(model, '_skip_step', False):
                    skipped += 1
                    continue
                train_loss.append(loss.item())
                updated += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            epoch_train_loss = np.mean(train_loss).item() if train_loss else float('inf')
            with torch.no_grad():
                param_norm_after = 0.0
                for p in model.parameters():
                    param_norm_after += p.norm().item()
            print_log(f"Param norm delta: {param_norm_after - param_norm_before:.6f}", log=log)
            print_log(f"Update steps: {updated}, Skipped steps: {skipped}", log=log)
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
            torch.save(model, os.path.join(save_dir, 'final_model.pth'))
            print_log(f"Initial model saved to: {os.path.join(save_dir, 'final_model.pth')}", log=log)
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        final_model_path = os.path.join(save_dir, f'model_{epoch}.pth')
        torch.save(model, os.path.join(save_dir, 'final_model.pth'))
        print_log(f"Best model saved to: {final_model_path}\nTraining complete.", log=log)

    def train_no_save(self, args, model, optimizer, train_loader, test_loader, log=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        best_train_loss = float('inf')
        best_model_state = None
        patience = args.num_epochs
        patience_counter = 0
        
        for epoch in range(int(args.num_epochs)):
            model.train()
            with torch.no_grad():
                param_norm_before = 0.0
                for p in model.parameters():
                    param_norm_before += p.norm().item()
            train_loss = []
            skipped = 0
            updated = 0
            pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch}')
            for i, data in enumerate(pbar):
                if len(data) == 6:
                    y_text, batch_x, nodeid, timeid, event_type, weather_type = data
                elif len(data) == 5:
                    y_text, batch_x, nodelist, event_type, weather_type = data
                    nodeid = nodelist
                    timeid = [0] * len(event_type)
                else:
                    y_text, batch_x, nodeid, timeid = data
                
                tensor_all_data_in_batch = batch_x.clone().detach().float().to(device)
                tensor_all_data_in_batch = torch.nan_to_num(tensor_all_data_in_batch, nan=0.0)
                loss, recon_error, x_recon, z = \
                    model.shared_eval(tensor_all_data_in_batch, optimizer, 'train', nodeid, timeid)
                if torch.isnan(loss) or torch.isinf(loss):
                    print_log(f"Warning: NaN/Inf loss in train_no_save at batch {i}, skipping...", log=log)
                    skipped += 1
                    continue
                if getattr(model, '_skip_step', False):
                    skipped += 1
                    continue
                train_loss.append(loss.item())
                updated += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            epoch_train_loss = np.mean(train_loss).item() if train_loss else float('inf')
            with torch.no_grad():
                param_norm_after = 0.0
                for p in model.parameters():
                    param_norm_after += p.norm().item()
            print_log(f"Param norm delta: {param_norm_after - param_norm_before:.6f}", log=log)
            print_log(f"Update steps: {updated}, Skipped steps: {skipped}", log=log)
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
