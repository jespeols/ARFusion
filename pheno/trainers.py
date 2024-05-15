import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time 
import matplotlib.pyplot as plt
import wandb
import os

from itertools import chain
from pathlib import Path
from torch.utils.data import DataLoader
from datetime import datetime

from utils import WeightedBCEWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
################################## Trainer for CLS task ##################################

class BertCLSTrainer(nn.Module):
    
    def __init__(self,
                 config: dict,
                 model,
                 antibiotics: list, # list of antibiotics in the dataset
                 train_set,
                 val_set,
                 results_dir: Path = None,
                 ):
        super(BertCLSTrainer, self).__init__()
        
        self.random_state = config["random_state"]
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        
        self.model = model
        self.project_name = config["project_name"]
        self.wandb_name = config["name"] if config["name"] else datetime.now().strftime("%Y%m%d-%H%M%S")
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics) 
        
        self.train_set, self.train_size = train_set, len(train_set)
        self.val_set, self.val_size = val_set, len(val_set) 
        assert round(self.val_size / (self.train_size + self.val_size), 2) == config["val_share"], "Validation set size does not match intended val_share"
        self.val_share, self.train_share = config["val_share"], 1 - config["val_share"]
        self.batch_size = config["batch_size"]
        self.num_batches = self.train_size // self.batch_size
         
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.patience = config["early_stopping_patience"]
        self.save_model = config["save_model"] if config["save_model"] else False
        
        self.mask_prob = self.train_set.mask_prob
        self.num_known_ab = self.train_set.num_known_ab
        self.num_known_classes = self.train_set.num_known_classes
        
        self.wl_strength = config["wl_strength"]  ## positive class weight for weighted BCELoss
        if self.wl_strength:
            self.ab_weights = config['data']['antibiotics']['ab_weights_'+self.wl_strength]
            self.ab_weights = {ab: v for ab, v in self.ab_weights.items() if ab in self.antibiotics}
            self.alphas = [v for v in self.ab_weights.values()]
        else:
            self.alphas = [0.5]*self.num_ab         ## equal class weights for all antibiotics
        self.criterions = [WeightedBCEWithLogitsLoss(alpha=alpha).to(device) for alpha in self.alphas]
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
                 
        self.current_epoch = 0
        self.report_every = config["report_every"] if config["report_every"] else 1000
        self.print_progress_every = config["print_progress_every"] if config["print_progress_every"] else 1000
        self._splitter_size = 70
        self.results_dir = results_dir
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True) 
        
        
    def print_model_summary(self):        
        print("Model summary:")
        print("="*self._splitter_size)
        print(f"Embedding dim: {self.model.emb_dim}")
        print(f"Feed-forward dim: {self.model.ff_dim}")
        print(f"Hidden dim: {self.model.hidden_dim}")
        print(f"Number of heads: {self.model.num_heads}")
        print(f"Number of encoder layers: {self.model.num_layers}")
        print(f"Dropout probability: {self.model.dropout_prob:.0%}")
        print(f"Max sequence length: {self.model.max_seq_len}")
        print(f"Vocab size: {len(self.train_set.vocab):,}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("="*self._splitter_size)
        
    
    def print_trainer_summary(self):
        print("Trainer summary:")
        print("="*self._splitter_size)
        if device.type == "cuda":
            print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Device: {device}")        
        print(f"Training dataset size: {self.train_size:,}")
        print(f"Number of antibiotics: {self.num_ab}")
        print(f"Antibiotics: {self.antibiotics}")
        if self.wl_strength:
            print("Antibiotic weights:", self.ab_weights)
        print(f"CV split: {self.train_share:.0%} train | {self.val_share:.0%} val")
        print(f"Masking method: {self.train_set.masking_method}")
        if self.mask_prob:
            print(f"Masking probability: {self.mask_prob:.0%}")
        if self.num_known_ab:
            print(f"Number of known antibiotics: {self.num_known_ab}")
        if self.num_known_classes:
            print(f"Number of known classes: {self.num_known_classes}")
        print(f"Number of epochs: {self.epochs}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {self.num_batches:,}")
        print(f"Learning rate: {self.lr}")
        print(f"Weight decay: {self.weight_decay}")
        print("="*self._splitter_size)
        
        
    def __call__(self):      
        self.wandb_run = self._init_wandb()
        self.val_set.prepare_dataset()
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        
        start_time = time.time()
        self.best_val_loss = float('inf') 
        self._init_result_lists()
        for self.current_epoch in range(self.current_epoch, self.epochs):
            self.model.train()
            # Dynamic masking: New mask for training set each epoch
            self.train_set.prepare_dataset()
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
            epoch_start_time = time.time()
            loss = self.train(self.current_epoch) # returns loss, averaged over batches
            self.losses.append(loss) 
            print(f"Epoch completed in {(time.time() - epoch_start_time)/60:.1f} min | Loss: {loss:.4f}")
            print("Evaluating on validation set...")
            val_results = self.evaluate(self.val_loader, self.val_set)
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            self._update_val_lists(val_results)
            self._report_epoch_results()
            early_stop = self.early_stopping()
            if early_stop:
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.4f}")
                s = f"Best validation loss {self.best_val_loss:.4f}"
                s += f" | Validation accuracy {self.val_accs[self.best_epoch]:.2%}"
                s += f" | Validation sequence accuracy {self.val_iso_accs[self.best_epoch]:.2%}"
                s += f" at epoch {self.best_epoch+1}"
                print(s)
                self.wandb_run.log({"Losses/final_val_loss": self.best_val_loss, 
                           "Accuracies/final_val_acc":self.val_accs[self.best_epoch],
                           "Accuracies/final_val_iso_acc": self.val_iso_accs[self.best_epoch],
                           "final_epoch": self.best_epoch+1})
                print("="*self._splitter_size)
                self.model.load_state_dict(self.best_model_state) 
                self.current_epoch = self.best_epoch
                break
            if self.scheduler:
                self.scheduler.step()
        if not early_stop:    
            self.wandb_run.log({"Losses/final_val_loss": self.val_losses[-1], 
                    "Accuracies/final_val_acc":self.val_accs[-1],
                    "Accuracies/final_val_iso_acc": self.val_iso_accs[-1],
                    "final_epoch": self.current_epoch+1})
        if self.save_model:
            self._save_model(self.results_dir / "model_state.pt") 
        train_time = (time.time() - start_time)/60
        self.wandb_run.log({"Training time (min)": train_time})
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        if not early_stop:
            s = f"Final validation loss {self.val_losses[-1]:.4f}"
            s += f" | Final validation accuracy {self.val_accs[-1]:.2%}"
            s += f" | Final validation sequence accuracy {self.val_iso_accs[-1]:.2%}"
            print(s)
        
        results = {
            "train_time": train_time,
            "best_epoch": self.current_epoch,
            "train_losses": self.losses,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "val_iso_accs": self.val_iso_accs,
            "ab_stats": self.val_ab_stats[self.best_epoch],
            "iso_stats": self.val_iso_stats[self.best_epoch]
        }
        return results
    
    
    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_loss = 0
        reporting_loss = 0
        printing_loss = 0
        for i, batch in enumerate(self.train_loader):
            batch_index = i + 1
            
            input, target_res, attn_mask = batch     
            self.optimizer.zero_grad() # zero out gradients
            pred_logits = self.model(input, attn_mask) # get predictions for all antibiotics
            
            ab_mask = target_res != -1 # (batch_size, num_ab), indicates which antibiotics are present in the batch
            ab_indices = ab_mask.any(dim=0).nonzero().squeeze(-1).tolist() # list of indices of antibiotics present in the batch
            losses = list()
            for j in ab_indices: 
                mask = ab_mask[:, j] # (batch_size,), indicates which samples contain the antibiotic masked
                # isolate the predictions and targets for the antibiotic
                ab_pred_logits = pred_logits[mask, j] # (num_masked_samples,)
                ab_targets = target_res[mask, j] # (num_masked_samples,)
                ab_loss = self.criterions[j](ab_pred_logits, ab_targets)
                losses.append(ab_loss)
            loss = sum(losses) / len(losses) # average loss over antibiotics
            epoch_loss += loss.item() 
            reporting_loss += loss.item()
            printing_loss += loss.item()
            
            loss.backward() 
            self.optimizer.step() 
            if batch_index % self.report_every == 0:
                self._report_loss_results(batch_index, reporting_loss)
                reporting_loss = 0 
                
            if batch_index % self.print_progress_every == 0:
                time_elapsed = time.gmtime(time.time() - time_ref) 
                self._print_loss_summary(time_elapsed, batch_index, printing_loss) 
                printing_loss = 0           
        avg_epoch_loss = epoch_loss / self.num_batches
        return avg_epoch_loss 
    
    
    def early_stopping(self):
        if self.val_losses[-1] < self.best_val_loss:
            self.best_val_loss = self.val_losses[-1]
            self.best_epoch = self.current_epoch
            self.best_model_state = self.model.state_dict()
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            return True if self.early_stopping_counter >= self.patience else False
        
            
    def evaluate(self, loader: DataLoader, ds_obj: pd.DataFrame, print_mode: bool = True):
        self.model.eval()
        # prepare evaluation statistics dataframes
        eval_stats_ab, eval_stats_iso = self._init_eval_stats(ds_obj)
        with torch.no_grad():
            loss = 0
            num = np.zeros((self.num_ab, 2)) # tracks the occurence for each antibiotic & resistance
            num_preds = np.zeros_like(num) # tracks the number of predictions for each antibiotic & resistance
            num_correct = np.zeros_like(num) # tracks the number of correct predictions for each antibiotic & resistance
            for batch_idx, batch in enumerate(loader):
                input, target_res, attn_mask = batch
                pred_logits = self.model(input, attn_mask) # get predictions for all antibiotics
                pred_res = torch.where(pred_logits > 0, torch.ones_like(pred_logits), torch.zeros_like(pred_logits))
                
                ab_mask = target_res != -1 # (batch_size, num_ab)
                eval_stats_iso = self._update_iso_stats(batch_idx, pred_res, target_res, ab_mask, eval_stats_iso)
                batch_loss = list()
                for j in range(self.num_ab): 
                    mask = ab_mask[:, j] 
                    if mask.any(): 
                        ab_pred_logits = pred_logits[mask, j] 
                        ab_targets = target_res[mask, j] 
                        num_R = ab_targets.sum().item()
                        num_S = ab_targets.shape[0] - num_R
                        num[j, :] += [num_S, num_R]
                        
                        ab_loss = self.criterions[j](ab_pred_logits, ab_targets)
                        batch_loss.append(ab_loss.item())
                        
                        ab_pred_res = pred_res[mask, j] 
                        num_correct[j, :] += self._get_num_correct(ab_pred_res, ab_targets)    
                        num_preds[j, :] += self._get_num_preds(ab_pred_res)
                loss += sum(batch_loss) / len(batch_loss) 
            loss /= len(loader) # average loss over batches
            acc = num_correct.sum() / num.sum() # overall accuracy
            iso_acc = eval_stats_iso['correct_all'].sum() / eval_stats_iso.shape[0] # accuracy over all sequences
            
            eval_stats_ab = self._update_ab_eval_stats(eval_stats_ab, num, num_preds, num_correct)
            
            eval_stats_iso['specificity'] = eval_stats_iso.apply(
                lambda row: row['correct_S']/row['num_masked_S'] if row['num_masked_S'] > 0 else np.nan, axis=1)
            eval_stats_iso['sensitivity'] = eval_stats_iso.apply(
                lambda row: row['correct_R']/row['num_masked_R'] if row['num_masked_R'] > 0 else np.nan, axis=1)
            eval_stats_iso['accuracy'] = eval_stats_iso.apply(
                lambda row: (row['correct_S'] + row['correct_R'])/row['num_masked'], axis=1)
        if print_mode:
            print(f"Loss: {loss:.4f} | Accuracy: {acc:.2%} | Isolate accuracy: {iso_acc:.2%}")
            print("="*self._splitter_size)
        
        results = {
            "loss": loss, 
            "acc": acc, 
            "iso_acc": iso_acc,
            "ab_stats": eval_stats_ab,
            "iso_stats": eval_stats_iso
        }
        return results
            
    
    def _init_result_lists(self):
        self.losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_iso_accs = []
        self.val_ab_stats = []
        self.val_iso_stats = []
        
        
    def _update_val_lists(self, results: dict):
        self.val_losses.append(results["loss"])
        self.val_accs.append(results["acc"])
        self.val_iso_accs.append(results["iso_acc"])
        self.val_ab_stats.append(results["ab_stats"])
        self.val_iso_stats.append(results["iso_stats"])
    
    
    def _init_eval_stats(self, ds_obj):
        eval_stats_ab = pd.DataFrame(columns=['antibiotic', 'num_tot', 'num_S', 'num_R', 'num_pred_S', 'num_pred_R', 
                                              'num_correct', 'num_correct_S', 'num_correct_R',
                                              'accuracy', 'sensitivity', 'specificity', 'precision', 'F1'])
        eval_stats_ab['antibiotic'] = self.antibiotics
        eval_stats_ab['num_tot'], eval_stats_ab['num_S'], eval_stats_ab['num_R'] = 0, 0, 0
        eval_stats_ab['num_pred_S'], eval_stats_ab['num_pred_R'] = 0, 0
        eval_stats_ab['num_correct'], eval_stats_ab['num_correct_S'], eval_stats_ab['num_correct_R'] = 0, 0, 0
    
        eval_stats_iso = ds_obj.ds.copy()
        eval_stats_iso['num_masked'], eval_stats_iso['num_masked_S'], eval_stats_iso['num_masked_R'] = 0, 0, 0
        eval_stats_iso['correct_S'], eval_stats_iso['correct_R'], eval_stats_iso['correct_all'] = 0, 0, False
        eval_stats_iso.drop(columns=['phenotypes'], inplace=True)
        
        return eval_stats_ab, eval_stats_iso
    
    
    def _update_ab_eval_stats(self, eval_stats_ab: pd.DataFrame, num, num_preds, num_correct):
        for j in range(self.num_ab): 
            eval_stats_ab.loc[j, 'num_tot'] = num[j, :].sum()
            eval_stats_ab.loc[j, 'num_S'], eval_stats_ab.loc[j, 'num_R'] = num[j, 0], num[j, 1]
            eval_stats_ab.loc[j, 'num_pred_S'], eval_stats_ab.loc[j, 'num_pred_R'] = num_preds[j, 0], num_preds[j, 1]
            eval_stats_ab.loc[j, 'num_correct'] = num_correct[j, :].sum()
            eval_stats_ab.loc[j, 'num_correct_S'], eval_stats_ab.loc[j, 'num_correct_R'] = num_correct[j, 0], num_correct[j, 1]
        eval_stats_ab['accuracy'] = eval_stats_ab.apply(
            lambda row: row['num_correct']/row['num_tot'] if row['num_tot'] > 0 else np.nan, axis=1)
        eval_stats_ab['sensitivity'] = eval_stats_ab.apply(
            lambda row: row['num_correct_R']/row['num_R'] if row['num_R'] > 0 else np.nan, axis=1)
        eval_stats_ab['specificity'] = eval_stats_ab.apply(
            lambda row: row['num_correct_S']/row['num_S'] if row['num_S'] > 0 else np.nan, axis=1)
        eval_stats_ab['precision'] = eval_stats_ab.apply(
            lambda row: row['num_correct_R']/row['num_pred_R'] if row['num_pred_R'] > 0 else np.nan, axis=1)
        eval_stats_ab['F1'] = eval_stats_ab.apply(
            lambda row: 2*row['precision']*row['sensitivity']/(row['precision']+row['sensitivity']) 
            if row['precision'] > 0 and row['sensitivity'] > 0 else np.nan, axis=1)
        return eval_stats_ab
    
    
    def _get_num_correct(self, pred_res: torch.Tensor, target_res: torch.Tensor):
        eq = torch.eq(pred_res, target_res)
        num_correct_S = eq[target_res == 0].sum().item()
        num_correct_R = eq[target_res == 1].sum().item()
        return [num_correct_S, num_correct_R]
    
    
    def _get_num_preds(self, pred_res: torch.Tensor):
        num_pred_S = (pred_res == 0).sum().item()
        num_pred_R = (pred_res == 1).sum().item()
        return [num_pred_S, num_pred_R]
    
    
    def _update_iso_stats(self, batch_index: int, pred_res: torch.Tensor, target_res: torch.Tensor, 
                          ab_mask: torch.Tensor, eval_stats_iso: pd.DataFrame):
        for i in range(pred_res.shape[0]): # for each isolate
            global_idx = batch_index * self.batch_size + i # index of the isolate in the dataframe
            iso_ab_mask = ab_mask[i]
            
            # counts
            eval_stats_iso.loc[global_idx, 'num_masked'] = int(iso_ab_mask.sum().item())
            iso_target_res = target_res[i][iso_ab_mask]
            num_R = iso_target_res.sum().item()
            num_S = iso_target_res.shape[0] - num_R
            eval_stats_iso.loc[global_idx, 'num_masked_S'] = num_S
            eval_stats_iso.loc[global_idx, 'num_masked_R'] = num_R
            
            # correct predictions
            iso_pred_res = pred_res[i][iso_ab_mask]
            eq = torch.eq(iso_pred_res, iso_target_res)
            num_R_correct = eq[iso_target_res == 1].sum().item()
            num_S_correct = eq[iso_target_res == 0].sum().item()
            eval_stats_iso.loc[global_idx, 'correct_S'] = num_S_correct
            eval_stats_iso.loc[global_idx, 'correct_R'] = num_R_correct
            
            eval_stats_iso.loc[global_idx, 'correct_all'] = bool(eq.all().item()) # 1 if all antibiotics are predicted correctly, 0 otherwise       
        return eval_stats_iso
    
     
    def _init_wandb(self):
        self.wandb_run = wandb.init(
            project=self.project_name, # name of the project
            name=self.wandb_name, # name of the run
            
            config={
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "emb_dim": self.model.emb_dim,
                'ff_dim': self.model.ff_dim,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "mask_prob": self.mask_prob,
                "num_known_ab": self.num_known_ab,
                "num_known_classes": self.num_known_classes,
                "max_seq_len": self.model.max_seq_len,
                "vocab_size": len(self.train_set.vocab),
                "num_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "num_antibiotics": self.num_ab,
                "antibiotics": self.antibiotics,
                "antibiotic weights:": self.ab_weights if self.wl_strength else None,
                "train_size": self.train_size,
                "random_state": self.random_state,
                'val_share': self.val_share,
                "val_size": self.val_size,
            }
        )
        self.wandb_run.watch(self.model) # watch the model for gradients and parameters
        self.wandb_run.define_metric("epoch", hidden=True)
        self.wandb_run.define_metric("batch", hidden=True)
        
        self.wandb_run.define_metric("Losses/live_loss", step_metric="batch")
        self.wandb_run.define_metric("Losses/train_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Losses/val_loss", summary="min", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_acc", summary="max", step_metric="epoch")
        self.wandb_run.define_metric("Accuracies/val_iso_acc", summary="max", step_metric="epoch")
        
        self.wandb_run.define_metric("Losses/final_val_loss")
        self.wandb_run.define_metric("Accuracies/final_val_acc")
        self.wandb_run.define_metric("Accuracies/final_val_iso_acc")
        self.wandb_run.define_metric("final_epoch")

        return self.wandb_run
     
    def _report_epoch_results(self):
        wandb_dict = {
            "epoch": self.current_epoch+1,
            "Losses/train_loss": self.losses[-1],
            "Losses/val_loss": self.val_losses[-1],
            "Accuracies/val_acc": self.val_accs[-1],
            "Accuracies/val_iso_acc": self.val_iso_accs[-1]
        }
        self.wandb_run.log(wandb_dict)
    
        
    def _report_loss_results(self, batch_index, tot_loss):
        avg_loss = tot_loss / self.report_every
        
        global_step = self.current_epoch * self.num_batches + batch_index # global step, total #batches seen
        self.wandb_run.log({"batch": global_step, "Losses/live_loss": avg_loss})
    
        
    def _print_loss_summary(self, time_elapsed, batch_index, tot_loss):
        progress = batch_index / self.num_batches
        mlm_loss = tot_loss / self.print_progress_every
          
        s = f"{time.strftime('%H:%M:%S', time_elapsed)}" 
        s += f" | Epoch: {self.current_epoch+1}/{self.epochs} | {batch_index}/{self.num_batches} ({progress:.2%}) | "\
                f"Loss: {mlm_loss:.4f}"
        print(s) 
    
    
    def _save_model(self, savepath: Path):
        torch.save(self.best_model_state, savepath)
        print(f"Model saved to {savepath}")
        print("="*self._splitter_size)
        
        
    def _load_model(self, savepath: Path):
        print("="*self._splitter_size)
        print(f"Loading model from {savepath}")
        self.model.load_state_dict(torch.load(savepath))
        self.model.to(device)
        print("Model loaded")
        print("="*self._splitter_size)