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
from sklearn.metrics import auc, roc_curve

from utils import WeightedBCEWithLogitsLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
################################## Trainer for CLS task ##################################

class BertTrainer(nn.Module):
    
    def __init__(self,
                 config: dict,
                 model,
                 antibiotics: list, # list of antibiotics in the dataset
                 train_set,
                 val_set,
                 results_dir: Path = None,
                 ):
        super(BertTrainer, self).__init__()
        
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
        self.save_model_ = config["save_model"] if config["save_model"] else False
        
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
            val_start = time.time()
            val_results = self.evaluate(self.val_loader, self.val_set)
            if time.time() - val_start > 60:
                    disp_time = f"{(time.time() - val_start)/60:.1f} min"
            else:
                    disp_time = f"{time.time() - val_start:.0f} sec"
            print(f"Validation completed in " + disp_time)
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            self._update_val_lists(val_results)
            self._report_epoch_results()
            early_stop = self.early_stopping()
            if early_stop:
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.4f}")
                s = f"Best validation loss {self.best_val_loss:.4f}"
                s += f" | Validation accuracy {self.val_accs[self.best_epoch]:.2%}"
                s += f" | Validation isolate accuracy {self.val_iso_accs[self.best_epoch]:.2%}"
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
        if self.save_model_:
            self.save_model(self.results_dir / "model_state.pt") 
        train_time = (time.time() - start_time)/60
        self.wandb_run.log({"Training time (min)": train_time})
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        if not early_stop:
            s = f"Final validation loss {self.val_losses[-1]:.4f}"
            s += f" | Final validation accuracy {self.val_accs[-1]:.2%}"
            s += f" | Final validation isolate accuracy {self.val_iso_accs[-1]:.2%}"
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
        ab_stats, iso_stats = self._init_eval_stats(ds_obj)
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
                iso_stats = self._update_iso_stats(batch_idx, pred_res, target_res, ab_mask, iso_stats)
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
            iso_acc = iso_stats['correct_all'].sum() / iso_stats.shape[0] # accuracy over all sequences
            
            ab_stats = self._update_ab_eval_stats(ab_stats, num, num_preds, num_correct)
            
            iso_stats['specificity'] = iso_stats.apply(
                lambda row: row['correct_S']/row['num_masked_S'] if row['num_masked_S'] > 0 else np.nan, axis=1)
            iso_stats['sensitivity'] = iso_stats.apply(
                lambda row: row['correct_R']/row['num_masked_R'] if row['num_masked_R'] > 0 else np.nan, axis=1)
            iso_stats['accuracy'] = iso_stats.apply(
                lambda row: (row['correct_S'] + row['correct_R'])/row['num_masked'], axis=1)
        if print_mode:
            print(f"Loss: {loss:.4f} | Accuracy: {acc:.2%} | Isolate accuracy: {iso_acc:.2%}")
            print("="*self._splitter_size)
        
        results = {
            "loss": loss, 
            "acc": acc, 
            "iso_acc": iso_acc,
            "ab_stats": ab_stats,
            "iso_stats": iso_stats
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
        ab_stats = pd.DataFrame(columns=['antibiotic', 'num_tot', 'num_S', 'num_R', 'num_pred_S', 'num_pred_R', 
                                              'num_correct', 'num_correct_S', 'num_correct_R',
                                              'accuracy', 'sensitivity', 'specificity', 'precision', 'F1'])
        ab_stats['antibiotic'] = self.antibiotics
        ab_stats['num_tot'], ab_stats['num_S'], ab_stats['num_R'] = 0, 0, 0
        ab_stats['num_pred_S'], ab_stats['num_pred_R'] = 0, 0
        ab_stats['num_correct'], ab_stats['num_correct_S'], ab_stats['num_correct_R'] = 0, 0, 0
    
        iso_stats = ds_obj.ds.copy()
        iso_stats['num_masked'], iso_stats['num_masked_S'], iso_stats['num_masked_R'] = 0, 0, 0
        iso_stats['correct_S'], iso_stats['correct_R'], iso_stats['correct_all'] = 0, 0, False
        iso_stats.drop(columns=['phenotypes'], inplace=True)
        
        return ab_stats, iso_stats
    
    
    def _update_ab_eval_stats(self, ab_stats: pd.DataFrame, num, num_preds, num_correct):
        for j in range(self.num_ab): 
            ab_stats.loc[j, 'num_tot'] = num[j, :].sum()
            ab_stats.loc[j, 'num_S'], ab_stats.loc[j, 'num_R'] = num[j, 0], num[j, 1]
            ab_stats.loc[j, 'num_pred_S'], ab_stats.loc[j, 'num_pred_R'] = num_preds[j, 0], num_preds[j, 1]
            ab_stats.loc[j, 'num_correct'] = num_correct[j, :].sum()
            ab_stats.loc[j, 'num_correct_S'], ab_stats.loc[j, 'num_correct_R'] = num_correct[j, 0], num_correct[j, 1]
        ab_stats['accuracy'] = ab_stats.apply(
            lambda row: row['num_correct']/row['num_tot'] if row['num_tot'] > 0 else np.nan, axis=1)
        ab_stats['sensitivity'] = ab_stats.apply(
            lambda row: row['num_correct_R']/row['num_R'] if row['num_R'] > 0 else np.nan, axis=1)
        ab_stats['specificity'] = ab_stats.apply(
            lambda row: row['num_correct_S']/row['num_S'] if row['num_S'] > 0 else np.nan, axis=1)
        ab_stats['precision'] = ab_stats.apply(
            lambda row: row['num_correct_R']/row['num_pred_R'] if row['num_pred_R'] > 0 else np.nan, axis=1)
        ab_stats['F1'] = ab_stats.apply(
            lambda row: 2*row['precision']*row['sensitivity']/(row['precision']+row['sensitivity']) 
            if row['precision'] > 0 and row['sensitivity'] > 0 else np.nan, axis=1)
        return ab_stats
    
    
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
                          ab_mask: torch.Tensor, iso_stats: pd.DataFrame):
        for i in range(pred_res.shape[0]): # for each isolate
            global_idx = batch_index * self.batch_size + i # index of the isolate in the dataframe
            iso_ab_mask = ab_mask[i]
            
            # counts
            iso_stats.loc[global_idx, 'num_masked'] = int(iso_ab_mask.sum().item())
            iso_target_res = target_res[i][iso_ab_mask]
            num_R = iso_target_res.sum().item()
            num_S = iso_target_res.shape[0] - num_R
            iso_stats.loc[global_idx, 'num_masked_S'] = num_S
            iso_stats.loc[global_idx, 'num_masked_R'] = num_R
            
            # correct predictions
            iso_pred_res = pred_res[i][iso_ab_mask]
            eq = torch.eq(iso_pred_res, iso_target_res)
            num_R_correct = eq[iso_target_res == 1].sum().item()
            num_S_correct = eq[iso_target_res == 0].sum().item()
            iso_stats.loc[global_idx, 'correct_S'] = num_S_correct
            iso_stats.loc[global_idx, 'correct_R'] = num_R_correct
            
            iso_stats.loc[global_idx, 'correct_all'] = bool(eq.all().item()) # 1 if all antibiotics are predicted correctly, 0 otherwise       
        return iso_stats
    
     
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
    
    
    def save_model(self, savepath: Path):
        torch.save(self.best_model_state, savepath)
        print(f"Model saved to {savepath}")
        print("="*self._splitter_size)
        
        
    def load_model(self, savepath: Path):
        print("="*self._splitter_size)
        print(f"Loading model from {savepath}")
        self.model.load_state_dict(torch.load(savepath))
        self.model.to(device)
        print("Model loaded")
        print("="*self._splitter_size)
        


class BertTuner(nn.Module):
    
    def __init__(self,
                 config: dict,
                 model,
                 antibiotics: list, # list of antibiotics in the dataset
                 train_set,
                 val_set,
                 results_dir: Path = None,
                 ds_size: int = None,
                 CV_mode: bool = False,
                 ):
        super(BertTuner, self).__init__()
        
        config_ft = config["fine_tuning"]
        self.random_state = config_ft["random_state"]
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        
        self.model = model
        self.project_name = config_ft["project_name"]
        self.wandb_name = config_ft["name"] if config_ft["name"] else datetime.now().strftime("%Y%m%d-%H%M%S")
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics) 
        
        self.train_set, self.train_size = train_set, len(train_set)
        self.val_set, self.val_size = val_set, len(val_set) 
        if ds_size:
            self.dataset_size = ds_size
        else:
            self.dataset_size = self.train_size + self.val_size
        self.val_share, self.train_share = self.val_size / self.dataset_size, self.train_size / self.dataset_size
        self.batch_size = config_ft["batch_size"]
        self.num_batches = round(self.train_size / self.batch_size)
        self.val_batch_size = self.batch_size * 64
         
        self.lr = config_ft["lr"]
        self.weight_decay = config_ft["weight_decay"]
        self.epochs = config_ft["epochs"]
        self.patience = config_ft["early_stopping_patience"]
        self.save_model_ = config_ft["save_model"] if config_ft["save_model"] else False
        
        self.masking_method = self.train_set.masking_method
        self.mask_prob = self.train_set.mask_prob
        self.num_known_ab = self.train_set.num_known_ab
        self.num_known_classes = self.train_set.num_known_classes
        
        self.loss_fn = config_ft["loss_fn"]
        self.wl_strength = config_ft["wl_strength"]  ## positive class weight for weighted BCELoss
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
        self.CV_mode = CV_mode
        self.report_every = config_ft["report_every"] if config_ft["report_every"] else 1000
        self.print_progress_every = config_ft["print_progress_every"] if config_ft["print_progress_every"] else 1000
        self._splitter_size = 70
        self.exp_folder = config_ft["exp_folder"]
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
        print(f"Masking method: {self.masking_method}")
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
        self.val_loader = DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False)
        
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
            val_start = time.time()
            val_results = self.evaluate(self.val_loader, self.val_set)
            if time.time() - val_start > 60:
                    disp_time = f"{(time.time() - val_start)/60:.1f} min"
            else:
                    disp_time = f"{time.time() - val_start:.0f} sec"
            print(f"Validation completed in " + disp_time)
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            self._update_val_lists(val_results)
            if not self.CV_mode:
                self._report_epoch_results()
            early_stop = self.early_stopping()
            if early_stop:
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.4f}")
                s = f"Best validation loss {self.best_val_loss:.4f}"
                s += f" | Validation accuracy {self.val_accs[self.best_epoch]:.2%}"
                s += f" | Validation isolate accuracy {self.val_iso_accs[self.best_epoch]:.2%}"
                s += f" at epoch {self.best_epoch+1}"
                print(s)
                if not self.CV_mode:
                    self.wandb_run.log({
                        "Losses/final_val_loss": self.best_val_loss, 
                        "Accuracies/final_val_acc": self.val_accs[self.best_epoch],
                        "Accuracies/final_val_iso_acc": self.val_iso_accs[self.best_epoch],
                        "Class_metrics/final_val_sens": self.val_sensitivities[self.best_epoch],
                        "Class_metrics/final_val_spec": self.val_specificities[self.best_epoch],
                        "Class_metrics/final_val_F1": self.val_F1_scores[self.best_epoch],
                        "Class_metrics/final_val_auc_score": self.val_auc_scores[self.best_epoch],
                        "best_epoch": self.best_epoch+1
                    })
                print("="*self._splitter_size)
                self.model.load_state_dict(self.best_model_state) 
                self.current_epoch = self.best_epoch
                break
            if self.scheduler:
                self.scheduler.step()
        if not early_stop and not self.CV_mode: 
            self.wandb_run.log({
                    "Losses/final_val_loss": self.best_val_loss, 
                    "Accuracies/final_val_acc": self.val_accs[-1],
                    "Accuracies/final_val_iso_acc": self.val_iso_accs[-1],
                    "Class_metrics/final_val_sens": self.val_sensitivities[self.best_epoch],
                    "Class_metrics/final_val_spec": self.val_specificities[self.best_epoch],
                    "Class_metrics/final_val_F1": self.val_F1_scores[self.best_epoch],
                    "Class_metrics/final_val_auc_score": self.val_auc_scores[self.best_epoch],
                    "best_epoch": self.current_epoch+1
                })
        if self.save_model_:
            self.save_model(self.results_dir / "model_state.pt") 
        train_time = (time.time() - start_time)/60
        self.wandb_run.log({"Training time (min)": train_time})
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        if not early_stop:
            s = f"Final validation loss {self.val_losses[-1]:.4f}"
            s += f" | Final validation accuracy {self.val_accs[-1]:.2%}"
            s += f" | Final validation isolate accuracy {self.val_iso_accs[-1]:.2%}"
            print(s)
        
        results = {
            "train_loss": self.losses[self.best_epoch],
            "loss": self.val_losses[self.best_epoch],
            "acc": self.val_accs[self.best_epoch],
            "iso_acc": self.val_iso_accs[self.best_epoch],
            "sens": self.val_sensitivities[self.best_epoch],
            "spec": self.val_specificities[self.best_epoch],
            "prec": self.val_precisions[self.best_epoch],
            "F1": self.val_F1_scores[self.best_epoch],
            "auc_score": self.val_auc_scores[self.best_epoch],
            "roc": self.val_roc_results[self.best_epoch],
            "iso_stats": self.val_iso_stats[self.best_epoch],
            "ab_stats": self.val_ab_stats[self.best_epoch]
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
        ab_stats, iso_stats = self._init_eval_stats(ds_obj)
        with torch.no_grad():
            num = np.zeros((self.num_ab, 2)) # tracks the occurence for each antibiotic & resistance
            num_preds = np.zeros_like(num) # tracks the number of predictions for each antibiotic & resistance
            num_correct = np.zeros_like(num) # tracks the number of correct predictions for each antibiotic & resistance
            pred_sigmoids = torch.tensor([]).to(device)
            target_resistances = torch.tensor([]).to(device)
            loss = 0
            for batch_idx, batch in enumerate(loader):
                input, target_res, attn_mask = batch
                pred_logits = self.model(input, attn_mask) # get predictions for all antibiotics
                
                ###### ROC ######
                pred_sigmoids = torch.cat((pred_sigmoids, torch.sigmoid(pred_logits)), dim=0) # (+batch_size, num_ab)
                target_resistances = torch.cat((target_resistances, target_res), dim=0)
                
                pred_res = torch.where(pred_logits > 0, torch.ones_like(pred_logits), torch.zeros_like(pred_logits))
                ab_mask = target_res != -1 # (batch_size, num_ab)
                iso_stats = self._update_iso_stats(batch_idx, pred_res, target_res, ab_mask, iso_stats)
                losses = list()
                for j in range(self.num_ab): 
                    mask = ab_mask[:, j] 
                    if mask.any(): 
                        ab_pred_logits = pred_logits[mask, j] 
                        ab_targets = target_res[mask, j] 
                        num_masked_R = ab_targets.sum().item()
                        num_masked_S = ab_targets.shape[0] - num_masked_R
                        num[j, :] += [num_masked_S, num_masked_R]
                        
                        ab_loss = self.criterions[j](ab_pred_logits, ab_targets)
                        losses.append(ab_loss)
                        
                        ab_pred_res = pred_res[mask, j] 
                        num_correct[j, :] += self._get_num_correct(ab_pred_res, ab_targets)    
                        num_preds[j, :] += self._get_num_preds(ab_pred_res)
                loss += sum(losses) / len(losses) 
            avg_loss = loss.item() / len(loader) # average loss over batches
            
            pred_sigmoids = pred_sigmoids.cpu().numpy()
            target_resistances = target_resistances.cpu().numpy()
            roc_results = self._get_roc_results(pred_sigmoids, target_resistances)
            
            ab_stats = self._update_ab_eval_stats(ab_stats, num, num_preds, num_correct)
            iso_stats = self._calculate_iso_stats(iso_stats)
            
            acc = ab_stats['num_correct'].sum() / ab_stats['num_masked_tot'].sum()
            iso_acc = iso_stats['all_correct'].sum() / iso_stats.shape[0]
            sens = ab_stats['num_correct_R'].sum() / ab_stats['num_masked_R'].sum() 
            spec = ab_stats['num_correct_S'].sum() / ab_stats['num_masked_S'].sum()
            prec = ab_stats['num_correct_R'].sum() / ab_stats['num_pred_R'].sum()
            F1_score = 2 * sens * prec / (sens + prec)
        if print_mode:
            print(f"Loss: {avg_loss:.4f} | Accuracy: {acc:.2%} | Isolate accuracy: {iso_acc:.2%}")
            print("="*self._splitter_size)
        
        results = {
                "loss": avg_loss, 
                "acc": acc,
                "iso_acc": iso_acc,
                "sensitivity": sens,
                "specificity": spec,
                "precision": prec,
                "F1": F1_score,
                "ab_stats": ab_stats,
                "iso_stats": iso_stats,
                "roc_results": roc_results,
                "auc_score": roc_results["auc_score"]
            }
        return results
            
    
    def _init_result_lists(self):
        self.losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_iso_accs = []
        self.val_sensitivities = []
        self.val_specificities = []
        self.val_precisions = []
        self.val_F1_scores = []
        self.val_ab_stats = []
        self.val_iso_stats = []
        self.val_roc_results = []
        self.val_auc_scores = []
        
        
    def _update_val_lists(self, results: dict):
        self.val_losses.append(results["loss"])
        self.val_accs.append(results["acc"])
        self.val_iso_accs.append(results["iso_acc"])
        self.val_sensitivities.append(results["sensitivity"])
        self.val_specificities.append(results["specificity"])
        self.val_precisions.append(results["precision"])
        self.val_F1_scores.append(results["F1"])
        self.val_ab_stats.append(results["ab_stats"])
        self.val_iso_stats.append(results["iso_stats"])
        self.val_roc_results.append(results["roc_results"])
        self.val_auc_scores.append(results["auc_score"])
    
    
    def _init_eval_stats(self, ds_obj):
        ab_stats = pd.DataFrame(columns=[
            'antibiotic', 'num_masked_tot', 'num_masked_S', 'num_masked_R', 'num_pred_S', 'num_pred_R', 'num_correct', 'num_correct_S',
            'num_correct_R', 'accuracy', 'sensitivity', 'specificity', 'precision', 'F1'
        ])
        ab_stats['antibiotic'] = self.antibiotics
        ab_stats['num_masked_tot'], ab_stats['num_masked_S'], ab_stats['num_masked_R'] = 0, 0, 0
        ab_stats['num_pred_S'], ab_stats['num_pred_R'] = 0, 0
        ab_stats['num_correct'], ab_stats['num_correct_S'], ab_stats['num_correct_R'] = 0, 0, 0
        ab_stats['auc_score'] = 0.0
        ab_stats['roc_fpr'], ab_stats['roc_tpr'], ab_stats['roc_thresholds'] = None, None, None
    
        iso_stats = ds_obj.ds.copy()
        iso_stats['num_masked_ab'], iso_stats['num_masked_S'], iso_stats['num_masked_R'] = 0, 0, 0
        iso_stats['num_correct'], iso_stats['correct_S'], iso_stats['correct_R'] = 0, 0, 0
        iso_stats['sensitivity'], iso_stats['specificity'], iso_stats['accuracy'] = 0, 0, 0
        iso_stats['all_correct'] = False  
        iso_stats.drop(columns=['phenotypes'], inplace=True)
        
        return ab_stats, iso_stats
    
    
    def _update_ab_eval_stats(self, ab_stats: pd.DataFrame, num, num_preds, num_correct):
        for j in range(self.num_ab): 
            ab_stats.loc[j, 'num_masked_tot'] = num[j, :].sum()
            ab_stats.loc[j, 'num_masked_S'], ab_stats.loc[j, 'num_masked_R'] = num[j, 0], num[j, 1]
            ab_stats.loc[j, 'num_pred_S'], ab_stats.loc[j, 'num_pred_R'] = num_preds[j, 0], num_preds[j, 1]
            ab_stats.loc[j, 'num_correct'] = num_correct[j, :].sum()
            ab_stats.loc[j, 'num_correct_S'], ab_stats.loc[j, 'num_correct_R'] = num_correct[j, 0], num_correct[j, 1]
        ab_stats['accuracy'] = ab_stats.apply(
            lambda row: row['num_correct']/row['num_masked_tot'] if row['num_masked_tot'] > 0 else np.nan, axis=1)
        ab_stats['sensitivity'] = ab_stats.apply(
            lambda row: row['num_correct_R']/row['num_masked_R'] if row['num_masked_R'] > 0 else np.nan, axis=1)
        ab_stats['specificity'] = ab_stats.apply(
            lambda row: row['num_correct_S']/row['num_masked_S'] if row['num_masked_S'] > 0 else np.nan, axis=1)
        ab_stats['precision'] = ab_stats.apply(
            lambda row: row['num_correct_R']/row['num_pred_R'] if row['num_pred_R'] > 0 else np.nan, axis=1)
        ab_stats['F1'] = ab_stats.apply(
            lambda row: 2*row['precision']*row['sensitivity']/(row['precision']+row['sensitivity']) 
            if row['precision'] > 0 and row['sensitivity'] > 0 else np.nan, axis=1)
        return ab_stats
    
    
    def _calculate_iso_stats(self, iso_stats: pd.DataFrame): 
        iso_stats['accuracy'] = iso_stats['num_correct'] / iso_stats['num_masked_ab']
        iso_stats['sensitivity'] = iso_stats.apply(
            lambda row: row['correct_R']/row['num_masked_R'] if row['num_masked_R'] > 0 else np.nan, axis=1
        )
        iso_stats['specificity'] = iso_stats.apply(
            lambda row: row['correct_S']/row['num_masked_S'] if row['num_masked_S'] > 0 else np.nan, axis=1
        )
        return iso_stats
    
    
    def _get_num_correct(self, pred_res: torch.Tensor, target_res: torch.Tensor):
        eq = torch.eq(pred_res, target_res)
        num_correct_S = eq[target_res == 0].sum().item()
        num_correct_R = eq[target_res == 1].sum().item()
        return [num_correct_S, num_correct_R]
    
    
    def _get_num_preds(self, pred_res: torch.Tensor):
        num_pred_S = (pred_res == 0).sum().item()
        num_pred_R = (pred_res == 1).sum().item()
        return [num_pred_S, num_pred_R]
    
    
    def _get_roc_results(self, pred_sigmoids: np.ndarray, target_resistances: np.ndarray, drop_intermediate: bool = False):
        pred_sigmoids_flat = pred_sigmoids[target_resistances >= 0]
        target_resistances_flat = target_resistances[target_resistances >= 0]
        assert pred_sigmoids_flat.shape == target_resistances_flat.shape, "Shapes do not match"
        assert len(pred_sigmoids_flat.shape) == 1, "Only 1D arrays are supported"
        
        fpr, tpr, thresholds = roc_curve(target_resistances_flat, pred_sigmoids_flat, drop_intermediate=drop_intermediate)
        auc_score = auc(fpr, tpr)
        roc_results = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc_score": auc_score}
        
        return roc_results
    
    
    def _update_iso_stats(self, batch_index: int, pred_res: torch.Tensor, target_res: torch.Tensor, 
                          ab_mask: torch.Tensor, iso_stats: pd.DataFrame):
        for i in range(pred_res.shape[0]): # for each isolate
            global_idx = batch_index * self.batch_size + i # index of the isolate in the dataframe
            iso_ab_mask = ab_mask[i]
            
            # counts
            iso_stats.loc[global_idx, 'num_masked_ab'] = int(iso_ab_mask.sum().item())
            iso_target_res = target_res[i][iso_ab_mask]
            num_masked_R = iso_target_res.sum().item()
            num_masked_S = iso_target_res.shape[0] - num_masked_R
            iso_stats.loc[global_idx, 'num_masked_S'] = num_masked_S
            iso_stats.loc[global_idx, 'num_masked_R'] = num_masked_R
            
            # correct predictions
            iso_pred_res = pred_res[i][iso_ab_mask]
            eq = torch.eq(iso_pred_res, iso_target_res)
            num_R_correct = eq[iso_target_res == 1].sum().item()
            num_S_correct = eq[iso_target_res == 0].sum().item()
            iso_stats.loc[global_idx, 'correct_S'] = num_S_correct
            iso_stats.loc[global_idx, 'correct_R'] = num_R_correct
            iso_stats.loc[global_idx, 'all_correct'] = bool(eq.all().item()) # 1 if all antibiotics are predicted correctly, 0 otherwise                   
            
        return iso_stats
    
     
    def _init_wandb(self):
        self.wandb_run = wandb.init(
            project=self.project_name, # name of the project
            name=self.wandb_name, # name of the run
            
            config={
                "trainer_type": 'fine-tuning',
                'exp_folder': self.exp_folder,
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
            "Losses/val_loss": self.val_losses[-1],
            "Class_metrics/val_sens": self.val_sensitivities[-1],
            "Class_metrics/val_spec": self.val_specificities[-1],
            "Class_metrics/val_F1": self.val_F1_scores[-1],
            "Class_metrics/val_auc_score": self.val_auc_scores[-1],
            "Accuracies/val_acc": self.val_accs[-1],
            "Accuracies/val_iso_acc": self.val_iso_accs[-1],
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
    
    
    def save_model(self, savepath: Path):
        torch.save(self.best_model_state, savepath)
        print(f"Model saved to {savepath}")
        print("="*self._splitter_size)
        
        
    def load_model(self, savepath: Path):
        print("="*self._splitter_size)
        print(f"Loading model from {savepath}")
        self.model.load_state_dict(torch.load(savepath))
        self.model.to(device)
        print("Model loaded")
        print("="*self._splitter_size)