import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time 
import matplotlib.pyplot as plt
import wandb

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

############################################ Trainer for MLM task ############################################

class BertMLMTrainer(nn.Module):
    
    def __init__(self,
                 config: dict,
                 model,
                 train_set,
                 val_set,
                 results_dir: Path = None,
                 ):
        super(BertMLMTrainer, self).__init__()
        
        self.random_state = config["random_state"]
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        
        self.project_name = config["project_name"]
        self.wandb_name = config["name"] if config["name"] else datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model = model
        self.classifier_type = config['classifier_type'] 
        
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
        self.criterion = nn.CrossEntropyLoss(ignore_index = -1).to(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
                 
        self.current_epoch = 0
        self.report_every = config["report_every"] if config["report_every"] else 500
        self.print_progress_every = config["print_progress_every"] if config["print_progress_every"] else 1000
        self._splitter_size = 70
        self.results_dir = results_dir
        if self.results_dir:
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        
    def print_model_summary(self):        
        print("Model summary:")
        print("="*self._splitter_size)
        print(f"Embedding dim: {self.model.emb_dim}")
        print(f"Classifier type: {self.classifier_type}")
        print(f"Feed-forward dim: {self.model.ff_dim}")
        print(f"Dropout probability: {self.model.dropout_prob:.0%}")
        print(f"Number of heads: {self.model.num_heads}")
        print(f"Number of encoder layers: {self.model.num_layers}")
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
        print(f"CV split: {self.train_share:.0%} train | {self.val_share:.0%} val")
        print(f"Mask probability: {self.mask_prob:.0%}")
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
            print(f"Epoch completed in {(time.time() - epoch_start_time)/60:.1f} min")
            print("Evaluating on validation set...")
            val_results = self.evaluate(self.val_loader, self.val_set)
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            self._update_val_lists(val_results)
            self._report_epoch_results()
            early_stop = self.early_stopping()
            if early_stop:
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.4f}")
                s = f"Best validation loss {self.best_val_loss:.4f}"
                s += f" | Validation accuracy {self.val_accuracies[self.best_epoch]:.2%}"
                s += f" | Validation sequence accuracy {self.val_iso_accuracies[self.best_epoch]:.2%}"
                s += f" at epoch {self.best_epoch+1}"
                print(s)
                self.wandb_run.log({"Losses/final_val_loss": self.best_val_loss, 
                           "Accuracies/final_val_acc":self.val_accuracies[self.best_epoch],
                           "Accuracies/final_val_iso_acc": self.val_iso_accuracies[self.best_epoch],
                           "final_epoch": self.best_epoch+1})
                print("="*self._splitter_size)
                self.model.load_state_dict(self.best_model_state) 
                self.current_epoch = self.best_epoch
                break
            if self.scheduler:
                self.scheduler.step() 
        
        if not early_stop:    
            self.wandb_run.log({"Losses/final_val_loss": self.val_losses[-1], 
                    "Accuracies/final_val_acc":self.val_accuracies[-1],
                    "Accuracies/final_val_iso_acc": self.val_iso_accuracies[-1],
                    "final_epoch": self.current_epoch+1})
        if self.save_model:
            self.save_model(self.results_dir / "model_state.pt") 
        train_time = (time.time() - start_time)/60
        self.wandb_run.log({"Training time (min)": train_time})
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        if not early_stop:
            s = f"Final validation loss {self.val_losses[-1]:.4f}"
            s += f" | Final validation accuracy {self.val_accuracies[-1]:.2%}"
            s += f" | Final validation sequence accuracy {self.val_iso_accuracies[-1]:.2%}"
            print(s)
        return self.val_iso_stats, self.current_epoch
    
        
    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_loss = 0
        reporting_loss = 0
        printing_loss = 0
        for i, batch in enumerate(self.train_loader):
            batch_index = i + 1
            input, token_target, attn_mask = batch
            
            self.optimizer.zero_grad() # zero out gradients
            tokens = self.model(input, attn_mask) # get predictions
            
            loss = self.criterion(tokens.transpose(-1, -2), token_target) # change dim to align with token_target
            
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
        
         
    def evaluate(self, loader: DataLoader, ds_obj, print_mode: bool = True):
        self.model.eval()
        eval_stats_iso = ds_obj.ds.copy()
        eval_stats_iso.replace({'[PAD]': np.nan}, inplace=True)
        eval_stats_iso['num_masked'], eval_stats_iso['num_correct'] = 0, 0
        eval_stats_iso['correct_all'] = False
        assert self.val_size == eval_stats_iso.shape[0], "Validation set size does not match loaded dataset object"
        with torch.no_grad():
            loss = 0
            for batch_idx, batch in enumerate(loader):
                input, token_target, token_mask, attn_mask = batch
                tokens = self.model(input, attn_mask)
                
                loss += self.criterion(tokens.transpose(-1, -2), token_target).item()
                eval_stats_iso = self._update_iso_stats(batch_idx, tokens, token_target, token_mask, eval_stats_iso)
            
            loss /= len(loader) 
            acc = eval_stats_iso['num_correct'].sum() / eval_stats_iso['num_masked'].sum()
            iso_acc = eval_stats_iso['correct_all'].sum() / eval_stats_iso.shape[0]
        
        if print_mode:
            print(f"Loss: {loss:.4f} | Accuracy: {acc:.2%} | Isolate accuracy: {iso_acc:.2%}")
            print("="*self._splitter_size)
        
        results = {
            "loss": loss,
            "acc": acc,
            "iso_acc": iso_acc,
            "iso_stats": eval_stats_iso
        }
        return results
     
     
    def _init_result_lists(self):
        self.losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_iso_accuracies = []
        self.val_iso_stats = []
    
    
    def _update_val_lists(self, val_results: dict):
        self.val_losses.append(val_results["loss"])
        self.val_accuracies.append(val_results["acc"])
        self.val_iso_accuracies.append(val_results["iso_acc"])
        self.val_iso_stats.append(val_results["iso_stats"])


    def _update_iso_stats(self, batch_idx:int, tokens: torch.Tensor, token_target: torch.Tensor, token_mask: torch.Tensor,
                          eval_stats_iso: pd.DataFrame):
        
        for i in range(token_target.shape[0]):
            global_idx = batch_idx * self.batch_size + i
            # get the predicted and target tokens for the sequence
            iso_mask = token_mask[i] # (seq_len,)
            pred_tokens = tokens[i, iso_mask].argmax(-1) # (num_masked_tokens,)
            targets = token_target[i, iso_mask] # (num_masked_tokens,)
            
            eq = torch.eq(pred_tokens, targets) 
            eval_stats_iso.loc[global_idx, 'num_masked'] = iso_mask.sum().item()
            eval_stats_iso.loc[global_idx, 'num_correct'] = eq.sum().item()
            eval_stats_iso.loc[global_idx, 'correct_all'] = eq.all().item()
        
        return eval_stats_iso
        
     
    def _init_wandb(self):
        self.wandb_run = wandb.init(
            project=self.project_name, # name of the project
            name=self.wandb_name, # name of the run
            
            config={
                # "dataset": "NCBI",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                # "model": "BERT",
                "hidden_dim": self.model.hidden_dim,
                "classifier_type": self.classifier_type,
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "emb_dim": self.model.emb_dim,
                'ff_dim': self.model.ff_dim,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "mask_prob": self.mask_prob,
                "max_seq_len": self.model.max_seq_len,
                "vocab_size": len(self.train_set.vocab),
                "num_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "train_size": self.train_size,
                "random_state": self.random_state,
                # "val_size": self.val_size,
                # "early_stopping_patience": self.patience,
                # "dropout_prob": self.model.dropout_prob,
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
            "Accuracies/val_acc": self.val_accuracies[-1],
            "Accuracies/val_iso_acc": self.val_iso_accuracies[-1]
        }
        self.wandb_run.log(wandb_dict)
    
        
    def _report_loss_results(self, batch_index, tot_loss):
        avg_loss = tot_loss / self.report_every
        
        global_step = self.current_epoch * self.num_batches + batch_index # global step, total #batches seen
        self.wandb_run.log({"batch": global_step, "Losses/live_loss": avg_loss})
        # self.writer.add_scalar("Loss", avg_loss, global_step=global_step)
    
        
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
        print("Model loaded")
        print("="*self._splitter_size)
    
    ######################### Function to load and save training checkpoints ###########################
    
    # def save_checkpoint(self, epoch, loss):
    #     if not self.checkpoint_dir: # if checkpoint_dir is None
    #         return
        
    #     name = f"bert_epoch{epoch+1}_loss{loss:.2f}.pt"

    #     if not self.checkpoint_dir.exists(): 
    #         self.checkpoint_dir.mkdir()
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'loss': loss
    #         }, self.checkpoint_dir / name)
        
    #     print(f"Checkpoint saved to {self.checkpoint_dir / name}")
    #     print("="*self._splitter_size)
        
        
    # def load_checkpoint(self, path: Path):
    #     print("="*self._splitter_size)
    #     print(f"Loading model checkpoint from {path}")
    #     checkpoint = torch.load(path)
    #     self.current_epoch = checkpoint['epoch']
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     print(f"Loaded checkpoint from epoch {self.current_epoch}")
    #     print("="*self._splitter_size)
