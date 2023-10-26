import os
import torch
import torch.nn as nn
import time 
import matplotlib.pyplot as plt
import wandb

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path

from datetime import datetime
from model import BERT
from datasets import GenotypeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

############################################ Trainer for MLM task ############################################
    
    
def token_accuracy(tokens: torch.Tensor, token_target: torch.Tensor, token_mask: torch.Tensor, debug: bool = False):
    r = tokens.argmax(-1).masked_select(token_mask) # select the indices of the predicted tokens
    t = token_target.masked_select(token_mask)
    if debug:
        print("\nDebugging token accuracy:")
        print("prediction shape:", tokens.shape)
        print("target shape:", token_target.shape)
        print("mask shape:", token_mask.shape)
        print("argmax(-1) shape:", tokens.argmax(-1).shape)
        print("predicted indices:")
        print(r)
        print("target indices:")
        print(t)
        
    s = (r == t).sum().item()
    print("correct predictions:", s) if debug else None
    tot = (token_mask).sum().item()
    print("total predictions:", tot) if debug else None
    acc = round(s / tot, 2) if tot > 0 else 1 # if tot == 0, then all tokens are masked
    return acc


class BertMLMTrainer(nn.Module):
    
    def __init__(self,
                 model: BERT,
                 dataset: GenotypeDataset,
                 wandb_name: str = None,
                 epochs: int = 5,
                 batch_size: int = 32,
                 train_share: float = 0.8,
                 mask_prob: float = 0.15,
                 report_every: int = 5,
                 print_progress_every: int = 50,
                 early_stopping_patience: int = 3,
                 learning_rate: float = 5e-3,
                 weight_decay: float = 0.01,
                 results_dir: Path = None,
                 ):
        super(BertMLMTrainer, self).__init__()
        
        self.model = model 
        self.dataset = dataset 
        self.dataset_size = len(self.dataset)      
        self.wandb_name = wandb_name if wandb_name else datetime.now().strftime("%Y%m%d-%H%M%S")  
        
        self.model.max_seq_len = self.dataset.max_seq_len # transfer max seq length from dataset to model
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = early_stopping_patience
        self.current_epoch = 0
        
        self.train_share = train_share
        self.train_size = int(self.dataset_size * train_share)
        self.num_batches = self.train_size // self.batch_size
        self.val_share = (1-self.train_share)/2
        self.val_size = int(self.dataset_size * self.val_share)
        self.test_share = self.val_share
        self.test_size = self.val_size
        
        self.mask_prob = mask_prob
        self.criterion = nn.NLLLoss(ignore_index=-100).to(device) # value -100 are ignored in NLLLoss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
                 
        self.losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_accuracies = []
        
        self.report_every = report_every
        self.print_progress_every = print_progress_every
        self._splitter_size = 70
        # self.log_dir = log_dir
        # os.makedirs(self.log_dir) if not os.path.exists(self.log_dir) else None
        # self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.results_dir = results_dir
        os.makedirs(self.results_dir) if not os.path.exists(self.results_dir) else None
        
        
    def print_model_summary(self):        
        print("Model summary:")
        print("="*self._splitter_size)
        print(f"Embedding dim: {self.model.emb_dim}")
        print(f"Number of heads: {self.model.num_heads}")
        print(f"Number of encoder layers: {self.model.num_encoder_layers}")
        print(f"Max sequence length: {self.dataset.max_seq_len}")
        print(f"Vocab size: {len(self.dataset.vocab):,}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("="*self._splitter_size)
        
    
    def print_trainer_summary(self):
        print("="*self._splitter_size)
        print("Trainer summary:")
        print("="*self._splitter_size)
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"Training dataset size: {self.train_size:,}")
        print(f"Mask probability: {self.mask_prob:.0%}")
        print(f"Number of epochs: {self.epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {self.num_batches:,}")
        print(f"Dropout probability: {self.model.dropout_prob:.0%}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")
        print("="*self._splitter_size)
        
        
    def __call__(self):
        
        self._init_wandb()
        
        start_time = time.time()
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        for self.current_epoch in range(self.current_epoch, self.epochs):
            self.model.train()
            self.train_loader, self.val_loader, self.test_loader = self.dataset.get_loaders(
                batch_size=self.batch_size,
                val_share=self.val_share,
                test_share=self.test_share,
                mask_prob=self.mask_prob,
                split=True if self.current_epoch == 0 else False # split only once
            )
            epoch_start_time = time.time()
            loss = self.train(self.current_epoch) # returns loss, averaged over batches
            self.losses.append(loss) 
            print(f"Epoch completed in {(time.time() - epoch_start_time)/60:.1f} min")
            print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
            # print("Evaluating on training set...")
            # _, train_acc = self.evaluate(self.train_loader)
            # self.train_accuracies.append(train_acc)
            print("Evaluating on validation set...")
            val_loss, val_acc = self.evaluate(self.val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self._report_epoch_results()
            # self.writer.add_scalar("Validation loss", val_loss, global_step=self.current_epoch)
            # self.writer.add_scalar("Validation accuracy", val_acc, global_step=self.current_epoch)
            if self.early_stopping():
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.3f}")
                print(f"Best validation loss: {self.best_val_loss:.3f} at epoch {self.best_epoch+1}")
                wandb.log({"best_val_loss": self.best_val_loss, 
                           "val_acc at best epoch":self.val_accuracies[self.best_epoch], 
                           "best_epoch": self.best_epoch+1})
                print("="*self._splitter_size)
                self.model.load_state_dict(self.best_model_state) 
                self.current_epoch = self.best_epoch
                break
        
        self.save_model(self.results_dir / "model_state.pt")
        train_time = (time.time() - start_time)/60
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        
        print("Evaluating on test set...")
        self.test_loss, self.test_acc = self.evaluate(self.test_loader)
        wandb.log({"test_loss": self.test_loss, "test_acc": self.test_acc})
        self._visualize_losses(savepath=self.results_dir / "losses.png")
        self._visualize_accuracy(savepath=self.results_dir / "accuracy.png")
    
        
    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_loss = 0
        reporting_loss = 0
        printing_loss = 0
        for i, batch in enumerate(self.train_loader): # iterate over batches
            batch_index = i + 1
            input, token_target, token_mask, attn_mask = batch
            # print example
            # if batch_index in [10]:
            #     print(f"input: {input[0]}")
            #     print(f"token_target: {token_target[0]}")
            #     print(f"token_mask: {token_mask[0]}")
            #     print(f"attn_mask: {attn_mask[0]}")
            #     print("="*self._splitter_size)
            
            self.optimizer.zero_grad() # zero out gradients
            tokens = self.model(input, attn_mask) # get predictions
            
            tm = token_mask.unsqueeze(-1).expand_as(tokens) # (batch_size, seq_len, vocab_size) 
            tokens = tokens.masked_fill(~tm, 0) # apply mask to tokens along the rows
            loss = self.criterion(tokens.transpose(-1, -2), token_target) # change dim to align with token_target
            
            epoch_loss += loss.item() 
            reporting_loss += loss.item()
            printing_loss += loss.item()
            
            loss.backward() # backpropagate
            self.optimizer.step() # update parameters
            if batch_index % self.report_every == 0:
                self._report_loss_results(batch_index, reporting_loss)
                reporting_loss = 0 
                
            if batch_index % self.print_progress_every == 0:
                time_elapsed = time.gmtime(time.time() - time_ref) # get time elapsed as a struct_time object 
                self._print_loss_summary(time_elapsed, batch_index, printing_loss) 
                printing_loss = 0           
        avg_epoch_loss = epoch_loss / self.num_batches
        return avg_epoch_loss # return loss for saving checkpoint
    
    
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
        
            
    def evaluate(self, loader: DataLoader, print_mode: bool = True):
        self.model.eval()
        loss = 0
        acc = 0
        i = 0
        for batch in loader:
            i+=1
            input, token_target, token_mask, attn_mask = batch
            tokens = self.model(input, attn_mask)
            
            loss += self.criterion(tokens.transpose(-1, -2), token_target).item()
            if i in [] and print_mode:
                acc += token_accuracy(tokens, token_target, token_mask, debug=True)
            else:
                acc += token_accuracy(tokens, token_target, token_mask)
            
        loss /= len(loader) 
        acc /= len(loader)
        
        if print_mode:
            print(f"Loss: {loss:.3f} | Accuracy: {acc:.2%}")
            print("="*self._splitter_size)
        
        return loss, acc
            
     
    def _init_wandb(self):
        wandb.init(
            project="test_project",
            name=self.wandb_name, # name of the run
            
            config={
                "dataset": "NCBI",
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "model": "BERT",
                "hidden_dim": self.model.hidden_dim,
                "num_encoder_layers": self.model.num_encoder_layers,
                "num_heads": self.model.num_heads,
                "embedding_dim": self.model.emb_dim,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "mask_prob": self.mask_prob,
                "max_seq_len": self.dataset.max_seq_len,
                "vocab_size": len(self.dataset.vocab),
                "train_size": self.train_size,
                "val_size": self.val_size,
                "test_size": self.test_size,
                "early_stopping_patience": self.patience,
                "dropout_prob": self.model.dropout_prob,
            }
        )
        wandb.watch(self.model) # watch the model for gradients and parameters
        wandb.define_metric("epoch")
        wandb.define_metric("batch")
        wandb.define_metric("reporting_loss", step_metric="batch")
        wandb.define_metric("train_loss", summary="min", step_metric="epoch")
        wandb.define_metric("val_loss", summary="min", step_metric="epoch")
        wandb.define_metric("val_acc", summary="max", step_metric="epoch")
     
     
    def _report_epoch_results(self):
        wandb_dict = {
            "epoch": self.current_epoch+1,
            "train_loss": self.losses[-1],
            "val_loss": self.val_losses[-1],
            "val_acc": self.val_accuracies[-1]
        }
        wandb.log(wandb_dict)
    
        
    def _report_loss_results(self, batch_index, tot_loss):
        avg_loss = tot_loss / self.report_every
        
        global_step = self.current_epoch * self.num_batches + batch_index # global step, total #batches seen
        wandb.log({"batch": global_step, "reporting_loss": avg_loss})
        # self.writer.add_scalar("Loss", avg_loss, global_step=global_step)
    
        
    def _print_loss_summary(self, time_elapsed, batch_index, tot_loss):
        progress = batch_index / self.num_batches
        mlm_loss = tot_loss / self.print_progress_every
          
        s = f"{time.strftime('%H:%M:%S', time_elapsed)}" 
        s += f" | Epoch: {self.current_epoch+1}/{self.epochs} | {batch_index}/{self.num_batches} ({progress:.2%}) | "\
                f"Loss: {mlm_loss:.3f}"
        print(s)
    
    
    def _visualize_losses(self, savepath: Path = None):
        plt.plot(range(len(self.losses)), self.losses, '-o', label='Training')
        plt.plot(range(len(self.val_losses)), self.val_losses, '-o', label='Validation')
        plt.axhline(y=self.test_loss, color='r', linestyle='--', label='Test')
        plt.title('MLM losses')
        plt.xlabel('Epoch')
        plt.xticks(range(len(self.losses))) if len(self.losses) < 10 else plt.xticks(range(0, len(self.losses), 5))
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(savepath, dpi=300) if savepath else None
        wandb.log({"losses": plt})
        
    
    def _visualize_accuracy(self, savepath: Path = None):
        # plt.plot(range(len(self.train_accuracies)), self.train_accuracies, '-o', label='Training')
        plt.plot(range(len(self.val_accuracies)), self.val_accuracies, '-o', label='Validation')
        plt.axhline(y=self.test_acc, color='r', linestyle='--', label='Test')
        plt.title('MLM accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(savepath, dpi=300) if savepath else None
        wandb.log({"accuracy": plt})
    
    
    def save_model(self, savepath: Path):
        torch.save(self.model.state_dict(), savepath)
        print(f"Model saved to {savepath}")
        print("="*self._splitter_size)
        
        
    def load_model(self, savepath: Path):
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