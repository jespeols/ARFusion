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
    
    
def token_accuracy(tokens: torch.Tensor, token_target: torch.Tensor, token_mask: torch.Tensor):
    r = tokens.argmax(-1).masked_select(token_mask) # select the indices of the predicted tokens
    t = token_target.masked_select(token_mask)
        
    s = (r == t).sum().item()
    tot = (token_mask).sum().item()
    acc = round(s / tot, 2) if tot > 0 else 1 # if tot == 0, then all tokens are masked (currently not possible)
    return acc


class BertMLMTrainer(nn.Module):
    
    def __init__(self,
                 config: dict,
                 model: BERT,
                 train_set: GenotypeDataset,
                 val_set: GenotypeDataset,
                 test_set: GenotypeDataset,
                 results_dir: Path = None,
                 ):
        super(BertMLMTrainer, self).__init__()
        
        self.random_state = config["random_state"]
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)
        
        self.model = model 
        self.train_set, self.train_size = train_set, len(train_set)
        self.train_size = len(self.train_set)      
        self.model.max_seq_len = self.train_set.max_seq_len 
        self.val_set, self.val_size = val_set, len(val_set)
        self.test_set, self.test_size = test_set, len(test_set)
        self.split = config["split"]
        self.project_name = config["project_name"]
        self.wandb_name = config["name"] if config["name"] else datetime.now().strftime("%Y%m%d-%H%M%S")
         
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.epochs = config["epochs"]
        self.patience = config["early_stopping_patience"]
        self.save_model = config["save_model"] if config["save_model"] else False
        
        self.do_testing = config["do_testing"] if config["do_testing"] else False
        self.num_batches = self.train_size // self.batch_size
        
        self.mask_prob = config["mask_prob"] if config["mask_prob"] else 0.15
        self.criterion = nn.NLLLoss(ignore_index=-100).to(device) # value -100 are ignored in NLLLoss
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = None
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.9)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98)
                 
        self.current_epoch = 0
        self.losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_accuracies = []
        
        self.report_every = config["report_every"] if config["report_every"] else 100
        self.print_progress_every = config["print_progress_every"] if config["print_progress_every"] else 1000
        self._splitter_size = 70
        self.results_dir = results_dir
        os.makedirs(self.results_dir) if not os.path.exists(self.results_dir) else None
        
        
    def print_model_summary(self):        
        print("Model summary:")
        print("="*self._splitter_size)
        print(f"Embedding dim: {self.model.emb_dim}")
        print(f"Hidden dim: {self.model.hidden_dim}")
        print(f"Number of heads: {self.model.num_heads}")
        print(f"Number of encoder layers: {self.model.num_layers}")
        print(f"Max sequence length: {self.model.max_seq_len}")
        print(f"Vocab size: {len(self.train_set.vocab):,}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print("="*self._splitter_size)
        
    
    def print_trainer_summary(self):
        print("Trainer summary:")
        print("="*self._splitter_size)
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"Training dataset size: {self.train_size:,}")
        print(f"Train-val-test split {self.split[0]:.0%} - {self.split[1]:.0%} - {self.split[2]:.0%}")
        print(f"Will test? {'Yes' if self.do_testing else 'No'}")
        print(f"Mask probability: {self.mask_prob:.0%}")
        print(f"Number of epochs: {self.epochs}")
        print(f"Early stopping patience: {self.patience}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of batches: {self.num_batches:,}")
        print(f"Dropout probability: {self.model.dropout_prob:.0%}")
        print(f"Learning rate: {self.lr}")
        print(f"Weight decay: {self.weight_decay}")
        print("="*self._splitter_size)
        
        
    def __call__(self):      
        self.wandb_run = self._init_wandb()
        
        self.val_set.prepare_dataset(mask_prob=self.mask_prob) 
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)
        if self.do_testing:
            self.test_set.prepare_dataset(mask_prob=self.mask_prob) 
            self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        
        start_time = time.time()
        self.best_val_loss = float('inf')
        
        for self.current_epoch in range(self.current_epoch, self.epochs):
            self.model.train()
            # Dynamic masking: New mask for training set each epoch
            self.train_set.prepare_dataset(mask_prob=self.mask_prob)
            # self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, sampler=RandomSampler(self.train_set))
            self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
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
            early_stop = self.early_stopping()
            if early_stop:
                print(f"Early stopping at epoch {self.current_epoch+1} with validation loss {self.val_losses[-1]:.3f}")
                print(f"Best validation loss: {self.best_val_loss:.3f} | validation accuracy \
                      {self.val_accuracies[self.best_epoch]} at epoch {self.best_epoch+1}")
                self.wandb_run.log({"Losses/final_val_loss": self.best_val_loss, 
                           "Accuracies/final_val_acc":self.val_accuracies[self.best_epoch], 
                           "final_epoch": self.best_epoch+1})
                print("="*self._splitter_size)
                self.model.load_state_dict(self.best_model_state) 
                self.current_epoch = self.best_epoch
                break
            self.scheduler.step() if self.scheduler else None
        
        if not early_stop:    
            self.wandb_run.log({"Losses/final_val_loss": self.val_losses[-1], 
                    "Accuracies/final_val_acc":self.val_accuracies[-1], 
                    "final_epoch": self.current_epoch+1})
        self.save_model(self.results_dir / "model_state.pt") if self.save_model else None
        train_time = (time.time() - start_time)/60
        self.wandb_run.log({"Training time (min)": train_time})
        disp_time = f"{train_time//60:.0f}h {train_time % 60:.1f} min" if train_time > 60 else f"{train_time:.1f} min"
        print(f"Training completed in {disp_time}")
        if not early_stop:
            print(f"Final validation loss: {self.val_losses[-1]:.3f} | Final validation accuracy: {self.val_accuracies[-1]:.2%}")
        
        if self.do_testing:
            print("Evaluating on test set...")
            self.test_loss, self.test_acc = self.evaluate(self.test_loader)
            self.wandb_run.log({"Losses/test_loss": self.test_loss, "Accuracies/test_acc": self.test_acc})
        self._visualize_losses(savepath=self.results_dir / "losses.png")
        self._visualize_accuracy(savepath=self.results_dir / "accuracy.png")
    
        
    def train(self, epoch: int):
        print(f"Epoch {epoch+1}/{self.epochs}")
        time_ref = time.time()
        
        epoch_loss = 0
        reporting_loss = 0
        printing_loss = 0
        for i, batch in enumerate(self.train_loader):
            batch_index = i + 1
            input, token_target, token_mask, attn_mask = batch
            
            self.optimizer.zero_grad() # zero out gradients
            tokens = self.model(input, attn_mask) # get predictions
            
            tm = token_mask.unsqueeze(-1).expand_as(tokens) # (batch_size, seq_len, vocab_size) 
            tokens = tokens.masked_fill(~tm, 0) # apply mask to tokens along the rows
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
        
            
    def evaluate(self, loader: DataLoader, print_mode: bool = True):
        self.model.eval()
        loss = 0
        acc = 0
        for batch in loader:
            input, token_target, token_mask, attn_mask = batch
            tokens = self.model(input, attn_mask)
            
            loss += self.criterion(tokens.transpose(-1, -2), token_target).item()
            acc += token_accuracy(tokens, token_target, token_mask)
            
        loss /= len(loader) 
        acc /= len(loader)
        
        if print_mode:
            print(f"Loss: {loss:.3f} | Accuracy: {acc:.2%}")
            print("="*self._splitter_size)
        
        return loss, acc
            
     
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
                "num_layers": self.model.num_layers,
                "num_heads": self.model.num_heads,
                "emb_dim": self.model.emb_dim,
                "lr": self.lr,
                "weight_decay": self.weight_decay,
                "mask_prob": self.mask_prob,
                "max_seq_len": self.model.max_seq_len,
                "vocab_size": len(self.train_set.vocab),
                "train_size": self.train_size,
                "random_state": self.random_state,
                # "val_size": self.val_size,
                # "test_size": self.test_size,
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
        
        self.wandb_run.define_metric("Losses/test_loss")
        self.wandb_run.define_metric("Accuracies/test_acc")
        self.wandb_run.define_metric("Losses/final_val_loss")
        self.wandb_run.define_metric("Accuracies/final_val_acc")
        self.wandb_run.define_metric("final_epoch")

        return self.wandb_run
     
    def _report_epoch_results(self):
        wandb_dict = {
            "epoch": self.current_epoch+1,
            "Losses/train_loss": self.losses[-1],
            "Losses/val_loss": self.val_losses[-1],
            "Accuracies/val_acc": self.val_accuracies[-1]
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
                f"Loss: {mlm_loss:.3f}"
        print(s)
    
    
    def _visualize_losses(self, savepath: Path = None):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.losses)), self.losses, '-o', label='Training')
        ax.plot(range(len(self.val_losses)), self.val_losses, '-o', label='Validation')
        ax.axhline(y=self.test_loss, color='r', linestyle='--', label='Test') if self.do_testing else None
        ax.set_title('MLM losses')
        ax.set_xlabel('Epoch')
        ax.set_xticks(range(len(self.losses))) if len(self.losses) < 10 else ax.set_xticks(range(0, len(self.losses), 5))
        ax.set_ylabel('Loss')
        ax.legend()
        plt.savefig(savepath, dpi=300) if savepath else None
        # self.wandb_run.log({"Losses/losses": wandb.log(ax)})
        self.wandb_run.log({"Losses/losses": wandb.Image(ax)})
        plt.close()
        
    
    def _visualize_accuracy(self, savepath: Path = None):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.val_accuracies)), self.val_accuracies, '-o', label='Validation')
        ax.axhline(y=self.test_acc, color='r', linestyle='--', label='Test') if self.do_testing else None
        ax.set_title('MLM accuracy')
        ax.set_xlabel('Epoch')
        ax.set_xticks(range(len(self.val_accuracies))) if len(self.val_accuracies) < 10 else ax.set_xticks(range(0, len(self.val_accuracies), 5))
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.savefig(savepath, dpi=300) if savepath else None
        # self.wandb_run.log({"Accuracies/accuracy": wandb.log(ax)})
        self.wandb_run.log({"Accuracies/accuracy": wandb.Image(ax)})
        plt.close() 
    
    
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