# %%
import os
from datetime import datetime
import torch
import yaml
import wandb

from pathlib import Path
from model import BERT
from datasets import GenotypeDataset
from trainers import BertMLMTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

BASE_DIR = Path(__file__).resolve().parent.parent
# LOG_DIR = Path(os.path.join(BASE_DIR / "logs"))
LOG_DIR = Path(os.path.join(BASE_DIR / "logs", "experiment_" + str(time_str)))
RESULTS_DIR = Path(os.path.join(BASE_DIR / "results", "experiment_" + str(time_str)))

if __name__ == "__main__":
    
    wandb.login() 
    
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")
        
    print(f"\nCurrent working directory: {BASE_DIR}")
    print("Loading config file...")
    with open(BASE_DIR / "config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        
    print("Loading dataset...")
    ds_path = BASE_DIR / "data" / "NCBI" / "genotype_parsed.pkl"
    savepath_vocab = 'data/NCBI/geno_vocab.pt'
    ds = GenotypeDataset(ds_path, savepath_vocab=savepath_vocab, subset_share=1, base_dir=BASE_DIR)
            
    max_seq_len = ds.max_seq_len
    vocab_size = len(ds.vocab)
    
    print("Loading model...")
    bert = BERT(config, vocab_size).to(device)
    trainer = BertMLMTrainer(
        model=bert,
        dataset=ds,
        wandb_name=config["wandb_name"],
        results_dir=RESULTS_DIR,
        epochs=config["epochs"],
        early_stopping_patience=config["early_stopping_patience"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        mask_prob=config["mask_prob"],
        report_every=config["report_every"],
        print_progress_every=config["print_progress_every"]
    )
    
    trainer.print_model_summary()
    trainer.print_trainer_summary()
    trainer()
    print("Done!")