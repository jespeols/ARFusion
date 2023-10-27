# %%
import os
from datetime import datetime
import torch
import yaml
import wandb
import argparse

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
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--name", type=str)
    argparser.add_argument("--num_layers", type=int)
    argparser.add_argument("--num_heads", type=int)
    argparser.add_argument("--emb_dim", type=int)
    # argparser.add_argument("--ff_dim", type=int)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--lr", type=float)
    
    wandb.login() 
    
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")  
        
    print(f"\nCurrent working directory: {BASE_DIR}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # overwrite config with command line arguments
    args = argparser.parse_args()
    config['name'] = args.name if args.name else config['name']
    config['num_layers'] = args.num_layers if args.num_layers else config['num_layers']
    config['num_heads'] = args.num_heads if args.num_heads else config['num_heads']
    config['emb_dim'] = args.emb_dim if args.emb_dim else config['emb_dim']
    # config['ff_dim'] = args.ff_dim if args.ff_dim else config['ff_dim']
    config['hidden_dim'] = config['emb_dim']        
    config['batch_size'] = args.batch_size if args.batch_size else config['batch_size']
    config['epochs'] = args.epochs if args.epochs else config['epochs']
    config['lr'] = args.lr if args.lr else config['lr']
        
    print("Loading dataset...")
    ds_path = BASE_DIR / "data" / "NCBI" / "genotype_parsed.pkl"
    savepath_vocab = 'data/NCBI/geno_vocab.pt'
    ds = GenotypeDataset(ds_path, 
                         savepath_vocab=savepath_vocab, 
                         base_dir=BASE_DIR,
                         train_share=config['train_share'],
                         test_share=config['test_share'])
            
    max_seq_len = ds.max_seq_len
    vocab_size = len(ds.vocab)
    
    print("Loading model...")
    bert = BERT(config, vocab_size).to(device)
    trainer = BertMLMTrainer(
        config=config,
        model=bert,
        dataset=ds,
        results_dir=RESULTS_DIR,
    )
    
    trainer.print_model_summary()
    trainer.print_trainer_summary()
    trainer()
    print("Done!")