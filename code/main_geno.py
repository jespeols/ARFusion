# %%
import os
import torch
import yaml
import wandb
import argparse
import pandas as pd

from datetime import datetime
from pathlib import Path

# user-defined modules
from model import BERT
from datasets import GenotypeDataset
from trainers import BertMLMTrainer

# user-defined functions
from construct_vocab import construct_geno_vocab
from utils import get_split_indices
from data_preprocessing import preprocess_NCBI

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
    argparser.add_argument("--mask_prob", type=int)
    argparser.add_argument("--num_layers", type=int)
    argparser.add_argument("--num_heads", type=int)
    argparser.add_argument("--emb_dim", type=int)
    # argparser.add_argument("--hidden_dim", type=int)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--lr", type=float)
    argparser.add_argument("--random_state", type=int)
    
    # os.environ['WANDB_MODE'] = 'disabled' # 'dryrun' or 'run' or 'offline' or 'disabled' or 'online'
    
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")  
        
    print(f"\nCurrent working directory: {BASE_DIR}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config_geno.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # overwrite config with command line arguments
    args = argparser.parse_args()
    config['name'] = args.name if args.name else config['name']
    config['mask_prob'] = args.mask_prob if args.mask_prob else config['mask_prob']
    config['num_layers'] = args.num_layers if args.num_layers else config['num_layers']
    config['num_heads'] = args.num_heads if args.num_heads else config['num_heads']
    config['emb_dim'] = args.emb_dim if args.emb_dim else config['emb_dim']
    # config['hidden_dim'] = args.hidden_dim if args.hidden_dim else config['hidden_dim']
    config['hidden_dim'] = config['emb_dim']        
    config['batch_size'] = args.batch_size if args.batch_size else config['batch_size']
    config['epochs'] = args.epochs if args.epochs else config['epochs']
    config['lr'] = args.lr if args.lr else config['lr']
    config['random_state'] = args.random_state if args.random_state else config['random_state']
        
    print("Loading dataset...")
    path = BASE_DIR / "data" / "raw" / "NCBI.tsv"

    ds = preprocess_NCBI(path, 
                        include_phenotype=False, 
                        threshold_year=config['data']['threshold_year'],
                        exclude_genotypes=config['data']['exclude_genotypes'],
                        exclude_assembly_variants=config['data']['exclude_assembly_variants'],
                        exclusion_chars=config['data']['exclusion_chars'],
                        gene_count_threshold=None,
                        save_path=None)
    num_samples = ds.shape[0]
    
    # replace missing values with PAD token -> will not be included in vocabulary or in self-attention
    specials = config['specials']
    PAD = specials['PAD']
    ds.fillna(PAD, inplace=True)
    # replace missing values with [NA] token -> will be included in vocabulary and in self-attention
    # NA = '[NA]' # not available, missing values
    # ds.fillna(NA, inplace=True)
    
    print("Constructing vocabulary...")
    savepath_vocab = BASE_DIR / "data" / "NCBI" / "geno_vocab.pt" if config['save_vocab'] else None
    vocab = construct_geno_vocab(ds, specials, savepath_vocab)
    vocab_size = len(vocab)
    
    if config['max_seq_len'] == 'auto':
        max_genotypes_len = max([ds['num_genotypes'].iloc[i] for i in range(ds.shape[0]) if  
                                 ds['year'].iloc[i] != PAD and ds['country'].iloc[i] != PAD])
        max_seq_len = max_genotypes_len + 2 + 1 # +2 for year & country, +1 for CLS token
    else:
        max_seq_len = config['max_seq_len']
    
    # split dataset into train, test, val
    train_indices, val_indices, test_indices = get_split_indices(num_samples, config['split'], 
                                                                 random_state=config['random_state'])

    train_set = GenotypeDataset(ds.iloc[train_indices], vocab, specials, max_seq_len, base_dir=BASE_DIR)
    val_set = GenotypeDataset(ds.iloc[val_indices], vocab, specials, max_seq_len, base_dir=BASE_DIR)
    test_set = GenotypeDataset(ds.iloc[test_indices], vocab, specials, max_seq_len, base_dir=BASE_DIR)
    
    print("Loading model...")
    bert = BERT(config, vocab_size).to(device)
    trainer = BertMLMTrainer(
        config=config,
        model=bert,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        results_dir=RESULTS_DIR,
    )
    
    trainer.print_model_summary()
    trainer.print_trainer_summary()
    trainer()
    print("Done!")