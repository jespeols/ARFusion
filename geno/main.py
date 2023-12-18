# %%
import torch
import yaml
import argparse
import pandas as pd
import os
import sys
from datetime import datetime
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

from datetime import datetime
from pathlib import Path

# user-defined modules
from geno.models import BERT
from geno.datasets import GenotypeDataset
from geno.trainers import BertMLMTrainer

# user-defined functions
from construct_vocab import construct_geno_vocab
from utils import get_split_indices
from data_preprocessing import preprocess_NCBI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wandb_mode", type=str)
    argparser.add_argument("--name", type=str)
    argparser.add_argument("--mask_prob", type=float)
    argparser.add_argument("--num_layers", type=int)
    argparser.add_argument("--num_heads", type=int)
    argparser.add_argument("--emb_dim", type=int)
    argparser.add_argument("--ff_dim", type=int)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--lr", type=float)
    argparser.add_argument("--random_state", type=int)
        
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")  
        
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Loading config file...")
    config_path = BASE_DIR / "config_geno.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    # overwrite config with command line arguments
    args = argparser.parse_args()
    config['wandb_mode'] = args.wandb_mode if args.wandb_mode else config['wandb_mode']
    config['name'] = args.name if args.name else config['name']
    config['mask_prob'] = args.mask_prob if args.mask_prob else config['mask_prob']
    config['num_layers'] = args.num_layers if args.num_layers else config['num_layers']
    config['num_heads'] = args.num_heads if args.num_heads else config['num_heads']
    config['emb_dim'] = args.emb_dim if args.emb_dim else config['emb_dim']
    config['ff_dim'] = args.ff_dim if args.ff_dim else config['ff_dim']
    config['batch_size'] = args.batch_size if args.batch_size else config['batch_size']
    config['epochs'] = args.epochs if args.epochs else config['epochs']
    config['lr'] = args.lr if args.lr else config['lr']
    config['random_state'] = args.random_state if args.random_state else config['random_state']
        
    os.environ['WANDB_MODE'] = config['wandb_mode']
    if config['name']:
        results_dir = Path(os.path.join(BASE_DIR / "results" / "geno", config['name']))
    else:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(os.path.join(BASE_DIR / "results" / "geno", "experiment_" + str(time_str)))
    print(f"Name of experiment: {config['name']}")
    print(f"Results directory: {results_dir}")
    print("\nLoading dataset...")
    path = BASE_DIR / "data" / "raw" / "NCBI.tsv"

    ds = preprocess_NCBI(path, 
                        include_phenotype=False,
                        threshold_year=config['data']['threshold_year'],
                        exclude_genotypes=config['data']['exclude_genotypes'],
                        exclude_assembly_variants=config['data']['exclude_assembly_variants'],
                        exclusion_chars=config['data']['exclusion_chars'],
                        gene_count_threshold=config['data']['gene_count_threshold'],
                        save_path=config['data']['save_path'])
    num_samples = ds.shape[0]
    
    # replace missing values with PAD token -> will not be included in vocabulary or in self-attention
    specials = config['specials']
    pad_token = specials['PAD']
    ds.fillna(pad_token, inplace=True)
    # replace missing values with [NA] token -> will be included in vocabulary and in self-attention
    # NA = '[NA]' # not available, missing values
    # ds.fillna(NA, inplace=True)
    pad_idx = list(specials.values()).index(pad_token)
    
    print("Constructing vocabulary...")
    savepath_vocab = os.path.join(results_dir, "vocab.pt") if config['save_vocab'] else None
    vocab = construct_geno_vocab(ds, specials, savepath_vocab)
    vocab_size = len(vocab)
    
    if config['max_seq_len'] == 'auto':
        max_genotypes_len = max([ds['num_genotypes'].iloc[i] for i in range(num_samples) if  
                                 ds['year'].iloc[i] != pad_token and ds['country'].iloc[i] != pad_token])
        max_seq_len = max_genotypes_len + 2 + 1 # +2 for year & country, +1 for CLS token
    else:
        max_seq_len = config['max_seq_len']
    
    train_indices, val_indices = get_split_indices(num_samples, config['val_share'], random_state=config['random_state'])
    train_set = GenotypeDataset(ds.iloc[train_indices], vocab, specials, max_seq_len, config['mask_prob'])
    val_set = GenotypeDataset(ds.iloc[val_indices], vocab, specials, max_seq_len, config['mask_prob'])
    
    print("Loading model...")
    bert = BERT(config, vocab_size, max_seq_len, pad_idx=pad_idx).to(device)
    trainer = BertMLMTrainer(
        config=config,
        model=bert,
        train_set=train_set,
        val_set=val_set,
        results_dir=results_dir,
    )
    
    trainer.print_model_summary()
    trainer.print_trainer_summary()
    iso_stats, best_epoch = trainer()
    print("Best epoch: ", best_epoch+1)
    print("Exporting results...")
    iso_stats[best_epoch].to_csv(results_dir / "iso_stats.csv", index=False)
    print("isolate statistics: \n", iso_stats[best_epoch].head(n=10))
    print("Done!")