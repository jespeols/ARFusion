# %%
import os
import torch
import yaml
import wandb
import argparse
import pandas as pd
import time

from datetime import datetime
from pathlib import Path

# user-defined modules
from models import BERT
from datasets import PhenotypeMLMDataset, PhenotypeDataset
from trainers import BertMLMTrainer, BertCLSTrainer

# user-defined functions
from construct_vocab import construct_pheno_vocab
from utils import get_split_indices
from data_preprocessing import preprocess_TESSy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wandb_mode", type=str)
    argparser.add_argument("--name", type=str)
    argparser.add_argument("--mask_prob", type=float)
    argparser.add_argument("--num_layers", type=int)
    argparser.add_argument("--num_heads", type=int)
    argparser.add_argument("--emb_dim", type=int)
    argparser.add_argument("--ff_dim", type=int)
    argparser.add_argument("--hidden_dim", type=int)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--lr", type=float)
    argparser.add_argument("--random_state", type=int)
    argparser.add_argument("--classifier_type", type=str)
    argparser.add_argument("--separate_phenotypes", type=bool)
        
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")  
    
    os.chdir(BASE_DIR)
    print(f"\nCurrent working directory: {BASE_DIR}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config_pheno.yaml"
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
    # config['ff_dim'] = config['emb_dim']        
    config['classifier_type'] = args.classifier_type if args.classifier_type else config['classifier_type']
    config['hidden_dim'] = args.hidden_dim if args.hidden_dim else config['hidden_dim']
    config['batch_size'] = args.batch_size if args.batch_size else config['batch_size']
    config['epochs'] = args.epochs if args.epochs else config['epochs']
    config['lr'] = args.lr if args.lr else config['lr']
    config['random_state'] = args.random_state if args.random_state else config['random_state']
    config['do_testing'] = args.do_testing if args.do_testing else config['do_testing']
    config['separate_phenotypes'] = args.separate_phenotypes if args.separate_phenotypes else config['separate_phenotypes']
        
    os.environ['WANDB_MODE'] = config['wandb_mode']
    if config['name']:
        results_dir = Path(os.path.join(BASE_DIR / "results" / "pheno", config['name']))
    else:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(os.path.join(BASE_DIR / "results" / "pheno", "experiment_" + str(time_str)))
    
    if config['data']['prepare_data']:
        print("Preprocessing dataset...")
        start = time.time()
        ds = preprocess_TESSy(path=config['data']['path'],
                              pathogens=config['data']['pathogens'],
                              save_path=config['data']['save_path'],
                              exclude_antibiotics=config['data']['exclude_antibiotics'],
                              impute_age=config['data']['impute_age'],
                              impute_gender=config['data']['impute_gender'])
        print(f"Preprocessing finished after {(time.time()-start)/60:.1f} min")
    else:
        print("Loading dataset...")
        ds = pd.read_pickle(config['data']['load_path'])
    num_samples = ds.shape[0]
    
    ### If gender and age are not imputed, their missing values must be handled, both here and when constructing the vocabulary
    # replace missing values with PAD token -> will not be included in vocabulary or in self-attention
    specials = config['specials']
    # PAD = specials['PAD']
    # ds.fillna(PAD, inplace=True)
    # replace missing values with [NA] token -> will be included in vocabulary and in self-attention
    # NA = '[NA]' # not available, missing values
    # ds.fillna(NA, inplace=True)
    
    print("Constructing vocabulary...")
    savepath_vocab = BASE_DIR / "data" / "pheno_vocab.pt" if config['save_vocab'] else None
    vocab, antibiotics = construct_pheno_vocab(ds, 
                                               specials, 
                                               savepath_vocab=savepath_vocab, 
                                               separate_phenotypes=config['separate_phenotypes'])
    print("Antibiotics:", antibiotics)
    vocab_size = len(vocab)
    
    max_phenotypes_len = ds['num_ab'].max()    
    if config['max_seq_len'] == 'auto':
        if config['separate_phenotypes']:
            max_seq_len = 2*max_phenotypes_len + 4 + 1 # +4 for year, country, age & gender, +1 for CLS token
        else:
            max_seq_len = max_phenotypes_len + 4 + 1
    else:
        max_seq_len = config['max_seq_len']
    
    train_indices, val_indices = get_split_indices(num_samples, config['val_share'], random_state=config['random_state'])
    print("Loading model...")
    if config['classifier_type'] == "MLM":
        assert config['separate_phenotypes'] == True, "Separate phenotypes must be used for MLM classifier"
        
        train_set = PhenotypeMLMDataset(ds.iloc[train_indices], vocab, specials, max_seq_len, base_dir=BASE_DIR)
        val_set = PhenotypeMLMDataset(ds.iloc[val_indices], vocab, specials, max_seq_len, base_dir=BASE_DIR)

        bert = BERT(config, vocab_size, max_seq_len).to(device)
        trainer = BertMLMTrainer(
            config=config,
            model=bert,
            train_set=train_set,
            val_set=val_set,
            results_dir=results_dir,
        )
        trainer.print_model_summary()
        trainer.print_trainer_summary()
        trainer()
    elif config['classifier_type'] == "CLS":
        assert config['separate_phenotypes'] == False, "Separate phenotypes not supported for CLS classifier"
        
        train_set = PhenotypeDataset(ds.iloc[train_indices], vocab, antibiotics, specials, max_seq_len, base_dir=BASE_DIR)
        val_set = PhenotypeDataset(ds.iloc[val_indices], vocab, antibiotics, specials, max_seq_len, base_dir=BASE_DIR)
        bert = BERT(config, vocab_size, max_seq_len, num_ab=len(antibiotics)).to(device)
        trainer = BertCLSTrainer(
            config=config,
            model=bert,
            antibiotics=antibiotics,
            train_set=train_set,
            val_set=val_set,
            results_dir=results_dir,
        )
        trainer.print_model_summary()
        trainer.print_trainer_summary()
        eval_stats_ab, eval_stats_iso, best_epoch = trainer()
        print("Finished training!")
        print("Best epoch:", best_epoch+1)
        ab_stats = eval_stats_ab[best_epoch]
        iso_stats = eval_stats_iso[best_epoch]
        print("Exporting results...")
        ab_stats.to_csv(BASE_DIR / "results" / "pheno" / "ab_stats.csv", index=False)
        iso_stats.to_csv(BASE_DIR / "results" / "pheno" / "iso_stats.csv", index=False)
        print("antibiotics stats:")
        print(ab_stats)
        print("isolate stats:")
        print(iso_stats.head(n=20))
    else:
        raise ValueError("Classifier type not supported, must be 'MLM' or 'CLS'")
        
    print("Done!")