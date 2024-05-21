# %%
import torch
import yaml
import wandb
import argparse
import pandas as pd
import time
import os
import sys
from datetime import datetime
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

# user-defined modules
from pheno.models import BERT
from pheno.datasets import PhenotypeDataset
from pheno.trainers import BertTrainer

# user-defined functions
from construct_vocab import construct_pheno_vocab
from utils import get_split_indices, export_results
from data_preprocessing import preprocess_TESSy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wandb_mode", type=str)
    argparser.add_argument("--name", type=str)
    argparser.add_argument("--mask_prob", type=float)
    argparser.add_argument("--num_known_ab", type=int)
    argparser.add_argument("--num_known_classes", type=int)
    argparser.add_argument("--num_layers", type=int)
    argparser.add_argument("--num_heads", type=int)
    argparser.add_argument("--emb_dim", type=int)
    argparser.add_argument("--ff_dim", type=int)
    argparser.add_argument("--hidden_dim", type=int)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--lr", type=float)
    argparser.add_argument("--val_share", type=float)
    argparser.add_argument("--wl_strength", type=str, help="Strength of weighted CE loss functions for antibiotics ('mild' or 'strong')")
    argparser.add_argument("--random_state", type=int)
    argparser.add_argument("--prepare_data", type=bool)
    argparser.add_argument("--save_model", type=bool)
        
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
    if args.num_known_ab:
        config['num_known_ab'] = args.num_known_ab
        config['mask_prob'] = None
    if args.mask_prob:
        config['mask_prob'] = args.mask_prob
        config['num_known_ab'] = None
    config['mask_prob'] = args.mask_prob if args.mask_prob else config['mask_prob']
    config['num_heads'] = args.num_heads if args.num_heads else config['num_heads']
    config['emb_dim'] = args.emb_dim if args.emb_dim else config['emb_dim']
    config['ff_dim'] = args.ff_dim if args.ff_dim else config['ff_dim']
    config['hidden_dim'] = args.hidden_dim if args.hidden_dim else config['hidden_dim']
    config['batch_size'] = args.batch_size if args.batch_size else config['batch_size']
    config['epochs'] = args.epochs if args.epochs else config['epochs']
    config['lr'] = args.lr if args.lr else config['lr']
    assert sum([
        (args.mask_prob is not None), (args.num_known_ab is not None), (args.num_known_classes is not None)
    ]) <= 1, "Choose only one masking method."
    if args.mask_prob:
        config['masking_method'] = 'random'
        config['mask_prob'] = args.mask_prob
        config['num_known_ab'], config['num_known_classes'] = None, None
    elif args.num_known_ab:
        config['masking_method'] = 'num_known_ab'
        config['num_known_ab'] = args.num_known_ab
        config['mask_prob'], config['num_known_classes'] = None, None
    elif args.num_known_classes:
        config['masking_method'] = 'num_known_classes'
        config['num_known_classes'] = args.num_known_classes
        config['mask_prob'], config['num_known_ab'] = None, None
    if args.wl_strength:
        assert args.wl_strength in ['mild', 'strong'], "Invalid weighted loss strength, choose from ['mild', 'strong']"
        config['wl_strength'] = args.wl_strength
    config['random_state'] = args.random_state if args.random_state else config['random_state']
    config['data']['prepare_data'] = args.prepare_data if args.prepare_data else config['data']['prepare_data']
        
    os.environ['WANDB_MODE'] = config['wandb_mode']
    if config['name']:
        results_dir = Path(os.path.join(BASE_DIR / "results" / "pheno", config['name']))
    else:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(os.path.join(BASE_DIR / "results" / "pheno", "experiment_" + str(time_str)))
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Name of experiment: {config['name']}")
    print(f"Results directory: {results_dir}")
    
    if config['data']['prepare_data']:
        print("\nPreprocessing dataset...")
        start = time.time()
        ds = preprocess_TESSy(path=config['data']['raw_path'],
                              pathogens=config['data']['pathogens'],
                              save_path=config['data']['save_path'],
                              exclude_antibiotics=config['data']['exclude_antibiotics'],
                              impute_age=config['data']['impute_age'],
                              impute_gender=config['data']['impute_gender'])
        print(f"Preprocessing finished after {(time.time()-start)/60:.1f} min")
    else:
        print("\nLoading dataset...")
        ds = pd.read_pickle(config['data']['load_path'])
    # ds = ds.sample(frac=0.1, random_state=config['random_state']).reset_index(drop=True)
    num_samples = ds.shape[0]
    
    abbr_to_class_enc = config['data']['antibiotics']['abbr_to_class_enc']
    ds['ab_classes'] = ds['phenotypes'].apply(lambda x: [abbr_to_class_enc[p.split('_')[0]] for p in x])
    
    print("Constructing vocabulary...")
    specials = config['specials']
    pad_token = specials['PAD']
    pad_idx = list(specials.values()).index(pad_token)
    antibiotics = sorted(list(set(config['data']['antibiotics']['abbr_to_name'].keys()) - set(config['data']['exclude_antibiotics'])))
    savepath_vocab = os.path.join(BASE_DIR, "pheno_vocab.pt") if config['save_vocab'] else None
    vocab = construct_pheno_vocab(ds, specials, antibiotics, savepath_vocab=savepath_vocab)
    vocab_size = len(vocab)
    
    max_phenotypes_len = ds['num_ab'].max()    
    if config['max_seq_len'] == 'auto':
        max_seq_len = max_phenotypes_len + 4 + 1
    else:
        max_seq_len = config['max_seq_len']
    
    train_indices, val_indices = get_split_indices(num_samples, config['val_share'], random_state=config['random_state'])
    train_set = PhenotypeDataset(
        ds.iloc[train_indices], 
        vocab,
        antibiotics,
        specials,
        max_seq_len,
        masking_method=config['masking_method'],
        num_known_ab=config['num_known_ab'],
        mask_prob=config['mask_prob'],
        num_known_classes=config['num_known_classes'],
        always_mask_replace=True
    )
    val_set = PhenotypeDataset(
        ds.iloc[val_indices],
        vocab, antibiotics,
        specials, 
        max_seq_len,
        masking_method=config['masking_method'],
        num_known_ab=config['num_known_ab'],
        mask_prob=config['mask_prob'],
        num_known_classes=config['num_known_classes'],
        always_mask_replace=True
    )
    print("Loading model...")
    bert = BERT(config, vocab_size, max_seq_len, num_ab=len(antibiotics), pad_idx=pad_idx).to(device)
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
    results = trainer()
    print("Finished training!")
    print("Exporting results...")
    export_results(results, results_dir / "results.pkl")
    print("Done!")