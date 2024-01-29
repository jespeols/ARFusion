# %%
import torch
import yaml
import wandb
import argparse
import pandas as pd
import os
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

from datetime import datetime
from pathlib import Path

# user-defined modules
from multimodal.models import BERT
from multimodal.datasets import MMFinetuneDataset
from multimodal.trainers import MMBertFineTuner

# user-defined functions
from utils import get_split_indices, export_results

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wandb_mode", type=str)
    argparser.add_argument("--name", type=str)
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--naive_model", action="store_true", help="Enable naive model")
    argparser.add_argument("--masking_method", type=str)
    argparser.add_argument("--mask_prob_geno", type=float)
    argparser.add_argument("--mask_prob_pheno", type=float)
    argparser.add_argument("--num_known_ab", type=int)
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
    
    config_path = BASE_DIR / "config_MM.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config_ft = config['fine_tuning']
    data_dict = config['data']
    
    # overwrite config with command line arguments
    args = argparser.parse_args()
    config_ft['wandb_mode'] = args.wandb_mode if args.wandb_mode else config_ft['wandb_mode']
    config_ft['name'] = args.name if args.name else config_ft['name']
    config_ft['model_path'] = args.model_path if args.model_path else config_ft['model_path']
    config_ft['naive_model'] = args.naive_model if args.naive_model else config_ft['naive_model']
    config_ft['mask_prob_geno'] = args.mask_prob_geno if args.mask_prob_geno else config_ft['mask_prob_geno']
    config_ft['masking_method'] = args.masking_method if args.masking_method else config_ft['masking_method']
    assert config_ft['masking_method'] in ['random', 'num_known', 'keep_one_class'], "Invalid masking method"
    if config_ft['masking_method'] == 'random':
        config_ft['mask_prob_pheno'] = args.mask_prob_pheno if args.mask_prob_pheno else config_ft['mask_prob_pheno']
        config_ft['num_known_ab'] = None
    elif config_ft['masking_method'] == 'num_known':
        config_ft['num_known_ab'] = args.num_known_ab if args.num_known_ab else config_ft['num_known_ab']
        config_ft['mask_prob_pheno'] = None
    elif config_ft['masking_method'] == 'keep_one_class':
        config_ft['num_known_ab'] = None
        config_ft['mask_prob_pheno'] = None
    config_ft['batch_size'] = args.batch_size if args.batch_size else config_ft['batch_size']
    config_ft['epochs'] = args.epochs if args.epochs else config_ft['epochs']
    config_ft['lr'] = args.lr if args.lr else config['lr']
    config_ft['random_state'] = args.random_state if args.random_state else config_ft['random_state']
        
    os.environ['WANDB_MODE'] = config_ft['wandb_mode']
    if config['name']:
        results_dir = Path(os.path.join(BASE_DIR / "results" / "MM", config_ft['name']))
    else:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(os.path.join(BASE_DIR / "results" / "MM", "experiment_" + str(time_str)))
    print(f"Name of experiment: {config_ft['name']}")
    print(f"Results directory: {results_dir}")
    
    print("\nLoading dataset...")
    ds_NCBI = pd.read_pickle(BASE_DIR / config_ft['ds_path'])
    ds_MM = ds_NCBI[ds_NCBI['num_ab'] > 0].reset_index(drop=True)
    # ds_MM = ds_MM[ds_MM['country'] != 'USA'].reset_index(drop=True) # smaller, non-American dataset
    
    abbr_to_class_enc = config['data']['antibiotics']['abbr_to_class_enc']
    ds_MM['ab_classes'] = ds_MM['phenotypes'].apply(lambda x: [abbr_to_class_enc[p.split('_')[0]] for p in x])
    # if config_ft['masking_method'] == 'keep_one_class':
    #     ds_MM = ds_MM[ds_MM['ab_classes'].apply(lambda x: len(set(x)) > 1)].reset_index(drop=True)
    #     print(f"Removed {ds_NCBI[ds_NCBI['num_ab'] > 0].shape[0] - ds_MM.shape[0]} samples with only one antibiotic class")
    
    print("Loading vocabulary...")
    vocab = torch.load(BASE_DIR / config_ft['loadpath_vocab'])
    vocab_size = len(vocab)
    specials = config['specials']
    pad_token = specials['PAD']
    ds_MM.fillna(pad_token, inplace=True)
    
    antibiotics = sorted(list(set(data_dict['antibiotics']['abbr_to_names'].keys()) - set(data_dict['exclude_antibiotics'])))
    if config['max_seq_len'] == 'auto':
        max_seq_len = int((ds_NCBI['num_genotypes'] + ds_NCBI['num_ab']).max() + 3)
    else:
        max_seq_len = config['max_seq_len']

    train_indices, val_indices = get_split_indices(
        ds_MM.shape[0], 
        val_share=config_ft['val_share'], 
        random_state=config_ft['random_state']
    )  
    ds_ft_train = MMFinetuneDataset(
        df_MM=ds_MM.iloc[train_indices],
        vocab=vocab,
        antibiotics=antibiotics,
        specials=specials,
        max_seq_len=max_seq_len,
        masking_method=config_ft['masking_method'],
        mask_prob_geno=config_ft['mask_prob_geno'],
        mask_prob_pheno=config_ft['mask_prob_pheno'],
        num_known_ab=config_ft['num_known_ab'],
        random_state=config_ft['random_state']
    )
    ds_ft_val = MMFinetuneDataset(
        df_MM=ds_MM.iloc[val_indices],
        vocab=vocab,
        antibiotics=antibiotics,
        specials=specials,
        max_seq_len=max_seq_len,
        masking_method=config_ft['masking_method'],
        mask_prob_geno=config_ft['mask_prob_geno'],
        mask_prob_pheno=config_ft['mask_prob_pheno'],
        num_known_ab=config_ft['num_known_ab'],
        random_state=config_ft['random_state']
    )
    pad_idx = vocab[pad_token]
    bert = BERT(config, vocab_size, max_seq_len, len(antibiotics), pad_idx, pheno_only=True).to(device)
    tuner = MMBertFineTuner(
        config=config,
        model=bert,
        antibiotics=antibiotics,
        train_set=ds_ft_train,
        val_set=ds_ft_val,
        results_dir=results_dir,
    )
    if not config_ft['naive_model']:
        tuner.load_model(Path(BASE_DIR / 'results' / 'MM' / config_ft['model_path']))
        tuner.model.is_pretrained = True
    tuner.print_model_summary()
    tuner.print_trainer_summary()
    ft_results = tuner()
    export_results(ft_results, results_dir / 'results.pkl')