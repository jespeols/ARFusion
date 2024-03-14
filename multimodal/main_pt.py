# %%
import torch
import yaml
import wandb
import argparse
import pandas as pd
import os
import sys
from pathlib import Path
from datetime import datetime
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)


# user-defined modules
from multimodal.models import BERT
from multimodal.datasets import MMPretrainDataset
from multimodal.trainers import MMBertPreTrainer

# user-defined functions
from construct_vocab import construct_MM_vocab
from utils import get_multimodal_split_indices, export_results
from data_preprocessing import preprocess_NCBI, preprocess_TESSy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--wandb_mode", type=str)
    argparser.add_argument("--name", type=str)
    argparser.add_argument("--mask_prob_geno", type=float)
    argparser.add_argument("--masking_method", type=str)
    argparser.add_argument("--mask_prob_pheno", type=float)
    argparser.add_argument("--num_known_ab", type=int)
    argparser.add_argument("--num_layers", type=int)
    argparser.add_argument("--num_heads", type=int)
    argparser.add_argument("--emb_dim", type=int)
    argparser.add_argument("--ff_dim", type=int)
    argparser.add_argument("--hidden_dim", type=int)
    argparser.add_argument("--batch_size", type=int)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--lr", type=float)
    argparser.add_argument("--use_weighted_loss", action="store_true", help="Use weighted loss functions")
    argparser.add_argument("--random_state", type=int)
    argparser.add_argument("--prepare_TESSy", action="store_true", help="Prepare TESSy data")
    argparser.add_argument("--prepare_NCBI", action="store_true", help="Prepare NCBI data")
    argparser.add_argument("--no_eval", action="store_true", help="Disable evaluation")
        
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
    
    # overwrite config with command line arguments
    args = argparser.parse_args()
    config['wandb_mode'] = args.wandb_mode if args.wandb_mode else config['wandb_mode']
    config['name'] = args.name if args.name else config['name']
    config['mask_prob_geno'] = args.mask_prob_geno if args.mask_prob_geno else config['mask_prob_geno']
    config['masking_method'] = args.masking_method if args.masking_method else config['masking_method']
    assert config['masking_method'] in ['random', 'num_known', 'keep_one_class'], "Invalid masking method"
    if config['masking_method'] == 'random':
        config['mask_prob_pheno'] = args.mask_prob_pheno if args.mask_prob_pheno else config['mask_prob_pheno']
        config['num_known_ab'] = None
    elif config['masking_method'] == 'num_known':
        config['num_known_ab'] = args.num_known_ab if args.num_known_ab else config['num_known_ab']
        config['mask_prob_pheno'] = None
    elif config['masking_method'] == 'keep_one_class':
        config['num_known_ab'] = None
        config['mask_prob_pheno'] = None
    config['num_layers'] = args.num_layers if args.num_layers else config['num_layers']
    config['num_heads'] = args.num_heads if args.num_heads else config['num_heads']
    config['emb_dim'] = args.emb_dim if args.emb_dim else config['emb_dim']
    config['ff_dim'] = args.ff_dim if args.ff_dim else config['ff_dim']
    config['batch_size'] = args.batch_size if args.batch_size else config['batch_size']
    config['epochs'] = args.epochs if args.epochs else config['epochs']
    config['lr'] = args.lr if args.lr else config['lr']
    config['use_weighted_loss'] = args.use_weighted_loss if args.use_weighted_loss else config['use_weighted_loss']
    config['random_state'] = args.random_state if args.random_state else config['random_state']
    config['data']['TESSy']['prepare_data'] = args.prepare_TESSy if args.prepare_TESSy else config['data']['TESSy']['prepare_data']
    config['data']['NCBI']['prepare_data'] = args.prepare_NCBI if args.prepare_NCBI else config['data']['NCBI']['prepare_data']
    if args.no_eval:
        config['do_eval'] = False
        
    os.environ['WANDB_MODE'] = config['wandb_mode']
    if config['name']:
        results_dir = Path(os.path.join(BASE_DIR / "results" / "MM", config['name']))
    else:
        time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        results_dir = Path(os.path.join(BASE_DIR / "results" / "MM", "experiment_" + str(time_str)))
    print(f"Name of experiment: {config['name']}")
    print(f"Results directory: {results_dir}")
    
    print("\nLoading dataset...")
    data_dict = config['data']
    if data_dict['TESSy']['prepare_data']:
        ds_TESSy = preprocess_TESSy(
            path=data_dict['TESSy']['raw_path'],
            pathogens=data_dict['pathogens'],
            save_path=data_dict['TESSy']['save_path'],
            exclude_antibiotics=data_dict['exclude_antibiotics'],
            impute_age=data_dict['TESSy']['impute_age'],
            impute_gender=data_dict['TESSy']['impute_gender']
        )
    else:
        ds_TESSy = pd.read_pickle(os.path.join(BASE_DIR, data_dict['TESSy']['load_path']))
    ds_pheno = ds_TESSy.copy()
    ds_pheno['country'] = ds_pheno['country'].map(config['data']['TESSy']['country_code_to_name'])
    abbr_to_class_enc = data_dict['antibiotics']['abbr_to_class_enc']
    ds_pheno['ab_classes'] = ds_pheno['phenotypes'].apply(lambda x: [abbr_to_class_enc[p.split('_')[0]] for p in x])
    
    if data_dict['NCBI']['prepare_data']:
        ds_NCBI = preprocess_NCBI(
            path=data_dict['NCBI']['raw_path'],
            save_path=data_dict['NCBI']['save_path'],
            include_phenotype=data_dict['NCBI']['include_phenotype'],
            ab_names_to_abbr=data_dict['antibiotics']['ab_names_to_abbr'],
            exclude_antibiotics=data_dict['exclude_antibiotics'], 
            threshold_year=data_dict['NCBI']['threshold_year'],
            exclude_genotypes=data_dict['NCBI']['exclude_genotypes'],
            exclude_assembly_variants=data_dict['NCBI']['exclude_assembly_variants'],
            exclusion_chars=data_dict['NCBI']['exclusion_chars'],
            gene_count_threshold=data_dict['NCBI']['gene_count_threshold']
        )
    else:
        ds_NCBI = pd.read_pickle(os.path.join(BASE_DIR, data_dict['NCBI']['load_path']))
    
        
    specials = config['specials']
    pad_token = specials['PAD']
    pad_idx = list(specials.values()).index(pad_token) # pass to model for embedding
    ds_geno = ds_NCBI[ds_NCBI['num_ab'] == 0].reset_index(drop=True)
    ds_geno.fillna(pad_token, inplace=True)
        
    ## construct vocabulary
    antibiotics = sorted(list(set(data_dict['antibiotics']['abbr_to_names'].keys()) - set(data_dict['exclude_antibiotics'])))
    vocab = construct_MM_vocab(
        df_geno=ds_NCBI,
        df_pheno=ds_pheno,
        antibiotics=antibiotics,
        specials=specials,
        savepath_vocab=BASE_DIR / config['savepath_vocab']
    )
    vocab_size = len(vocab)
        
    if config['max_seq_len'] == 'auto':
        max_seq_len = int((ds_NCBI['num_genotypes'] + ds_NCBI['num_ab']).max() + 3)
    else:
        max_seq_len = config['max_seq_len']
        
    train_indices, val_indices = get_multimodal_split_indices(
        [ds_geno.shape[0], ds_pheno.shape[0]], 
        val_share=config['val_share'], 
        random_state=config['random_state']
    )
    ds_pt_train = MMPretrainDataset(
        ds_geno=ds_geno.iloc[train_indices[0]],
        ds_pheno=ds_pheno.iloc[train_indices[1]],
        vocab=vocab,
        antibiotics=antibiotics,
        specials=specials,
        max_seq_len=max_seq_len,
        mask_prob_geno=config['mask_prob_geno'],
        masking_method=config['masking_method'],
        mask_prob_pheno=config['mask_prob_pheno'],
        num_known_ab=config['num_known_ab'],
        random_state=config['random_state']
    )
    ds_pt_val = MMPretrainDataset(
        ds_geno=ds_geno.iloc[val_indices[0]],
        ds_pheno=ds_pheno.iloc[val_indices[1]],
        vocab=vocab,
        antibiotics=antibiotics,
        specials=specials,
        max_seq_len=max_seq_len,
        mask_prob_geno=config['mask_prob_geno'],
        masking_method=config['masking_method'],
        mask_prob_pheno=config['mask_prob_pheno'],
        num_known_ab=config['num_known_ab'],
        random_state=config['random_state']
    )
    bert = BERT(config, vocab_size, max_seq_len, len(antibiotics), pad_idx=pad_idx).to(device)
    trainer = MMBertPreTrainer(
        config=config,
        model=bert,
        antibiotics=antibiotics,
        train_set=ds_pt_train,
        val_set=ds_pt_val,
        results_dir=results_dir,
    )
    trainer.print_model_summary()
    trainer.print_trainer_summary()
    results = trainer()    
    print("Finished training!")
    print("Exporting results...") 
    export_results(results, results_dir / "pt_results.pkl")