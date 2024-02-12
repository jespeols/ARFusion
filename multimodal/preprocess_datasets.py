#%%
import yaml
import argparse
import pandas as pd
import os
import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

from data_preprocessing import preprocess_NCBI, preprocess_TESSy

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--preprocess_TESSy", action="store_true", help="Prepare TESSy data")
    argparser.add_argument("--preprocess_NCBI", action="store_true", help="Prepare NCBI data")
    
    print(f"\nCurrent working directory: {os.getcwd()}")
    print("Loading config file...")
    
    config_path = BASE_DIR / "config_MM.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    
    args = argparser.parse_args()
    
    assert args.preprocess_TESSy or args.preprocess_NCBI, "You must instruct the program to prepare at least one of the datasets."
    
    data_dict = config['data']
    if args.preprocess_NCBI:
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
    if args.preprocess_TESSy:
        ds_TESSy = preprocess_TESSy(
            path=data_dict['TESSy']['raw_path'],
            pathogens=data_dict['pathogens'],
            save_path=data_dict['TESSy']['save_path'],
            exclude_antibiotics=data_dict['exclude_antibiotics'],
            impute_age=data_dict['TESSy']['impute_age'],
            impute_gender=data_dict['TESSy']['impute_gender']
        )