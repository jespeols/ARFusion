# %%
import os
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter
from itertools import chain
from copy import deepcopy

BASE_DIR = Path(__file__).resolve().parent.parent

def preprocess_NCBI(path,
                    base_dir: Path = BASE_DIR,
                    include_phenotype: bool = False, 
                    save_path = None,
                    threshold_year: int = None,
                    exclude_genotypes: list = None,
                    exclude_assembly_variants: list = None,
                    exclusion_chars: list = None,
                    ):
    os.chdir(base_dir)
    
    NCBI_data = pd.read_csv(path, sep='\t', low_memory=False)
    cols = ['collection_date', 'geo_loc_name', 'AMR_genotypes_core']
    cols += ['AST_phenotypes'] if include_phenotype else []
    df = NCBI_data[cols]
    df = df.rename(columns={'AMR_genotypes_core': 'genotypes'})
    df = df.rename(columns={'AST_phenotypes': 'AST_phenotypes'}) if include_phenotype else df
    df = df[df['genotypes'].notnull()] # filter missing genotypes
    
    #################################### PARSING ####################################
    
    #### geo_loc_name -> country 
    alternative_nan = ['not determined', 'not collected', 'not provided', 'Not Provided',
                   'OUTPATIENT', 'Not collected', 'Not Collected', 'not available']
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].replace(alternative_nan, np.nan) 
    
    # Remove regional information
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].str.split(',').str[0]
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].str.split(':').str[0] 
    df = df.rename(columns={'geo_loc_name': 'country'})
    
    ##### collection_date -> year
    alternative_nan = ['missing']
    df.loc[:,'collection_date'] = df['collection_date'].replace(alternative_nan, np.nan)
    df.loc[:,'collection_date'] = df['collection_date'].str.split('-').str[0]
    df.loc[:,'collection_date'] = df['collection_date'].str.split('/').str[0]
    df = df.rename(columns={'collection_date': 'year'})
    if threshold_year:
        indices = df[df['year'].astype(float) <= threshold_year].index
        df.drop(indices, inplace=True)
    
    #### Parse genotypes
    df['genotypes'] = df['genotypes'].str.split(',')
    print("Parsing genotypes...")
    print(f"Number of samples before parsing: {df.shape[0]}")
    
    df['genotypes'] = df['genotypes'].apply(lambda x: list(set([g.strip() for g in x]))) # remove whitespace and duplicates
    if exclude_genotypes: 
        print(f"Removing genotypes: {exclude_genotypes}")
        df['genotypes'] = df['genotypes'].apply(lambda x: [g for g in x if g not in exclude_genotypes]) 
        
    if exclude_assembly_variants: # Examples: ["=PARTIAL", "=MISTRANSLATION", "=HMM"]
        print(f"Removing genotypes with assembly variants: {exclude_assembly_variants}")
        df['genotypes'] = df['genotypes'].apply(
            lambda x: [g for g in x if not any([variant in g for variant in exclude_assembly_variants])]) 
    df = df[df['genotypes'].apply(lambda x: len(x) > 0)] # Remove any rows where genotypes are empty
    
    if exclusion_chars:
        # old_genotypes = df['genotypes'].copy() # save old genotypes for later
        for char in exclusion_chars:
            print(f"Splitting genotypes by '{char}', removing it and everything after it")
            df['genotypes'] = df['genotypes'].apply(lambda x: list(set([g.split(char)[0] for g in x])))
          
    df['num_genotypes'] = df['genotypes'].apply(lambda x: len(set(x)))
    df['num_point_mutations'] = df['genotypes'].apply(lambda x: len([g for g in x if '=POINT' in g]))

    # Exclude cases where there is only one genotype and no other info
    df = df[~((df['num_genotypes'] == 1) & (df['country'].isnull()) & (df['year'].isnull()))]
    print(f"Number of samples after parsing: {df.shape[0]}")
    
    df.to_pickle(save_path) if save_path else None
    
    return df
            