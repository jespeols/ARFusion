# %%
import os
import torch
import pandas as pd

from pathlib import Path
from torchtext.vocab import vocab as Vocab
from itertools import chain
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent

def construct_geno_vocab(dataset: pd.DataFrame, specials:dict, savepath_vocab: Path = None):
    token_counter = Counter()
    ds = dataset.copy()
    
    PAD, UNK = specials['PAD'], specials['UNK']
    special_tokens = specials.values() 
    
    year = ds[ds['year'] != PAD]['year'].astype('Int16')
    year_range = range(year.min(), year.max()+1)
    token_counter.update([str(y) for y in year_range]) # make sure all years are included in the vocab
    
    # update counter
    token_counter.update(ds[ds['year'] != PAD]['year'].tolist()) # count tokens that are not [PAD] (missing values)
    token_counter.update(ds[ds['country'] != PAD]['country'].tolist())
    token_counter.update(list(chain(*ds['genotypes'])))
        
    vocab = Vocab(token_counter, specials=special_tokens)
    vocab.set_default_index(vocab[UNK])
    if savepath_vocab:
        torch.save(vocab, savepath_vocab)
    return vocab


def construct_pheno_vocab(dataset: pd.DataFrame, specials:dict, antbiotics:list, savepath_vocab: Path = None):
    token_counter = Counter()
    ds = dataset.copy() 
    
    UNK = specials['UNK']
    special_tokens = specials.values() 
    
    min_year, max_year = ds['year'].min(), ds['year'].max()
    year_range = range(min_year, max_year + 1)
    token_counter.update([str(y) for y in year_range]) 
    
    min_age, max_age = ds['age'].min(), ds['age'].max()
    age_range = range(int(min_age), int(max_age + 1))
    token_counter.update([str(a) for a in age_range])
    
    countries = ds['country'].unique().astype('str').tolist()
    token_counter.update(countries)
    
    gender = ds['gender'].unique().astype('str').tolist()
    token_counter.update(gender)
    
    token_counter.update([ab + '_' + res for ab in antbiotics for res in ['S', 'R']])
    
    vocab = Vocab(token_counter, specials=special_tokens)
    vocab.set_default_index(vocab[UNK])
    torch.save(vocab, savepath_vocab) if savepath_vocab else None
    return vocab
    
    
def construct_MM_vocab(
        df_geno: pd.DataFrame,
        df_pheno: pd.DataFrame,
        antibiotics: list,
        specials: dict,
        savepath_vocab: Path = None
    ):
    token_counter = Counter()
    ds_geno = df_geno.copy()
    ds_pheno = df_pheno.copy()
    
    PAD, UNK = specials['PAD'], specials['UNK']
    special_tokens = specials.values()
    
    year_geno = ds_geno[ds_geno['year'] != PAD]['year'].astype('Int16')
    min_year = min(year_geno.min(), ds_pheno['year'].min())
    max_year = max(year_geno.max(), ds_pheno['year'].max())
    year_range = range(min_year, max_year+1)
    token_counter.update([str(y) for y in year_range])
    
    min_age, max_age = ds_pheno['age'].min(), ds_pheno['age'].max()
    age_range = range(int(min_age), int(max_age+1))
    token_counter.update([str(a) for a in age_range])
    
    genders = ds_pheno['gender'].unique().astype(str).tolist()
    token_counter.update(genders)
    
    pheno_countries = ds_pheno['country'].sort_values().unique()
    geno_countries = ds_geno['country'].sort_values().dropna().unique()
    countries = set(pheno_countries).union(set(geno_countries))
    token_counter.update(countries)
    
    token_counter.update(list(chain(*ds_geno['genotypes'])))
    token_counter.update([ab + '_' + res for ab in antibiotics for res in ['R', 'S']])  
    
    vocab = Vocab(token_counter, specials=special_tokens)
    vocab.set_default_index(vocab[UNK])
    if savepath_vocab:
        torch.save(vocab, savepath_vocab)
    
    return vocab
    