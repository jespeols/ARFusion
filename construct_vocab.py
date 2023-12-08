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
    
    CLS, PAD, MASK, UNK = specials['CLS'], specials['PAD'], specials['MASK'], specials['UNK']
    special_tokens = specials.values() 
    
    year = ds[ds['year'] != PAD]['year'].astype('Int16')
    # year_range = [str(p)[:-12] for p in pd.date_range(start=str(year.min()), end=(year.max()), freq='M')] # for monthly
    year_range = range(year.min(), year.max()+1)
    token_counter.update([str(y) for y in year_range]) # make sure all years are included in the vocab
    
    # replace missing values with PAD token -> will not be included in vocabulary or in self-attention
    # ds.fillna(PAD, inplace=True)
    # replace missing values with [NA] token -> will be included in vocabulary and in self-attention
    # NA = '[NA]' # not available, missing values
    # ds.fillna(NA, inplace=True)
    
    # update counter
    token_counter.update(list(chain(*ds['genotypes'])))
    token_counter.update(ds[ds['year'] != PAD]['year'].tolist()) # count tokens that are not [PAD] (missing values)
    token_counter.update(ds[ds['country'] != PAD]['country'].tolist())
    
    vocab = Vocab(token_counter, specials=special_tokens)
    vocab.set_default_index(vocab[UNK])
    if savepath_vocab:
        torch.save(vocab, savepath_vocab)
    return vocab


def construct_pheno_vocab(dataset: pd.DataFrame, specials:dict, savepath_vocab: Path = None, separate_phenotypes: bool = True):
    token_counter = Counter()
    ds = dataset.copy() 
    
    CLS, PAD, MASK, UNK = specials['CLS'], specials['PAD'], specials['MASK'], specials['UNK']
    special_tokens = specials.values() 
    
    min_year, max_year = ds['year'].min(), ds['year'].max()
    year_range = range(min_year, max_year + 1)
    token_counter.update([str(y) for y in year_range]) 
    
    min_age, max_age = ds['age'].min(), ds['age'].max()
    age_range = range(int(min_age), int(max_age + 1))
    token_counter.update([str(a) for a in age_range])
    
    country = ds['country'].unique().astype('str').tolist()
    token_counter.update(country)
    
    gender = ds['gender'].unique().astype('str').tolist()
    token_counter.update(gender)
    
    unique_antibiotics = ds['phenotypes'].apply(lambda x: [p.split('_')[0] for p in x]).explode().unique().tolist()
    if separate_phenotypes:
        token_counter.update(unique_antibiotics)
        token_counter.update(['S', 'R'])
    else:
        token_counter.update([ab + '_' + res for ab in unique_antibiotics for res in ['S', 'R']])
    
    vocab = Vocab(token_counter, specials=special_tokens)
    vocab.set_default_index(vocab[UNK])
    torch.save(vocab, savepath_vocab) if savepath_vocab else None
    return vocab, unique_antibiotics
    
    
def construct_MM_vocab(df_geno: pd.DataFrame,
                       df_pheno: pd.DataFrame,
                       specials: dict,
                       savepath_vocab: Path = None):
    token_counter = Counter()
    ds_geno = df_geno.copy()
    ds_pheno = df_pheno.copy()
    
    CLS, PAD, MASK, UNK = specials['CLS'], specials['PAD'], specials['MASK'], specials['UNK']
    # SEP = specials['SEP']
    special_tokens = specials.values()
    
    min_year = min(ds_geno['year'].min(), ds_pheno['year'].min())
    max_year = max(ds_geno['year'].max(), ds_pheno['year'].max())
    year_range = range(min_year, max_year+1)
    token_counter.update([str(y) for y in year_range])
    
    min_age = min(ds_pheno['age'].min(), ds_pheno['age'].min())
    max_age = max(ds_pheno['age'].max(), ds_pheno['age'].max())
    age_range = range(min_age, max_age+1)
    token_counter.update([str(a) for a in age_range])
    