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
    ds = dataset.copy() # use this if you don't want to modify the original dataset
    
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
    vocab_size = len(vocab)
    vocab.set_default_index(vocab[UNK])
    torch.save(vocab, savepath_vocab) if savepath_vocab else None
    return vocab
