# %% 
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import typing

from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from torch.utils.data import Dataset
from torchtext.vocab import vocab
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the base directory
# base_dir = Path(os.path.abspath('')).parent # for notebooks
base_dir = Path(__file__).resolve().parent.parent 
os.chdir(base_dir)
print("base directory:", base_dir)

class GenotypeDataset(Dataset):
    CLS = '[CLS]'
    # SEP = '[SEP]'
    PAD = '[PAD]'
    UNK = '[UNK]'
    MASK = '[MASK]'
    SPECIAL_TOKENS = [CLS, PAD, MASK, UNK]
    
    # df column names
    INDICES_MASKED = 'indices_masked'
    TARGET_INDICES = 'target_indices'
    TOKEN_MASK = 'token_mask'
    # if original text is included
    ORIGINAL_SENTENCE = 'original_sentence'
    MASKED_SENTENCE = 'masked_sentence'
    
    
    def __init__(self,
                 path: Path = None,
                 include_original: bool = False,
                 save_vocab: bool = True,
                 ):
        assert path is not None, "Provide a data path"
        
        self.ds = pd.read_pickle(path)
        self.num_samples = self.ds.shape[0]
        self.counter = Counter()
        self.sequences = list()
        self.vocab = None
        
        self._create_vocabulary()

    
    def __len__(self):
        return len(self.ds)
    
    
    def _create_vocabulary(self):
        print("Constructing vocabulary...")
        
        year = self.ds['year'].astype('Int16')
        # year_range = [str(p)[:-12] for p in pd.date_range(start=str(year.min()), end=(year.max()), freq='M')] # for monthly
        year_range = range(year.min(), year.max()+1)
        self.counter.update([str(y) for y in year_range]) # make sure all years are included in the vocab
        
        # replace missing values with PAD token -> will not be included in vocabulary or in self-attention
        self.ds.fillna(self.PAD, inplace=True)
        # replace missing values with [NA] token -> will be included in vocabulary and in self-attention
        # NA = '[NA]' # not available, missing values
        # self.ds.fillna(NA, inplace=True)
        
        # tokens += self.ds['country'].unique().tolist()
        # tokens += list(set(chain(*self.ds['genotypes']))) # unique genotypes
        
        # update counter
        self.counter.update(list(chain(*self.ds['genotypes'])))
        self.counter.update(self.ds[self.ds['year'] != 'PAD]']['year'].tolist()) # count tokens that are not [PAD] (missing values)
        self.counter.update(self.ds[self.ds['country'] != 'PAD]']['country'].tolist())
        
        # for i in range(self.num_samples):
        #     full_sequence = [self.ds['year'].iloc[i], self.ds['country'].iloc[i]] + self.ds['genotypes'].iloc[i]
        #     filtered_sequence = [token for token in full_sequence if token != self.PAD] 
        #     # print(full_sequence, "\n", filtered_sequence) if i == 19 else None # print example
        #     self.counter.update(filtered_sequence)
        #     self.sequences.append(full_sequence)
        
        self.vocab = vocab(self.counter, specials=self.SPECIAL_TOKENS)
        self.vocab_size = len(self.vocab)
        self.vocab.set_default_index(self.vocab[self.UNK])
    
    def mask_dataset(self, mask_prob = 0.8):
        # masking
        # RoBERTa: 80% -> [MASK], 10% -> original token, 10% -> random token
        self.genotype_sequences = self.ds['genotypes'].tolist()
        self.masked_sequences = list()
        self.target_indices = list()
        self.token_masks = list()
        self.indices_masked = list()
        for seq in deepcopy(self.genotype_sequences.copy()):
            self.target_indices.append(self.vocab.lookup_indices(seq)) 
            seq_len = len(seq)
            token_mask = [False] * seq_len
            for i in range(seq_len):
                if np.random.rand() < mask_prob:
                    r = np.random.rand()
                    if r < 0.8: 
                        seq[i] = self.MASK
                    elif r < 0.9:
                        j = np.random.randint(len(self.SPECIAL_TOKENS), self.vocab_size) # select random token, excluding specials
                        seq[i] = self.vocab.lookup_token(j)
                    # else: do nothing, since r > 0.9 and we keep the same token
                    token_mask[i] = True 
            self.masked_sequences.append(seq)
            self.indices_masked.append(self.vocab.lookup_indices(seq))
            self.token_masks.append(token_mask)      

# %%
path = 'data/NCBI/genotype_parsed.pkl'
dataset = GenotypeDataset(path=path)
    
dataset.mask_dataset()
print("Done")
# %%
