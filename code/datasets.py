# %% 
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import typing

from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext.vocab import vocab
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    ORIGINAL_SEQUENCE = 'original_sequence'
    MASKED_SEQUENCE = 'masked_sequence'
    
    
    def __init__(self,
                 ds_path: Path = None,
                 include_sequences: bool = False,
                 savepath_vocab: Path = None,
                 base_dir: Path = None,
                 subset_share: float = 1.0,
                 random_state: int = 42,
                 train_share: float = 0.8,
                 test_share: float = 0.1,
                 ):
        assert ds_path is not None, "Provide a data path"
        assert base_dir is not None, "Provide a base directory"
        
        os.chdir(base_dir)
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.ds = pd.read_pickle(ds_path).reset_index(drop=True) # reset index to avoid problems with torch methods
        self.ds = self.ds.sample(frac=subset_share, random_state=self.random_state).reset_index(drop=True) # subset of data
        self.num_samples = self.ds.shape[0]
        self.token_counter = Counter()
        self.sequences = list()
        self.vocab = None
        self.savepath_vocab = savepath_vocab
        self.train_share = train_share
        self.train_size = int(self.train_share * self.num_samples)
        self.test_share = test_share
        self.test_size = int(self.test_share * self.num_samples)
        self.val_share = 1 - train_share - test_share
        self.val_size = int(self.val_share * self.num_samples)
        
        self._create_vocabulary()
        max_genotypes_len = max([self.ds['num_genotypes'].iloc[i] for i in range(self.ds.shape[0]) if  
                                 self.ds['year'].iloc[i] != '[PAD]' and self.ds['country'].iloc[i] != '[PAD]'])
        self.max_seq_len = max_genotypes_len + 2 + 1 # +2 for year & country, +1 for CLS token
        
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK, 
                            self.ORIGINAL_SEQUENCE, self.MASKED_SEQUENCE]
        else: 
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK]
        
        
    def __len__(self):
        return len(self.ds)
    
    
    def __getitem__(self, idx):
        # direct to the correct dataframe of train, val, test
        if idx in self.train_indices:
            df = self.train_df
        elif idx in self.val_indices:
            df = self.val_df
        else:
            df = self.test_df
        item = df.loc[idx] # since the original indices are preserved
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        token_mask = torch.tensor(item[self.TOKEN_MASK], dtype=torch.bool, device=device)
        token_target = torch.tensor(item[self.TARGET_INDICES], dtype=torch.long, device=device)
        token_target = token_target.masked_fill_(~token_mask, -100) # -100 is default ignored index for NLLLoss
        attn_mask = (input == self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        
        if self.include_sequences:
            original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, token_target, token_mask, attn_mask, original_sequence, masked_sequence
        else:
            return input, token_target, token_mask, attn_mask

       
    def get_loaders(self, batch_size:int, mask_prob:float=0.15, split:bool=False):
        self.mask_prob = mask_prob
        self.df = self._prepare_dataset() # prepare dataset that will be split into train, val, test
        self._split_dataset() if split else None
        
        self.train_loader = DataLoader(self, batch_size=batch_size, sampler=SubsetRandomSampler(self.train_indices))
        self.val_loader = DataLoader(self, batch_size=batch_size, sampler=SubsetRandomSampler(self.val_indices))
        self.test_loader = DataLoader(self, batch_size=batch_size, sampler=SubsetRandomSampler(self.test_indices))
        
        return self.train_loader, self.val_loader, self.test_loader
    
    
    def _prepare_dataset(self): # will be called at the start of each epoch (dynamic masking)
        self._mask_dataset()
        self._construct_sequences()
    
        self.target_indices = [self.vocab.lookup_indices(seq) for seq in self.sequences]
        self.indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in self.masked_sequences]
        
        if self.include_sequences:
            rows = zip(self.indices_masked, self.target_indices, self.token_masks, self.sequences, self.masked_sequences)
        else:
            rows = zip(self.indices_masked, self.target_indices, self.token_masks)
        df = pd.DataFrame(rows, columns=self.columns)
        return df
    
    
    def _split_dataset(self):
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        self.train_indices = indices[:self.train_size]
        self.val_indices = indices[self.train_size:self.train_size + self.val_size]
        self.test_indices = indices[self.train_size + self.val_size:]
        
        self.train_df = self.df.iloc[self.train_indices]
        self.val_df = self.df.iloc[self.val_indices]
        self.test_df = self.df.iloc[self.test_indices]
    
        
    def _create_vocabulary(self):
        print("Constructing vocabulary...")
        
        year = self.ds['year'].astype('Int16')
        # year_range = [str(p)[:-12] for p in pd.date_range(start=str(year.min()), end=(year.max()), freq='M')] # for monthly
        year_range = range(year.min(), year.max()+1)
        self.token_counter.update([str(y) for y in year_range]) # make sure all years are included in the vocab
        
        # replace missing values with PAD token -> will not be included in vocabulary or in self-attention
        self.ds.fillna(self.PAD, inplace=True)
        # replace missing values with [NA] token -> will be included in vocabulary and in self-attention
        # NA = '[NA]' # not available, missing values
        # self.ds.fillna(NA, inplace=True)
        
        # update counter
        self.token_counter.update(list(chain(*self.ds['genotypes'])))
        self.token_counter.update(self.ds[self.ds['year'] != 'PAD]']['year'].tolist()) # count tokens that are not [PAD] (missing values)
        self.token_counter.update(self.ds[self.ds['country'] != 'PAD]']['country'].tolist())
        
        self.vocab = vocab(self.token_counter, specials=self.SPECIAL_TOKENS)
        self.vocab_size = len(self.vocab)
        self.vocab.set_default_index(self.vocab[self.UNK])
        torch.save(self.vocab, self.savepath_vocab) if self.savepath_vocab else None
    
    def _mask_dataset(self):
        # masking                               
        # RoBERTa: 80% -> [MASK], 10% -> original token, 10% -> random token
        self.sequences = self.ds['genotypes'].tolist()
        self.masked_sequences = list()
        self.token_masks = list()
        for seq in deepcopy(self.sequences):
            seq_len = len(seq)
            token_mask = [False] * seq_len
            tokens_masked = 0
            for i in range(seq_len):
                if np.random.rand() < self.mask_prob: 
                    tokens_masked += 1
                    r = np.random.rand()
                    if r < 0.8: 
                        seq[i] = self.MASK
                    elif r < 0.9:
                        j = np.random.randint(len(self.SPECIAL_TOKENS), self.vocab_size) # select random token, excluding specials
                        seq[i] = self.vocab.lookup_token(j)
                    # else: do nothing, since r > 0.9 and we keep the same token
                    token_mask[i] = True 
            if tokens_masked == 0: # mask at least one token
                i = np.random.randint(seq_len)
                r = np.random.rand()
                if r < 0.8: 
                    seq[i] = self.MASK
                elif r < 0.9:
                    j = np.random.randint(len(self.SPECIAL_TOKENS), self.vocab_size) # select random token, excluding specials
                    seq[i] = self.vocab.lookup_token(j)
                # else: do nothing, since r > 0.9 and we keep the same token
                token_mask[i] = True
                
            self.masked_sequences.append(seq)
            self.token_masks.append(token_mask)
    
    def _construct_sequences(self):  
        self.token_masks = [[False]*3 + mask for mask in self.token_masks] # always False for CLS, year, country
        for i in range(len(self.sequences)):
            seq_start = [self.CLS, self.ds['year'].iloc[i], self.ds['country'].iloc[i]]
            
            self.sequences[i][:0] = seq_start
            self.masked_sequences[i][:0] = seq_start
            
            seq_len = len(self.sequences[i])
            if seq_len < self.max_seq_len:
                self.sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                self.masked_sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                self.token_masks[i].extend([False] * (self.max_seq_len - seq_len))            
            else: # since we set max_seq_len to be the max length of genotypes + 2 + 1, this should not happen
                self.sequences[i] = self.sequences[i][:self.max_seq_len]
                self.masked_sequences[i] = self.masked_sequences[i][:self.max_seq_len]
                self.token_masks[i] = self.token_masks[i][:self.max_seq_len]      
    
    def reconstruct_seq_from_batch(self, seq_from_batch):
        tuple_len = len(seq_from_batch[0])
        sequences = list()
        for j in range(tuple_len):
            sequence = list()
            for i in range(self.max_seq_len):
                sequence.append(seq_from_batch[i][j])
            sequences.append(sequence)
        return sequences
