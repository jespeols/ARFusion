# %% 
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from copy import deepcopy
from itertools import chain
from torch.utils.data import Dataset
from torchtext.vocab import vocab as Vocab
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PhenotypeDataset(Dataset):      
    ## df column names
    INDICES_MASKED = 'indices_masked' # input to BERT, token indices of the masked sequence
    TARGET_RESISTANCES = 'target_resistances' # resistance of the target antibiotics, what we want to predict
    ## if sequences are included
    # ORIGINAL_SEQUENCE = 'original_sequence'
    MASKED_SEQUENCE = 'masked_sequence'
    
    def __init__(self,
                 ds: pd.DataFrame,
                 vocab,
                 antibiotics: list,
                 specials: dict,
                 max_seq_len: int,
                 num_known_ab: int = None,
                 mask_prob: float = None,
                 include_sequences: bool = False,
                 always_mask_replace: bool = False,
                 random_state: int = 42,
                 ):
        
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state) # creates a new generator
        
        self.ds = ds.reset_index(drop=True)
        assert num_known_ab or mask_prob, "Either num_known_ab or mask_prob must be specified"
        assert not (num_known_ab and mask_prob), "Only one of num_known_ab or mask_prob can be specified"
        self.num_known_ab = num_known_ab
        self.mask_prob = mask_prob
        self.always_mask_replace = always_mask_replace
        if self.num_known_ab:
            original_num_samples = self.ds.shape[0] 
            print(f"Preparing dataset for masking with {num_known_ab} known antibiotics")
            self.ds = self.ds[self.ds['num_ab'] > self.num_known_ab].reset_index(drop=True)
            num_samples = self.ds.shape[0]
            print(f"Original number of isolates: {original_num_samples:,}")
            print(f"Dropping {(original_num_samples - num_samples):,} isolates with less than {self.num_known_ab+1} antibiotics")
            tot_pheno = self.ds['num_ab'].sum()
            tot_S = self.ds['num_S'].sum()
            tot_R = self.ds['num_R'].sum()
            print(f"Now {num_samples:,} samples left")
        else:
            print(f"Preparing dataset for masking with mask_prob = {self.mask_prob}")
        tot_pheno = self.ds['num_ab'].sum()
        tot_S = self.ds['num_S'].sum()
        tot_R = self.ds['num_R'].sum()
        print(f"Proportion of S/R {tot_S / tot_pheno:.1%}/{tot_R / tot_pheno:.1%}")
        self.num_samples = self.ds.shape[0]
        self.vocab = vocab
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics)
        self.ab_to_idx = {ab: i for i, ab in enumerate(self.antibiotics)}
        self.enc_res = {'S': 0, 'R': 1}
        self.vocab_size = len(self.vocab)
        self.CLS, self.PAD, self.MASK = specials['CLS'], specials['PAD'], specials['MASK']
        self.max_seq_len = max_seq_len
           
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES, self.MASKED_SEQUENCE]
        else: 
            self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES]        
        
        
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        target_res = torch.tensor(item[self.TARGET_RESISTANCES], dtype=torch.float32, device=device)
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1)
        
        if self.include_sequences:
            # original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_res, attn_mask, masked_sequence
        else:
            return input, target_res, attn_mask

       
    def prepare_dataset(self): # will be called at the start of each epoch (dynamic masking)
        masked_sequences, target_res = self._construct_masked_sequences()
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_res, masked_sequences)
        else:
            rows = zip(indices_masked, target_res)
        self.df = pd.DataFrame(rows, columns=self.columns)
    
    
    def _get_replace_token(self, token, mask_token):
        # BERT: 80% -> [MASK], 10% -> original token, 10% -> random token   
        if self.always_mask_replace:
            return mask_token
        else:
            r = np.random.rand()
            # if r < 0.8:
            #     return mask_token
            # elif r < 0.9:
            #     return self.vocab.lookup_token(np.random.randint(self.vocab_size))
            # else:
            #     return token
            if r < 0.95: # include 5% chance of replacing with original token
                return mask_token
            else:
                return token
    
    def _construct_masked_sequences(self):  
        sequences = deepcopy(self.ds['phenotypes'].tolist())
        masked_sequences = list()
        target_resistances = list()
        
        years = self.ds['year'].astype(int).astype(str).tolist()
        countries = self.ds['country'].tolist()
        genders = self.ds['gender'].tolist()
        ages = self.ds['age'].astype(int).astype(str).tolist()
        seq_starts = [[self.CLS, years[i], countries[i], genders[i], ages[i]] for i in range(self.ds.shape[0])]
        
        if self.mask_prob:
            for i, seq in enumerate(sequences):
                seq_len = len(seq)
            
                target_res = [-1]*self.num_ab # -1 indicates padding, will indicate the target resistance, same indexing as ab_mask
                token_mask = np.random.rand(seq_len) < self.mask_prob
                if not token_mask.any(): # mask at least one token
                    idx = np.random.randint(seq_len)
                    ab, res = seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    seq[idx] = self._get_replace_token(seq[idx], self.MASK) ## BERT
                else:
                    for idx in token_mask.nonzero()[0]:
                        ab, res = seq[idx].split('_')
                        target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                        seq[idx] = self._get_replace_token(seq[idx], self.MASK) ## BERT
                seq = seq_starts[i] + seq
                masked_sequences.append(seq)
                target_resistances.append(target_res) 
        else:
            for i, seq in enumerate(sequences):
                seq_len = len(seq) 
                target_res = [-1]*self.num_ab # -1 indicates padding, will indicate the target resistance, same indexing as ab_mask

                # randomly select seq_len - num_known_ab antibiotics to mask
                mask_indices = np.random.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                for idx in mask_indices: 
                    ab, res = seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    seq[idx] = self._get_replace_token(seq[idx], self.MASK) ## BERT
                seq = seq_starts[i] + seq
                masked_sequences.append(seq)
                target_resistances.append(target_res)
                
        masked_sequences = [seq + [self.PAD] * (self.max_seq_len - len(seq)) for seq in masked_sequences]      
        return masked_sequences, target_resistances 