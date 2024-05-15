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
                 masking_method: str,
                 num_known_ab: int = None,
                 mask_prob: float = None,
                 num_known_classes: int = None,
                 include_sequences: bool = False,
                 always_mask_replace: bool = False,
                 random_state: int = 42,
                 ):
        
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state) # creates a new generator
        
        self.ds = ds.reset_index(drop=True)
        self.masking_method = masking_method # 'random', 'num_known' or 'keep_one_class'
        self.mask_prob = mask_prob
        self.num_known_ab = num_known_ab
        self.num_known_classes = num_known_classes
        if self.masking_method == 'random':
            assert self.mask_prob, "mask_prob must be given if masking_method is 'random'"
        elif self.masking_method == 'num_known_ab':
            assert self.num_known_ab, "num_known_ab must be given if masking_method is 'num_known'"
            self.ds = self.ds[self.ds['num_ab'] > self.num_known_ab].reset_index(drop=True)
        elif self.masking_method == 'num_known_classes':
            assert num_known_classes, "num_known_classes must be given if masking_method is 'num_known_classes'"
            self.ds = self.ds[self.ds['ab_classes'].apply(lambda x: len(set(x)) > self.num_known_classes)].reset_index(drop=True)
        self.always_mask_replace = always_mask_replace
        if self.always_mask_replace:
            print("Always masking using MASK tokens")
        else:
            print("Masking using BERT 80-10-10 strategy")
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
        if self.masking_method == 'num_known_classes':
            ab_classes = deepcopy(self.ds['ab_classes'].tolist())
            masked_sequences, target_res = self._construct_masked_sequences(ab_classes=ab_classes)
        else:
            masked_sequences, target_res = self._construct_masked_sequences()
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
            
        if self.include_sequences:
            rows = zip(indices_masked, target_res, masked_sequences)
        else:
            rows = zip(indices_masked, target_res)
        self.df = pd.DataFrame(rows, columns=self.columns)
    
    
    def _get_replace_token(self, mask_token, original_token): 
        if self.always_mask_replace:
            return mask_token
        else:                       ## BERT masking
            r = self.rng.random()
            if r < 0.8:
                return mask_token
            elif r < 0.9:
                return self.vocab.lookup_token(self.rng.integers(self.vocab_size))
            else:
                return original_token
    
    def _construct_masked_sequences(self, ab_classes=None):  
        sequences = deepcopy(self.ds['phenotypes'].tolist())
        masked_sequences = list()
        target_resistances = list()
        
        years = self.ds['year'].astype(int).astype(str).tolist()
        countries = self.ds['country'].tolist()
        genders = self.ds['gender'].tolist()
        ages = self.ds['age'].astype(int).astype(str).tolist()
        seq_starts = [[self.CLS, years[i], countries[i], genders[i], ages[i]] for i in range(self.ds.shape[0])]
        
        if self.masking_method == 'random':
            for i, pheno_seq in enumerate(sequences):
                seq_len = len(pheno_seq)
                token_mask = self.rng.random(seq_len) < self.mask_prob
                target_res = [-1]*self.num_ab
                if not token_mask.any():
                    token_mask[self.rng.integers(seq_len)] = True
                for idx in token_mask.nonzero()[0]:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.MASK, pheno_seq[idx])
                pheno_seq = seq_starts[i] + pheno_seq
                masked_sequences.append(pheno_seq)
                target_resistances.append(target_res)
        elif self.masking_method == 'num_known_ab':
            for i, pheno_seq in enumerate(sequences):
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                indices = self.rng.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.MASK, pheno_seq[idx])
                pheno_seq = seq_starts[i] + pheno_seq
                masked_sequences.append(pheno_seq)
                target_resistances.append(target_res)
        elif self.masking_method == "num_known_classes":
            for i, pheno_seq in enumerate(sequences):
                classes = ab_classes[i]                # randomly choose one class to keep
                unique_classes, counts = np.unique(classes, return_counts=True)
                # freq = counts / counts.sum()
                # inv_freq = 1 / freq
                # prob = inv_freq / inv_freq.sum()
                # keep_classes = self.rng.choice(unique_classes, self.num_known_classes, replace=False, p=prob) # less frequent classes are more likely
                keep_classes = self.rng.choice(unique_classes, self.num_known_classes, replace=False) # all classes are equally likely
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                indices = [idx for idx in range(seq_len) if classes[idx] not in keep_classes]
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.MASK, pheno_seq[idx])
                masked_sequences.append(pheno_seq)
                target_resistances.append(target_res)
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")  
        
        masked_sequences = [seq_starts[i] + seq for i, seq in enumerate(masked_sequences)]
        masked_sequences = [seq + [self.PAD] * (self.max_seq_len - len(seq)) for seq in masked_sequences]      
        return masked_sequences, target_resistances 