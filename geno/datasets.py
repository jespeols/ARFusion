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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenotypeDataset(Dataset):
    # df column names
    INDICES_MASKED = 'indices_masked'
    TARGET_INDICES = 'target_indices'
    TOKEN_MASK = 'token_mask'
    # if original text is included
    ORIGINAL_SEQUENCE = 'original_sequence'
    MASKED_SEQUENCE = 'masked_sequence'
    
    
    def __init__(self,
                 ds: pd.DataFrame,
                 vocab: Vocab,
                 specials: dict,
                 max_seq_len: int,
                 mask_prob: float,
                 include_sequences: bool = False,
                 random_state: int = 42,
                 ):
        
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.ds = ds.reset_index(drop=True) 
        self.num_samples = self.ds.shape[0]
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.CLS, self.PAD, self.MASK, self.UNK = specials.values()
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, 
                            self.MASKED_SEQUENCE]
        else: 
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES]
        
        
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        target_indices = torch.tensor(item[self.TARGET_INDICES], dtype=torch.long, device=device)
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        
        if self.include_sequences:
            # original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_indices, attn_mask, masked_sequence
        else:
            return input, target_indices, attn_mask

       
    def prepare_dataset(self): # will be called at the start of each epoch (dynamic masking)
        masked_sequences, target_indices = self._construct_masked_sequences()

        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_indices, masked_sequences)
        else:
            rows = zip(indices_masked, target_indices)
        self.df = pd.DataFrame(rows, columns=self.columns)

    
    def _construct_masked_sequences(self):  
        # masking                               
        # RoBERTa: 80% -> [MASK], 10% -> original token, 10% -> random token
        sequences = deepcopy(self.ds['genotypes'].tolist())
        masked_sequences = list()
        target_indices_list = list()
        
        seq_starts = [[self.CLS, self.ds['year'].iloc[i], self.ds['country'].iloc[i]] for i in range(self.ds.shape[0])]
        for i, geno_seq in enumerate(sequences):
            # np.random.shuffle(geno_seq) # if positional encoding is used, sequences ought to be shuffled
            seq_len = len(geno_seq)
            token_mask = np.random.rand(seq_len) < self.mask_prob   
            target_indices = np.array([-1]*seq_len)
            if not token_mask.any():
                # if no tokens are masked, mask one random token
                idx = np.random.randint(seq_len)
                target_indices[idx] = self.vocab[geno_seq[idx]]
                r = np.random.rand()
                if r < 0.8:
                    geno_seq[idx] = self.MASK
                elif r < 0.9:
                    geno_seq[idx] = self.vocab.lookup_token(np.random.randint(self.vocab_size))
            else:
                indices = token_mask.nonzero()[0]
                target_indices[indices] = self.vocab.lookup_indices([geno_seq[i] for i in indices])
                for i in indices:
                    r = np.random.rand()
                    if r < 0.8:
                        geno_seq[i] = self.MASK
                    elif r < 0.9:
                        geno_seq[i] = self.vocab.lookup_token(np.random.randint(self.vocab_size))
            geno_seq = seq_starts[i] + geno_seq
            target_indices = [-1]*3 + target_indices.tolist() 
            masked_sequences.append(geno_seq)
            target_indices_list.append(target_indices)
            
        masked_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_sequences]
        target_indices_list = [indices + [-1]*(self.max_seq_len - len(indices)) for indices in target_indices_list]
        return masked_sequences, target_indices_list          