# %% 
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from copy import deepcopy
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab
from collections import Counter

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
                 vocab: vocab,
                 specials: dict,
                 max_seq_len: int,
                 base_dir: Path,
                 include_sequences: bool = False,
                 random_state: int = 42,
                 ):
        
        os.chdir(base_dir)
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.ds = ds.reset_index(drop=True) 
        self.num_samples = self.ds.shape[0]
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.CLS = specials['CLS']
        self.PAD = specials['PAD']
        self.MASK = specials['MASK']
        self.UNK = specials['UNK']
        self.special_tokens = specials.values()
        self.max_seq_len = max_seq_len
        # self.max_seq_len = max([ds['num_genotypes'].iloc[i] for i in range(self.num_samples) if
                                # ds['year'].iloc[i] != self.PAD and ds['country'].iloc[i] != self.PAD]) + 2 + 1
        
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK, 
                            self.ORIGINAL_SEQUENCE, self.MASKED_SEQUENCE]
        else: 
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK]
        
        
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        token_mask = torch.tensor(item[self.TOKEN_MASK], dtype=torch.bool, device=device)
        token_target = torch.tensor(item[self.TARGET_INDICES], dtype=torch.long, device=device)
        token_target = token_target.masked_fill_(~token_mask, -100) # -100 is default ignored index for NLLLoss
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        
        if self.include_sequences:
            original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, token_target, token_mask, attn_mask, original_sequence, masked_sequence
        else:
            return input, token_target, token_mask, attn_mask

       
    def prepare_dataset(self, mask_prob: float = 0.15): # will be called at the start of each epoch (dynamic masking)
        sequences, masked_sequences, token_masks = self._construct_masked_sequences(mask_prob)

        target_indices = [self.vocab.lookup_indices(seq) for seq in sequences]
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_indices, token_masks, sequences, masked_sequences)
        else:
            rows = zip(indices_masked, target_indices, token_masks)
        self.df = pd.DataFrame(rows, columns=self.columns)

    
    def _construct_masked_sequences(self, mask_prob: float):  
        # masking                               
        # RoBERTa: 80% -> [MASK], 10% -> original token, 10% -> random token
        self.mask_prob = mask_prob
        sequences = deepcopy(self.ds['genotypes'].tolist())
        masked_sequences = list()
        token_masks = list()
        for seq in deepcopy(sequences):
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
                        j = np.random.randint(len(self.special_tokens), self.vocab_size) # select random token, excluding specials
                        seq[i] = self.vocab.lookup_token(j)
                    # else: do nothing, since r > 0.9 and we keep the same token
                    token_mask[i] = True 
            if tokens_masked == 0: # mask at least one token
                i = np.random.randint(seq_len)
                r = np.random.rand()
                if r < 0.8: 
                    seq[i] = self.MASK
                elif r < 0.9:
                    j = np.random.randint(len(self.special_tokens), self.vocab_size) # select random token, excluding specials
                    seq[i] = self.vocab.lookup_token(j)
                # else: do nothing, since r > 0.9 and we keep the same token
                token_mask[i] = True
                
            masked_sequences.append(seq)
            token_masks.append(token_mask)
        
        token_masks = [[False]*3 + mask for mask in token_masks] # always False for CLS, year, country
        for i in range(len(sequences)):
            seq_start = [self.CLS, self.ds['year'].iloc[i], self.ds['country'].iloc[i]]
            
            sequences[i][:0] = seq_start
            masked_sequences[i][:0] = seq_start
            
            seq_len = len(sequences[i])
            if seq_len < self.max_seq_len:
                sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                masked_sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                token_masks[i].extend([False] * (self.max_seq_len - seq_len))            
            else: # since we set max_seq_len to be the max length of genotypes + 2 + 1, this should not happen
                sequences[i] = sequences[i][:self.max_seq_len]
                masked_sequences[i] = masked_sequences[i][:self.max_seq_len]
                token_masks[i] = token_masks[i][:self.max_seq_len] 
                 
        return sequences, masked_sequences, token_masks    
    
    
    def reconstruct_sequence(self, seq_from_batch):
        tuple_len = len(seq_from_batch[0])
        sequences = list()
        for j in range(tuple_len):
            sequence = list()
            for i in range(self.max_seq_len):
                sequence.append(seq_from_batch[i][j])
            sequences.append(sequence)
        return sequences


####################################################################################################################################

class SimplePhenotypeDataset(Dataset):
           
    # df column names
    INDICES_MASKED = 'indices_masked'
    TARGET_INDICES = 'target_indices'
    TOKEN_MASK = 'token_mask'
    # if original text is included
    ORIGINAL_SEQUENCE = 'original_sequence'
    MASKED_SEQUENCE = 'masked_sequence'
    
    
    def __init__(self,
                 ds: pd.DataFrame,
                 vocab: vocab,
                 specials: dict,
                 max_seq_len: int,
                 base_dir: Path,
                 include_sequences: bool = False,
                 random_state: int = 42,
                 ):
        
        os.chdir(base_dir)
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.ds = ds.reset_index(drop=True) 
        self.num_samples = self.ds.shape[0]
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.CLS = specials['CLS']
        self.PAD = specials['PAD']
        self.MASK = specials['MASK']
        self.UNK = specials['UNK']
        self.special_tokens = specials.values()
        self.max_seq_len = max_seq_len
        
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK, 
                            self.ORIGINAL_SEQUENCE, self.MASKED_SEQUENCE]
        else: 
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK]
        
        
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        token_mask = torch.tensor(item[self.TOKEN_MASK], dtype=torch.bool, device=device)
        token_target = torch.tensor(item[self.TARGET_INDICES], dtype=torch.long, device=device)
        token_target = token_target.masked_fill_(~token_mask, -100) # -100 is default ignored index for NLLLoss
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        
        if self.include_sequences:
            original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, token_target, token_mask, attn_mask, original_sequence, masked_sequence
        else:
            return input, token_target, token_mask, attn_mask

       
    def prepare_dataset(self, mask_prob: float = 0.15): # will be called at the start of each epoch (dynamic masking)
        sequences, masked_sequences, token_masks = self._construct_masked_sequences(mask_prob)

        target_indices = [self.vocab.lookup_indices(seq) for seq in sequences]
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_indices, token_masks, sequences, masked_sequences)
        else:
            rows = zip(indices_masked, target_indices, token_masks)
        self.df = pd.DataFrame(rows, columns=self.columns)

    
    def _construct_masked_sequences(self, mask_prob: float):  
        # masking                               
        # RoBERTa: 80% -> [MASK], 10% -> original token, 10% -> random token
        self.mask_prob = mask_prob
        sequences = deepcopy(self.ds['phenotypes'].tolist())
        masked_sequences = list()
        token_masks = list()
        for seq in deepcopy(sequences):
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
                        j = np.random.randint(len(self.special_tokens), self.vocab_size) # select random token, excluding specials
                        seq[i] = self.vocab.lookup_token(j)
                    # else: do nothing, since r > 0.9 and we keep the same token
                    token_mask[i] = True 
            if tokens_masked == 0: # mask at least one token
                i = np.random.randint(seq_len)
                r = np.random.rand()
                if r < 0.8: 
                    seq[i] = self.MASK
                elif r < 0.9:
                    j = np.random.randint(len(self.special_tokens), self.vocab_size) # select random token, excluding specials
                    seq[i] = self.vocab.lookup_token(j)
                # else: do nothing, since r > 0.9 and we keep the same token
                token_mask[i] = True
                
            masked_sequences.append(seq)
            token_masks.append(token_mask)
        
        token_masks = [[False]*5 + mask for mask in token_masks] # always False for CLS, year, country, gender & age
        for i in range(len(sequences)):
            seq_start = [self.CLS, 
                         self.ds['year'].iloc[i], 
                         self.ds['country'].iloc[i], 
                         self.ds['gender'].iloc[i], 
                         str(self.ds['age'].iloc[i])]
            
            sequences[i][:0] = seq_start
            masked_sequences[i][:0] = seq_start
            
            seq_len = len(sequences[i])
            if seq_len < self.max_seq_len:
                sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                masked_sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                token_masks[i].extend([False] * (self.max_seq_len - seq_len))            
            else: # since we set max_seq_len to be the max length of genotypes + 2 + 1, this should not happen
                sequences[i] = sequences[i][:self.max_seq_len]
                masked_sequences[i] = masked_sequences[i][:self.max_seq_len]
                token_masks[i] = token_masks[i][:self.max_seq_len] 
                 
        return sequences, masked_sequences, token_masks    
    
    
    def reconstruct_sequence(self, seq_from_batch):
        tuple_len = len(seq_from_batch[0])
        sequences = list()
        for j in range(tuple_len):
            sequence = list()
            for i in range(self.max_seq_len):
                sequence.append(seq_from_batch[i][j])
            sequences.append(sequence)
        return sequences
    

class PhenotypeDataset(Dataset):
           
    # df column names
    INDICES_MASKED = 'indices_masked'
    TARGET_INDICES = 'target_indices'
    TOKEN_MASK = 'token_mask'
    # if original text is included
    ORIGINAL_SEQUENCE = 'original_sequence'
    MASKED_SEQUENCE = 'masked_sequence'
    
    
    def __init__(self,
                 ds: pd.DataFrame,
                 vocab: vocab,
                 specials: dict,
                 max_seq_len: int,
                 base_dir: Path,
                 include_sequences: bool = False,
                 random_state: int = 42,
                 ):
        
        os.chdir(base_dir)
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.ds = ds.reset_index(drop=True) 
        print(f"Proportion of S/R {self.ds['num_S'].sum()  / self.ds['num_phenotypes'].sum():.1%}/{self.ds['num_R'].sum() / self.ds['num_phenotypes'].sum():.1%}")
        self.num_samples = self.ds.shape[0]
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.CLS = specials['CLS']
        self.PAD = specials['PAD']
        self.MASK = specials['MASK']
        self.UNK = specials['UNK']
        self.special_tokens = specials.values()
        self.max_seq_len = max_seq_len
        
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK, 
                            self.ORIGINAL_SEQUENCE, self.MASKED_SEQUENCE]
        else: 
            self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TOKEN_MASK]
        
        
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        token_mask = torch.tensor(item[self.TOKEN_MASK], dtype=torch.bool, device=device)
        token_target = torch.tensor(item[self.TARGET_INDICES], dtype=torch.long, device=device)
        token_target = token_target.masked_fill_(~token_mask, -100) # -100 is default ignored index for NLLLoss
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        
        if self.include_sequences:
            original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, token_target, token_mask, attn_mask, original_sequence, masked_sequence
        else:
            return input, token_target, token_mask, attn_mask

       
    def prepare_dataset(self, mask_prob: float = 0.15): # will be called at the start of each epoch (dynamic masking)
        sequences, masked_sequences, token_masks = self._construct_masked_sequences(mask_prob)
        # target_indices = []
        # indices_masked = []
        # for i in range(len(sequences)):
        #     for j in range(len(sequences[i])):
        #         print("target token:", sequences[i][j])
        #         print("masked token:", masked_sequences[i][j])
        #         target_indices.append(self.vocab[sequences[i][j]])
        #         indices_masked.append(self.vocab[masked_sequences[i][j]])
        #     # target_indices.append([self.vocab[token] for token in sequences[i]])
        #     # indices_masked.append([self.vocab[token] for token in masked_sequences[i]])
        # print(target_indices[0:5])
        # print(indices_masked[0:5])
        target_indices = [self.vocab.lookup_indices(seq) for seq in sequences]
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_indices, token_masks, sequences, masked_sequences)
        else:
            rows = zip(indices_masked, target_indices, token_masks)
        self.df = pd.DataFrame(rows, columns=self.columns)

    
    def _construct_masked_sequences(self, mask_prob: float):  
        self.mask_prob = mask_prob
        phenotypes = deepcopy(self.ds['phenotypes'].tolist())
        sequences = list()
        masked_sequences = list()
        token_masks = list()
        for pheno_list in deepcopy(phenotypes):
            # seq = [p.split('_') + ['[SEP]'] for p in seq] # If separator is to be used
            seq = [p.split('_') for p in pheno_list]
            seq = [item for sublist in seq for item in sublist] # Flatten
            seq_len = len(seq)
            token_mask = [False] * seq_len
            masked_seq = deepcopy(seq)
            tokens_masked = 0
            for i_res in np.arange(1, seq_len, 2):
                if np.random.rand() < self.mask_prob:
                    masked_seq[i_res] = self.MASK
                    token_mask[i_res] = True
                    tokens_masked += 1
            if tokens_masked == 0:
                i_res = np.random.randint(len(pheno_list))*2 + 1 # choose random phenotype, convert to resistance index
                masked_seq[i_res] = self.MASK
                token_mask[i_res] = True
            sequences.append(seq)
            masked_sequences.append(masked_seq)
            token_masks.append(token_mask) 
        token_masks = [[False]*5 + mask for mask in token_masks] # always False for CLS, year, country, gender & age
        for i in range(len(sequences)):
            seq_start = [self.CLS, 
                         str(self.ds['year'].iloc[i]), 
                         self.ds['country'].iloc[i], 
                         self.ds['gender'].iloc[i], 
                         str(self.ds['age'].iloc[i])]
            
            sequences[i][:0] = seq_start
            masked_sequences[i][:0] = seq_start
            
            seq_len = len(sequences[i])
            if seq_len < self.max_seq_len:
                sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                masked_sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                token_masks[i].extend([False] * (self.max_seq_len - seq_len))            
            else: # since we set max_seq_len to be the max length of genotypes + 2 + 1, this should not happen
                sequences[i] = sequences[i][:self.max_seq_len]
                masked_sequences[i] = masked_sequences[i][:self.max_seq_len]
                token_masks[i] = token_masks[i][:self.max_seq_len]
        
        return sequences, masked_sequences, token_masks    
    
    
    def reconstruct_sequence(self, seq_from_batch):
        tuple_len = len(seq_from_batch[0])
        sequences = list()
        for j in range(tuple_len):
            sequence = list()
            for i in range(self.max_seq_len):
                sequence.append(seq_from_batch[i][j])
            sequences.append(sequence)
        return sequences