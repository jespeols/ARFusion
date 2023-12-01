# %% 
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from copy import deepcopy
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab as Vocab
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
                 vocab: Vocab,
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
    
class PhenotypeDataset(Dataset):      
    # df column names
    INDICES_MASKED = 'indices_masked' # input to BERT, token indices of the masked sequence
    TARGET_RESISTANCES = 'target_resistances' # resistance of the target antibiotics, what we want to predict
    TOKEN_MASK = 'token_mask' # True if token is masked, False otherwise
    AB_MASK = 'ab_mask' # True if antibiotic is masked, False otherwise
    # # if original text is included
    ORIGINAL_SEQUENCE = 'original_sequence'
    MASKED_SEQUENCE = 'masked_sequence'
    
    def __init__(self,
                 ds: pd.DataFrame,
                 vocab,
                 antibiotics: list,
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
        self.original_ds = deepcopy(self.ds) 
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
        self.CLS = specials['CLS']
        self.PAD = specials['PAD']
        self.MASK = specials['MASK']
        self.UNK = specials['UNK']
        self.special_tokens = specials.values()
        self.max_seq_len = max_seq_len
           
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES, self.TOKEN_MASK, self.AB_MASK,
                            self.ORIGINAL_SEQUENCE, self.MASKED_SEQUENCE]
        else: 
            self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES, self.TOKEN_MASK, self.AB_MASK]        
        
        
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        target_res = torch.tensor(item[self.TARGET_RESISTANCES], dtype=torch.float32, device=device)
        token_mask = torch.tensor(item[self.TOKEN_MASK], dtype=torch.bool, device=device)
        ab_mask = torch.tensor(item[self.AB_MASK], dtype=torch.bool, device=device)
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        
        if self.include_sequences:
            original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_res, token_mask, ab_mask, attn_mask, original_sequence, masked_sequence
        else:
            return input, target_res, token_mask, ab_mask, attn_mask

       
    def prepare_dataset(self,  mask_prob: float = None, num_known_ab: int = None): # will be called at the start of each epoch (dynamic masking)
        ## IT IS PROBABLY MORE EFFICIENT TO DO THIS IN THE PREPROCESSING STEP, GIVEN MASKING METHOD IS CONSTANT ACROSS EPOCHS
        if num_known_ab:
            print(f"Preparing dataset for masking with {num_known_ab} known antibiotics")
            self.num_known_ab = num_known_ab
            self.mask_prob = None
            self.ds = self.original_ds[self.original_ds['num_ab'] > self.num_known_ab].reset_index(drop=True)
            self.num_samples = self.ds.shape[0]
            print(f"Dropping {self.original_ds.shape[0] - self.num_samples} samples with less than {self.num_known_ab+1} antibiotics")
            tot_pheno = self.ds['num_ab'].sum()
            tot_S = self.ds['num_S'].sum()
            tot_R = self.ds['num_R'].sum()
            print(f"Now {self.num_samples} samples left, S/R proportion: {tot_S/tot_pheno:.1%}/{tot_R / tot_pheno:.1%}")
        else:
            print(f"Preparing dataset for masking with mask_prob = {mask_prob}")
            self.mask_prob = mask_prob
            self.num_known_ab = None
            
        sequences, masked_sequences, target_resistances, token_masks, ab_masks = self._construct_masked_sequences()
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_resistances, token_masks, ab_masks, sequences, masked_sequences)
        else:
            rows = zip(indices_masked, target_resistances, token_masks, ab_masks)
        self.df = pd.DataFrame(rows, columns=self.columns)

    
    def _encode_sequence(self, seq: list):
        dict = {ab: res for ab, res in [token.split('_') for token in seq]}
        indices = [self.ab_to_idx[ab] for ab in dict.keys()]
        resistances = [self.enc_res[res] for res in dict.values()]
        
        return indices, resistances
    
    
    def _construct_masked_sequences(self):  
        # RoBERTa: 80% -> [MASK], 10% -> original token, 10% -> random token
        
        sequences = deepcopy(self.ds['phenotypes'].tolist())
        masked_sequences = list()
        all_target_resistances = list()
        ab_masks = list() # will be applied to the output of the model, i.e. (batch_size, num_ab)
        token_masks = list() # will be applied to the the sequence itself, i.e. (batch_size, seq_len)
        for seq in deepcopy(sequences):
            seq_len = len(seq)
            
            token_mask = [False] * seq_len # indicates which tokens in the sequence are masked, includes all tokens
            ab_mask = [False] * self.num_ab # will indicate which antibiotics are masked, indexed in the order of self.antibiotics
            target_resistances = [-1]*self.num_ab # -1 indicates padding, will indicate the target resistance, same indexing as ab_mask
            if self.mask_prob:
                tokens_masked = 0
                for i in range(seq_len):
                    if np.random.rand() < self.mask_prob: 
                        ab, res = seq[i].split('_')
                        ab_idx = self.ab_to_idx[ab]
                        tokens_masked += 1
                        r = np.random.rand()
                        if r < 0.8: 
                            seq[i] = self.MASK
                        elif r < 0.9:
                            j = np.random.randint(self.vocab_size-self.num_ab*2, self.vocab_size) # select random pheno token
                            seq[i] = self.vocab.lookup_token(j)
                        # else: do nothing, since r > 0.9 and we keep the same token
                        token_mask[i] = True
                        ab_mask[ab_idx] = True # indicate which antibiotic is masked at this position
                        target_resistances[ab_idx] = self.enc_res[res] # the target resistance of the antibiotic
                if tokens_masked == 0: # mask at least one token
                    i = np.random.randint(seq_len)
                    ab, res = seq[i].split('_')
                    ab_idx = self.ab_to_idx[ab]
                    r = np.random.rand()
                    if r < 0.8: 
                        seq[i] = self.MASK
                    elif r < 0.9:
                        j = np.random.randint(self.vocab_size-self.num_ab*2, self.vocab_size) # select random token, excluding specials
                        seq[i] = self.vocab.lookup_token(j)
                    # else: do nothing, since r > 0.9 and we keep the same token
                    token_mask[i] = True
                    ab_mask[ab_idx] = True # indicate which antibiotic is masked at this position
                    target_resistances[ab_idx] = self.enc_res[res] # the target resistance of the antibiotic
            else:
                # randomly select seq_len - num_known_ab antibiotics to mask
                mask_indices = np.random.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                for i in mask_indices: # implement ROBERTa masking
                    ab, res = seq[i].split('_')
                    ab_idx = self.ab_to_idx[ab]
                    seq[i] = self.MASK
                    token_mask[i] = True
                    ab_mask[ab_idx] = True
                    target_resistances[ab_idx] = self.enc_res[res]
            masked_sequences.append(seq)
            token_masks.append(token_mask)
            ab_masks.append(ab_mask)
            all_target_resistances.append(target_resistances)
        
        for i in range(len(sequences)):
            token_masks[i] = 5*[False] + token_masks[i]
            seq_start = [self.CLS, 
                         str(self.ds['year'].iloc[i]), 
                         self.ds['country'].iloc[i], 
                         self.ds['gender'].iloc[i], 
                         str(int(self.ds['age'].iloc[i]))]
            
            sequences[i][:0] = seq_start
            masked_sequences[i][:0] = seq_start
            
            seq_len = len(sequences[i])
            if seq_len < self.max_seq_len:
                sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                masked_sequences[i].extend([self.PAD] * (self.max_seq_len - seq_len))
                token_masks[i].extend([False] * (self.max_seq_len - seq_len))
            # the antibiotic-specific lists should always be of length num_ab
            pheno_len = len(all_target_resistances[i])
            all_target_resistances[i].extend([-1] * (self.num_ab - pheno_len))
            # ab_mask is defined with correct length
        return sequences, masked_sequences, all_target_resistances, token_masks, ab_masks  
    
    
    def reconstruct_sequence(self, seq_from_batch):
        tuple_len = len(seq_from_batch[0])
        sequences = list()
        for j in range(tuple_len):
            sequence = list()
            for i in range(self.max_seq_len):
                sequence.append(seq_from_batch[i][j])
            sequences.append(sequence)
        return sequences 
    

class PhenotypeMLMDataset(Dataset):
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
                 base_dir: Path,
                 include_sequences: bool = False,
                 random_state: int = 42,
                 ):
        
        os.chdir(base_dir)
        self.random_state = random_state
        np.random.seed(self.random_state)
        
        self.ds = ds.reset_index(drop=True) 
        tot_pheno = self.ds['num_ab'].sum()
        tot_S = self.ds['num_S'].sum()
        tot_R = self.ds['num_R'].sum()
        print(f"Proportion of S/R {tot_S / tot_pheno:.1%}/{tot_R / tot_pheno:.1%}")
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
                         str(int(self.ds['age'].iloc[i]))]
            
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
