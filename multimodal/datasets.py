# %% 
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import time

from copy import deepcopy
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################################
############################################### PRE-TRAINING DATASET ####################################################
########################################################################################################################

class MMPretrainDataset(Dataset):
    # df column names
    INDICES_MASKED = 'indices_masked' # input to BERT, token indices of the masked sequence
    TARGET_RESISTANCES = 'target_resistances' # resistance of the masked antibiotic, what we want to predict
    TARGET_INDICES = 'target_indices' # indices of the target tokens for the genotype masking
    TOKEN_TYPES = 'token_types' # 0 for patient info, 1 for genotype, 2 for phenotype
    # if sequences are included
    MASKED_SEQUENCE = 'masked_sequence'
    # ORIGINAL_SEQUENCE = 'original_sequence'
    
    
    def __init__(
        self, 
        ds_geno: pd.DataFrame,
        ds_pheno: pd.DataFrame,
        vocab,
        antibiotics: list,
        specials: dict,
        max_seq_len: int,
        mask_prob_geno: float,
        masking_method: str,
        mask_prob_pheno: float = None,
        num_known_ab: int = None,
        include_sequences: bool = False,
        random_state: int = 42
    ):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state) # creates a new generator
        
        self.ds_geno = ds_geno.reset_index(drop=True)
        self.num_geno = ds_geno.shape[0]
        self.ds_pheno = ds_pheno.reset_index(drop=True)
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.antibiotics = antibiotics
        self.num_ab = len(antibiotics)
        self.ab_to_idx = {ab: idx for idx, ab in enumerate(antibiotics)}
        self.enc_res = {'S': 0, 'R': 1}
        self.max_seq_len = max_seq_len
        self.CLS, self.PAD, self.MASK = specials['CLS'], specials['PAD'], specials['MASK']
        
        self.masking_method = masking_method # 'random', 'num_known' or 'keep_one_class'
        self.mask_prob_geno = mask_prob_geno
        self.mask_prob_pheno = mask_prob_pheno
        self.num_known_ab = num_known_ab
        if self.masking_method == 'random':
            assert self.mask_prob_pheno, "mask_prob_pheno must be given if masking_method is 'random'"
        elif self.masking_method == 'num_known':
            assert self.num_known_ab, "num_known_ab must be given if masking_method is 'num_known'"
            self.ds_pheno = self.ds_pheno[self.ds_pheno['num_ab'] > self.num_known_ab].reset_index(drop=True)
        # elif self.masking_method == 'keep_one_class':
        self.ds_pheno = self.ds_pheno[self.ds_pheno['ab_classes'].apply(lambda x: len(set(x)) > 1)].reset_index(drop=True)
        self.num_pheno = self.ds_pheno.shape[0]
        self.num_samples = self.num_geno + self.num_pheno
        
        self.ds_geno['source'] = 'geno'
        self.ds_pheno['source'] = 'pheno'
        geno_cols = ['year', 'country', 'num_genotypes', 'source']
        pheno_cols = ['year', 'country', 'gender', 'age', 'num_ab', 'source']
        self.combined_ds = pd.concat([self.ds_geno[geno_cols], self.ds_pheno[pheno_cols]], ignore_index=True)
        
        self.columns = [self.INDICES_MASKED, self.TARGET_INDICES, self.TARGET_RESISTANCES, self.TOKEN_TYPES]
        self.include_sequences = include_sequences
        if self.include_sequences:
            self.columns += [self.MASKED_SEQUENCE]
        
        
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        target_res = torch.tensor(item[self.TARGET_RESISTANCES], dtype=torch.float32, device=device)
        token_types = torch.tensor(item[self.TOKEN_TYPES], dtype=torch.long, device=device)
        target_indices = torch.tensor(item[self.TARGET_INDICES], dtype=torch.long, device=device) 
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads   
        
        if self.include_sequences:
            # original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_indices, target_res, token_types, attn_mask, masked_sequence
        else:
            return input, target_indices, target_res, token_types, attn_mask
    
    
    def prepare_dataset(self):
        geno_sequences = deepcopy(self.ds_geno['genotypes'].tolist())
        pheno_sequences = deepcopy(self.ds_pheno['phenotypes'].tolist())
                
        masked_geno_sequences, geno_target_indices = self._mask_geno_sequences(geno_sequences)
        geno_target_resistances = [[-1]*self.num_ab for _ in range(self.num_geno)] # no ab masking for genotypes
        geno_token_types = [[0]*3 + [1]*(self.max_seq_len - 3) for _ in range(self.num_geno)]
        
        if self.masking_method == "keep_one_class":
            ab_classes = deepcopy(self.ds_pheno['ab_classes'].tolist())
            masked_pheno_sequences, pheno_target_resistances = self._mask_pheno_sequences(pheno_sequences, ab_classes)
        else:
            masked_pheno_sequences, pheno_target_resistances = self._mask_pheno_sequences(pheno_sequences)
        pheno_token_types = [[0]*5 + [2]*(self.max_seq_len - 5) for _ in range(self.num_pheno)]
        pheno_target_indices = [[-1]*self.max_seq_len for _ in range(self.num_pheno)]

        masked_sequences = masked_geno_sequences + masked_pheno_sequences
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        target_indices = geno_target_indices + pheno_target_indices
        token_types = geno_token_types + pheno_token_types
        target_resistances = geno_target_resistances + pheno_target_resistances
        
        if self.include_sequences:
            rows = zip(indices_masked, target_indices, target_resistances, token_types,
                       masked_sequences)
        else:
            rows = zip(indices_masked, target_indices, target_resistances, token_types)
        self.df = pd.DataFrame(rows, columns=self.columns)
        
        
    def _mask_geno_sequences(self, geno_sequences):
        masked_geno_sequences = list()
        target_indices_list = list()
        
        years = self.ds_geno['year'].astype(str).tolist()
        countries = self.ds_geno['country'].tolist()
        seq_starts = [[self.CLS, years[i], countries[i]] for i in range(self.ds_geno.shape[0])]
        for i, geno_seq in enumerate(geno_sequences):
            seq_len = len(geno_seq)
            token_mask = self.rng.random(seq_len) < self.mask_prob_geno   
            target_indices = np.array([-1]*seq_len)
            if not token_mask.any():
                # if no tokens are masked, mask one random token
                idx = self.rng.integers(seq_len)
                target_indices[idx] = self.vocab[geno_seq[idx]]
                r = self.rng.random()
                if r < 0.8:
                    geno_seq[idx] = self.MASK
                elif r < 0.9:
                    geno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
            else:
                indices = token_mask.nonzero()[0]
                target_indices[indices] = self.vocab.lookup_indices([geno_seq[i] for i in indices])
                for i in indices:
                    r = self.rng.random()
                    if r < 0.8:
                        geno_seq[i] = self.MASK
                    elif r < 0.9:
                        geno_seq[i] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
            geno_seq = seq_starts[i] + geno_seq
            target_indices = [-1]*3 + target_indices.tolist() 
            masked_geno_sequences.append(geno_seq)
            target_indices_list.append(target_indices)
            
        masked_geno_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_geno_sequences]
        target_indices_list = [indices + [-1]*(self.max_seq_len - len(indices)) for indices in target_indices_list]
        return masked_geno_sequences, target_indices_list
    
    
    def _mask_pheno_sequences(self, pheno_sequences, ab_classes=None):
        masked_pheno_sequences = list()
        target_resistances = list()
        
        years = self.ds_pheno['year'].astype('Int16').astype(str).tolist()
        countries = self.ds_pheno['country'].tolist()
        genders = self.ds_pheno['gender'].tolist()
        ages = self.ds_pheno['age'].astype(int).astype(str).tolist()
        seq_starts = [[self.CLS, years[i], countries[i], genders[i], ages[i]] for i in range(self.num_pheno)]

        if self.mask_prob_pheno:
            for i, pheno_seq in enumerate(pheno_sequences):
                seq_len = len(pheno_seq)
                token_mask = self.rng.random(seq_len) < self.mask_prob_pheno
                target_res = [-1]*self.num_ab
                if not token_mask.any():
                    idx = self.rng.integers(seq_len)
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]  
                    r = self.rng.random()
                    if r < 0.8:
                        pheno_seq[idx] = self.MASK
                    elif r < 0.9:
                        pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size)) 
                else:
                    for idx in token_mask.nonzero()[0]:
                        ab, res = pheno_seq[idx].split('_')
                        target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                        r = self.rng.random()
                        if r < 0.8:
                            pheno_seq[idx] = self.MASK
                        elif r < 0.9:
                            pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
                pheno_seq = seq_starts[i] + pheno_seq
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
        elif self.num_known_ab:
            for i, pheno_seq in enumerate(pheno_sequences):
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                indices = self.rng.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    r = self.rng.random()
                    if r < 0.8:
                        pheno_seq[idx] = self.MASK
                    elif r < 0.9:
                        pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
                pheno_seq = seq_starts[i] + pheno_seq
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
        elif self.masking_method == "keep_one_class":
            for i, pheno_seq in enumerate(pheno_sequences):
                classes = ab_classes[i]                # randomly choose one class to keep
                unique_classes, counts = np.unique(classes, return_counts=True)
                freq = counts / counts.sum()
                inv_freq = 1 / freq
                prob = inv_freq / inv_freq.sum()
                keep_class = self.rng.choice(unique_classes, p=prob) # less frequent classes are more likely
                # keep_class = self.rng.choice(unique_classes) # all classes are equally likely
                # keep_class = self.rng.choice(classes) # more frequent classes are more likely
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                indices = [idx for idx in range(seq_len) if classes[idx] != keep_class]
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    r = self.rng.random()
                    if r < 0.8:
                        pheno_seq[idx] = self.MASK
                    elif r < 0.9:
                        pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")
        
        masked_pheno_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_pheno_sequences]
        return masked_pheno_sequences, target_resistances
            
    def shuffle(self):
        self.df = self.df.sample(frac=1, random_state=self.random_state)
        self.combined_ds = self.combined_ds.loc[self.df.index].reset_index(drop=True) # combined dataset is aligned with df
        self.df.reset_index(drop=True, inplace=True)

########################################################################################################################
############################################### FINE-TUNING DATASET ####################################################
########################################################################################################################


class MMFinetuneDataset(Dataset):
    # df column names
    # ORIGINAL_SEQUENCE = 'original_sequence'
    INDICES_MASKED = 'indices_masked' # input to BERT, token indices of the masked sequence
    # TARGET_INDICES = 'target_indices' # target indices of the masked sequence ## USE IF GENOTYPES ARE ALSO MASKED ##
    TARGET_RESISTANCES = 'target_resistances' # resistance of the target antibiotics, what we want to predict
    TOKEN_TYPES = 'token_types' # # 0 for patient info, 1 for genotype, 2 for phenotype
    TARGET_INDICES = 'target_indices' # indices of the target tokens for the genotype masking
    # if sequences are included
    MASKED_SEQUENCE = 'masked_sequence'
    
    def __init__(
        self,
        df_MM: pd.DataFrame, 
        vocab,
        antibiotics: list,
        specials: dict,
        max_seq_len: int,
        masking_method: str,
        mask_prob_geno: float,
        mask_prob_pheno: float,
        num_known_ab: int,
        random_state: int = 42,
        include_sequences: bool = False
    ):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state) # creates a new generator 
        
        self.ds_MM = df_MM.reset_index(drop=True)
        assert all(self.ds_MM['num_ab'] > 0), "Dataset contains isolates without phenotypes"
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics)
        self.ab_to_idx = {ab: idx for idx, ab in enumerate(self.antibiotics)}
        self.enc_res = {'S': 0, 'R': 1}
        self.max_seq_len = max_seq_len
        self.CLS, self.PAD, self.MASK = specials['CLS'], specials['PAD'], specials['MASK']
        
        self.masking_method = masking_method # 'random', 'num_known' or 'keep_one_class'
        self.mask_prob_geno = mask_prob_geno
        self.mask_prob_pheno = mask_prob_pheno
        self.num_known_ab = num_known_ab
        if self.masking_method == 'random':
            assert self.mask_prob_pheno, "mask_prob_pheno must be given if masking_method is 'random'"
        elif self.masking_method == 'num_known':
            assert self.num_known_ab, "num_known_ab must be given if masking_method is 'num_known'"
            self.ds_MM = self.ds_MM[self.ds_MM['num_ab'] > self.num_known_ab].reset_index(drop=True)
        elif self.masking_method == 'keep_one_class':            
            self.ds_MM = self.ds_MM[self.ds_MM['ab_classes'].apply(lambda x: len(set(x)) > 1)].reset_index(drop=True)
        self.num_samples = self.ds_MM.shape[0]
                    
        self.include_sequences = include_sequences
        self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES, self.TARGET_INDICES, self.TOKEN_TYPES]
        if self.include_sequences:
            self.columns += [self.MASKED_SEQUENCE]
            
    
    def __len__(self):
        return self.num_samples
    
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        
        input = torch.tensor(item[self.INDICES_MASKED], dtype=torch.long, device=device)
        target_res = torch.tensor(item[self.TARGET_RESISTANCES], dtype=torch.float32, device=device)
        token_types = torch.tensor(item[self.TOKEN_TYPES], dtype=torch.long, device=device)
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads
        target_indices = torch.tensor(item[self.TARGET_INDICES], dtype=torch.long, device=device)
        
        if self.include_sequences:
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_res, target_indices, token_types, attn_mask, masked_sequence
        else:
            return input, target_res, target_indices, token_types, attn_mask   
    
    
    def prepare_dataset(self):
        geno_sequences = deepcopy(self.ds_MM['genotypes'].tolist())
        pheno_sequences = deepcopy(self.ds_MM['phenotypes'].tolist())
        years = self.ds_MM['year'].astype(str).tolist()
        countries = self.ds_MM['country'].tolist()
        seq_starts = [[self.CLS, years[i], countries[i]] for i in range(self.num_samples)]
        
        if self.masking_method == "keep_one_class":
            ab_classes = deepcopy(self.ds_MM['ab_classes'].tolist())
            masked_pheno_sequences, target_resistances, target_indices_pheno = self._mask_pheno_sequences(pheno_sequences, ab_classes)
        else:
            masked_pheno_sequences, target_resistances, target_indices_pheno = self._mask_pheno_sequences(pheno_sequences)
            
        pheno_token_types = [[2]*len(seq) for seq in masked_pheno_sequences]
        masked_geno_sequences, target_indices_geno = self._mask_geno_sequences(geno_sequences)
        geno_token_types = [[1]*len(seq) for seq in masked_geno_sequences]
        
        # combine sequences and pad
        target_indices = [[-1]*3 + target_indices_geno[i] + target_indices_pheno[i] for i in range(self.num_samples)]
        target_indices = [indices + [-1]*(self.max_seq_len - len(indices)) for indices in target_indices]
        
        masked_sequences = [seq_starts[i] + masked_geno_sequences[i] + masked_pheno_sequences[i] for i in range(self.num_samples)]
        masked_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_sequences]
        indices_masked = [self.vocab.lookup_indices(seq) for seq in masked_sequences]
        
        token_types = [[0]*3 + geno_token_types[i] + pheno_token_types[i] for i in range(self.num_samples)]
        token_types = [seq + [2]*(self.max_seq_len - len(seq)) for seq in token_types]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_resistances, target_indices, token_types, masked_sequences)
        else:
            rows = zip(indices_masked, target_resistances, target_indices, token_types)
        self.df = pd.DataFrame(rows, columns=self.columns)
         
    
    def _mask_geno_sequences(self, geno_sequences): # Just to remove info, no prediction task
        masked_geno_sequences = list()
        geno_target_indices = list() 
        
        for geno_seq in geno_sequences:
            # self.rngshuffle(geno_seq) # if positional encoding is used, sequences ought to be shuffled
            seq_len = len(geno_seq)
            token_mask = self.rng.random(seq_len) < self.mask_prob_geno
            target_indices = np.array([-1]*seq_len)
            if not token_mask.any():
                idx = self.rng.integers(seq_len)
                target_indices[idx] = self.vocab[geno_seq[idx]]
                geno_seq[idx] = self.PAD            ## TODO: maybe change to an '[NA]' token 
            else:
                target_indices[token_mask] = self.vocab.lookup_indices([geno_seq[i] for i in token_mask.nonzero()[0]])
                for idx in token_mask.nonzero()[0]:
                    geno_seq[idx] = self.PAD
            masked_geno_sequences.append(geno_seq)
            geno_target_indices.append(target_indices.tolist())
        return masked_geno_sequences, geno_target_indices
    
    
    def _mask_pheno_sequences(self, pheno_sequences, ab_classes=None):
        masked_pheno_sequences = list()
        target_resistances = list()
        pheno_target_indices = list()

        if self.mask_prob_pheno:
            for pheno_seq in pheno_sequences:
                # self.rng.shuffle(pheno_seq) # if positional encoding is used, sequences ought to be shuffled
                seq_len = len(pheno_seq)
                token_mask = self.rng.random(seq_len) < self.mask_prob_pheno
                target_res = [-1]*self.num_ab
                target_indices = np.array([-1]*seq_len)
                if not token_mask.any():
                    idx = self.rng.integers(seq_len)
                    target_indices[idx] = self.vocab[pheno_seq[idx]]
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]  
                    r = self.rng.random()
                    if r < 0.8:
                        pheno_seq[idx] = self.MASK
                    elif r < 0.9:
                        pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size)) 
                else:
                    target_indices[token_mask] = self.vocab.lookup_indices([pheno_seq[i] for i in token_mask.nonzero()[0]])
                    for idx in token_mask.nonzero()[0]:
                        ab, res = pheno_seq[idx].split('_')
                        target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                        r = self.rng.random()
                        if r < 0.8:
                            pheno_seq[idx] = self.MASK
                        elif r < 0.9:
                            pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_indices.append(target_indices.tolist())
        elif self.num_known_ab:
            for pheno_seq in pheno_sequences:
                # self.rng.shuffle(pheno_seq) # if positional encoding is used, sequences ought to be shuffled
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                target_indices = np.array([-1]*seq_len)
                if self.num_known_ab == 0:
                    indices = range(seq_len)
                else:
                    indices = self.rng.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                target_indices[indices] = self.vocab.lookup_indices([pheno_seq[i] for i in indices])
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    r = self.rng.random()
                    if r < 0.8:
                        pheno_seq[idx] = self.MASK
                    elif r < 0.9:
                        pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                target_indices.append(target_indices.tolist())  
        elif self.masking_method == "keep_one_class":
            for i, pheno_seq in enumerate(pheno_sequences):
                # self.rng.shuffle(pheno_seq) # if positional encoding is used, sequences ought to be shuffled
                classes = ab_classes[i]                # randomly choose one class to keep
                unique_classes, counts = np.unique(classes, return_counts=True)
                freq = counts / counts.sum()
                inv_freq = 1 / freq
                prob = inv_freq / inv_freq.sum()
                keep_class = self.rng.choice(unique_classes, p=prob) # less frequent classes are more likely
                # keep_class = self.rng.choice(unique_classes) # all classes are equally likely
                # keep_class = self.rng.choice(classes) # more frequent classes are more likely
                seq_len = len(pheno_seq)
                target_indices = np.array([-1]*seq_len)
                target_res = [-1]*self.num_ab
                indices = [idx for idx in range(seq_len) if classes[idx] != keep_class] 
                target_indices[indices] = self.vocab.lookup_indices([pheno_seq[i] for i in indices])
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    r = self.rng.random()
                    if r < 0.8:
                        pheno_seq[idx] = self.MASK
                    elif r < 0.9:
                        pheno_seq[idx] = self.vocab.lookup_token(self.rng.integers(self.vocab_size))
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_indices.append(target_indices.tolist())
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")
        
        return masked_pheno_sequences, target_resistances, pheno_target_indices
        