# %% 
import os
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import time

from copy import deepcopy
from torch.utils.data import Dataset
from utils import get_genotype_to_ab_class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################################################################################
############################################### PRE-TRAINING DATASET ####################################################
########################################################################################################################

class MMPretrainDataset(Dataset):
    # df column names
    INDICES_MASKED = 'indices_masked' # input to BERT, token indices of the masked sequence
    TARGET_RESISTANCES = 'target_resistances' # resistance of the masked antibiotic, what we want to predict
    TARGET_IDS = 'target_ids' # indices of the target tokens for the genotype masking
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
        always_mask_replace: bool = False,
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
        self.CLS, self.PAD = specials['CLS'], specials['PAD']
        self.AB_MASK, self.GENE_MASK = specials['AB_MASK'], specials['GENE_MASK']        
        self.always_mask_replace = always_mask_replace
        
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
        
        self.columns = [self.INDICES_MASKED, self.TARGET_IDS, self.TARGET_RESISTANCES, self.TOKEN_TYPES]
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
        target_ids = torch.tensor(item[self.TARGET_IDS], dtype=torch.long, device=device) 
        attn_mask = (input != self.vocab[self.PAD]).unsqueeze(0).unsqueeze(1) # one dim for batch, one for heads   
        
        if self.include_sequences:
            # original_sequence = item[self.ORIGINAL_SEQUENCE]
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_ids, target_res, token_types, attn_mask, masked_sequence
        else:
            return input, target_ids, target_res, token_types, attn_mask
    
    
    def prepare_dataset(self):
        geno_sequences = deepcopy(self.ds_geno['genotypes'].tolist())
        pheno_sequences = deepcopy(self.ds_pheno['phenotypes'].tolist())
                
        masked_geno_sequences, geno_target_ids = self._mask_geno_sequences(geno_sequences)
        geno_target_resistances = [[-1]*self.num_ab for _ in range(self.num_geno)] # no ab masking for genotypes
        geno_token_types = [[0]*3 + [1]*(self.max_seq_len - 3) for _ in range(self.num_geno)]
        
        if self.masking_method == "keep_one_class":
            ab_classes = deepcopy(self.ds_pheno['ab_classes'].tolist())
            masked_pheno_sequences, pheno_target_resistances = self._mask_pheno_sequences(pheno_sequences, ab_classes)
        else:
            masked_pheno_sequences, pheno_target_resistances = self._mask_pheno_sequences(pheno_sequences)
        pheno_token_types = [[0]*5 + [2]*(self.max_seq_len - 5) for _ in range(self.num_pheno)]
        pheno_target_ids = [[-1]*self.max_seq_len for _ in range(self.num_pheno)]

        masked_sequences = masked_geno_sequences + masked_pheno_sequences
        indices_masked = [self.vocab.lookup_indices(masked_seq) for masked_seq in masked_sequences]
        target_ids = geno_target_ids + pheno_target_ids
        token_types = geno_token_types + pheno_token_types
        target_resistances = geno_target_resistances + pheno_target_resistances
        
        if self.include_sequences:
            rows = zip(indices_masked, target_ids, target_resistances, token_types,
                       masked_sequences)
        else:
            rows = zip(indices_masked, target_ids, target_resistances, token_types)
        self.df = pd.DataFrame(rows, columns=self.columns)
        

    def _get_replace_token(self, mask_token, original_token): ## BERT masking
        if self.always_mask_replace:
            return mask_token
        else:                           ## BERT masking
            r = self.rng.random()
            if r < 0.8:
                return mask_token
            elif r < 0.9:
                return self.vocab.lookup_token(self.rng.integers(self.vocab_size))
            else:
                return original_token

        
    def _mask_geno_sequences(self, geno_sequences):
        masked_geno_sequences = list()
        target_ids_list = list()
        
        years = self.ds_geno['year'].astype(str).tolist()
        countries = self.ds_geno['country'].tolist()
        seq_starts = [[self.CLS, years[i], countries[i]] for i in range(self.ds_geno.shape[0])]
        for i, geno_seq in enumerate(geno_sequences):
            seq_len = len(geno_seq)
            token_mask = self.rng.random(seq_len) < self.mask_prob_geno   
            target_ids = np.array([-1]*seq_len)
            if not token_mask.any():                ## if no tokens are masked, mask one random token
                idx = self.rng.integers(seq_len)
                target_ids[idx] = self.vocab[geno_seq[idx]]
                geno_seq[idx] = self._get_replace_token(self.GENE_MASK, geno_seq[idx])
            else:
                masking_indices = token_mask.nonzero()[0]
                target_ids[masking_indices] = self.vocab.lookup_indices([geno_seq[i] for i in masking_indices])
                for idx in masking_indices:
                    geno_seq[idx] = self._get_replace_token(self.GENE_MASK, geno_seq[idx])
            geno_seq = seq_starts[i] + geno_seq
            target_ids = [-1]*3 + target_ids.tolist() 
            masked_geno_sequences.append(geno_seq)
            target_ids_list.append(target_ids)
            
        masked_geno_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_geno_sequences]
        target_ids_list = [indices + [-1]*(self.max_seq_len - len(indices)) for indices in target_ids_list]
        return masked_geno_sequences, target_ids_list
    
    
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
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx]) 
                else:
                    for idx in token_mask.nonzero()[0]:
                        ab, res = pheno_seq[idx].split('_')
                        target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                        pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
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
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                pheno_seq = seq_starts[i] + pheno_seq
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
        elif self.masking_method == "keep_one_class":
            for i, pheno_seq in enumerate(pheno_sequences):
                classes = ab_classes[i]                # randomly choose one class to keep
                unique_classes, counts = np.unique(classes, return_counts=True)
                # freq = counts / counts.sum()
                # inv_freq = 1 / freq
                # prob = inv_freq / inv_freq.sum()
                # keep_class = self.rng.choice(unique_classes, p=prob) # less frequent classes are more likely
                keep_class = self.rng.choice(unique_classes) # all classes are equally likely
                # keep_class = self.rng.choice(classes) # more frequent classes are more likely
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                indices = [idx for idx in range(seq_len) if classes[idx] != keep_class]
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
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
    # TARGET_IDS = 'target_ids' # target indices of the masked sequence ## USE IF GENOTYPES ARE ALSO MASKED ##
    TARGET_RESISTANCES = 'target_resistances' # resistance of the target antibiotics, what we want to predict
    TOKEN_TYPES = 'token_types' # # 0 for patient info, 1 for genotype, 2 for phenotype
    TARGET_IDS = 'target_ids' # indices of the target tokens for the genotype masking
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
        no_geno_masking: bool = False,
        always_mask_replace: bool = True,
        filter_genes_by_ab_class: list = None,
        random_state: int = 42,
        include_sequences: bool = False,
    ):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state) # creates a new generator 
        
        self.ds = df_MM.reset_index(drop=True)
        assert all(self.ds['num_ab'] > 0), "Dataset contains isolates without phenotypes"
        
        
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics)
        self.ab_to_idx = {ab: idx for idx, ab in enumerate(self.antibiotics)}
        self.enc_res = {'S': 0, 'R': 1}
        self.max_seq_len = max_seq_len
        self.CLS, self.PAD = specials['CLS'], specials['PAD']
        self.AB_MASK, self.GENE_MASK = specials['AB_MASK'], specials['GENE_MASK']
        self.always_mask_replace = always_mask_replace
        
        self.filter_genes_by_ab_class = filter_genes_by_ab_class
        if self.filter_genes_by_ab_class:
            unique_genotypes = self.ds['genotypes'].explode().unique().tolist()
            genotype_to_ab_class = get_genotype_to_ab_class(unique_genotypes)
            print(f"Filtering genes by antibiotic classes: {self.filter_genes_by_ab_class}")
            ## Feature: remove filtered genes from the genotypes ##
            # self.ds['genotypes_filtered'] = self.ds['genotypes'].apply(
            #     lambda x: [gene for gene in x if not any(f in genotype_to_ab_class(gene) for f in self.filter_genes_by_ab_class)]
            # )
            ## Alternative feature: change the tokens of filtered genes to special token ##
            self.ds['genotypes_filtered'] = self.ds['genotypes'].apply(
                lambda x: [gene if not any(f in genotype_to_ab_class[gene] for f in self.filter_genes_by_ab_class) else self.PAD for gene in x]
            )   
            self.ds['num_genotypes_filtered'] = self.ds['genotypes_filtered'].apply(len)
            self.genotype_col = 'genotypes_filtered'
        else:
            self.genotype_col = 'genotypes'
        
        self.masking_method = masking_method # 'random', 'num_known' or 'keep_one_class'
        self.mask_prob_geno = mask_prob_geno
        self.mask_prob_pheno = mask_prob_pheno
        self.no_genotype_masking = no_geno_masking
        self.mask_prob_pheno = mask_prob_pheno
        self.num_known_ab = num_known_ab
        if self.masking_method == 'random':
            assert self.mask_prob_pheno, "mask_prob_pheno must be given if masking_method is 'random'"
        elif self.masking_method == 'num_known':
            assert self.num_known_ab, "num_known_ab must be given if masking_method is 'num_known'"
            self.ds = self.ds[self.ds['num_ab'] > self.num_known_ab].reset_index(drop=True)
        elif self.masking_method == 'keep_one_class':            
            self.ds = self.ds[self.ds['ab_classes'].apply(lambda x: len(set(x)) > 1)].reset_index(drop=True)
        self.num_samples = self.ds.shape[0]
                    
        self.include_sequences = include_sequences
        self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES, self.TARGET_IDS, self.TOKEN_TYPES]
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
        target_ids = torch.tensor(item[self.TARGET_IDS], dtype=torch.long, device=device)
        
        if self.include_sequences:
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_res, target_ids, token_types, attn_mask, masked_sequence
        else:
            return input, target_res, target_ids, token_types, attn_mask   
    
    
    def prepare_dataset(self):
        geno_sequences = deepcopy(self.ds[self.genotype_col].tolist())
        pheno_sequences = deepcopy(self.ds['phenotypes'].tolist())
        years = self.ds['year'].astype(str).tolist()
        countries = self.ds['country'].tolist()
        seq_starts = [[self.CLS, years[i], countries[i]] for i in range(self.num_samples)]
        
        if self.masking_method == "keep_one_class":
            ab_classes = deepcopy(self.ds['ab_classes'].tolist())
            masked_pheno_sequences, target_resistances, target_ids_pheno = self._mask_pheno_sequences(pheno_sequences, ab_classes)
        else:
            masked_pheno_sequences, target_resistances, target_ids_pheno = self._mask_pheno_sequences(pheno_sequences)
            
        pheno_token_types = [[2]*len(seq) for seq in masked_pheno_sequences]
        if self.no_genotype_masking:
            masked_geno_sequences = geno_sequences
            target_ids_geno = [[-1]*len(seq) for seq in masked_geno_sequences]
        else:
            masked_geno_sequences, target_ids_geno = self._mask_geno_sequences(geno_sequences)
        geno_token_types = [[1]*len(seq) for seq in masked_geno_sequences]
        
        # combine sequences and pad
        target_ids = [[-1]*3 + target_ids_geno[i] + target_ids_pheno[i] for i in range(self.num_samples)]
        target_ids = [indices + [-1]*(self.max_seq_len - len(indices)) for indices in target_ids]
        
        masked_sequences = [seq_starts[i] + masked_geno_sequences[i] + masked_pheno_sequences[i] for i in range(self.num_samples)]
        masked_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_sequences]
        indices_masked = [self.vocab.lookup_indices(seq) for seq in masked_sequences]
        
        token_types = [[0]*3 + geno_token_types[i] + pheno_token_types[i] for i in range(self.num_samples)]
        token_types = [seq + [2]*(self.max_seq_len - len(seq)) for seq in token_types]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_resistances, target_ids, token_types, masked_sequences)
        else:
            rows = zip(indices_masked, target_resistances, target_ids, token_types)
        self.df = pd.DataFrame(rows, columns=self.columns)
         
         
    def _mask_geno_sequences(self, geno_sequences): # Just to remove info, no prediction task
        masked_geno_sequences = list()
        geno_target_ids = list() 
        
        for geno_seq in geno_sequences:
            seq_len = len(geno_seq)
            target_ids = np.array([-1]*seq_len)
            if not seq_len == 0:
                token_mask = self.rng.random(seq_len) < self.mask_prob_geno
                if not token_mask.any():
                    idx = self.rng.integers(seq_len)
                    target_ids[idx] = self.vocab[geno_seq[idx]]
                    # geno_seq[idx] = self.PAD 
                    geno_seq[idx] = self.GENE_MASK
                else:
                    masking_indices = token_mask.nonzero()[0]
                    target_ids[token_mask] = self.vocab.lookup_indices([geno_seq[i] for i in masking_indices])
                    for idx in masking_indices:
                        # geno_seq[idx] = self.PAD 
                        geno_seq[idx] = self.GENE_MASK
            masked_geno_sequences.append(geno_seq)
            geno_target_ids.append(target_ids.tolist())
        return masked_geno_sequences, geno_target_ids
    
    
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
    
    
    def _mask_pheno_sequences(self, pheno_sequences, ab_classes=None):
        masked_pheno_sequences = list()
        target_resistances = list()
        pheno_target_ids = list()

        if self.mask_prob_pheno:
            for pheno_seq in pheno_sequences:
                seq_len = len(pheno_seq)
                token_mask = self.rng.random(seq_len) < self.mask_prob_pheno
                target_res = [-1]*self.num_ab
                target_ids = np.array([-1]*seq_len)
                if not token_mask.any():
                    idx = self.rng.integers(seq_len)
                    target_ids[idx] = self.vocab[pheno_seq[idx]]
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]  
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx]) 
                else:
                    masking_indices = token_mask.nonzero()[0]
                    target_ids[token_mask] = self.vocab.lookup_indices([pheno_seq[i] for i in masking_indices])
                    for idx in masking_indices:
                        ab, res = pheno_seq[idx].split('_')
                        target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                        pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())
        elif self.num_known_ab:
            for pheno_seq in pheno_sequences:
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                target_ids = np.array([-1]*seq_len)  
                masking_indices = self.rng.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                target_ids[masking_indices] = self.vocab.lookup_indices([pheno_seq[i] for i in masking_indices])
                for idx in masking_indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])                
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())  
        elif self.masking_method == "keep_one_class":
            for i, pheno_seq in enumerate(pheno_sequences):
                classes = ab_classes[i]     
                ## randomly choose one class to keep        
                unique_classes, counts = np.unique(classes, return_counts=True)
                # freq = counts / counts.sum()
                # inv_freq = 1 / freq
                # prob = inv_freq / inv_freq.sum()
                # keep_class = self.rng.choice(unique_classes, p=prob) # less frequent classes are more likely
                keep_class = self.rng.choice(unique_classes) # all classes are equally likely
                # keep_class = self.rng.choice(classes) # more frequent classes are more likely
                seq_len = len(pheno_seq)
                target_ids = np.array([-1]*seq_len)
                target_res = [-1]*self.num_ab
                indices = [idx for idx in range(seq_len) if classes[idx] != keep_class] 
                target_ids[indices] = self.vocab.lookup_indices([pheno_seq[i] for i in indices])
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")
        
        return masked_pheno_sequences, target_resistances, pheno_target_ids


########################################## FINE-TUNING on TESSy ########################################################


class PhenoFinetuneDataset(Dataset):
    # df column names
    # ORIGINAL_SEQUENCE = 'original_sequence'
    INDICES_MASKED = 'indices_masked' # input to BERT, token indices of the masked sequence
    # TARGET_IDS = 'target_ids' # target indices of the masked sequence ## USE IF GENOTYPES ARE ALSO MASKED ##
    TARGET_RESISTANCES = 'target_resistances' # resistance of the target antibiotics, what we want to predict
    TOKEN_TYPES = 'token_types' # # 0 for patient info, 1 for genotype, 2 for phenotype
    TARGET_IDS = 'target_ids' # indices of the target tokens for the genotype masking
    # if sequences are included
    MASKED_SEQUENCE = 'masked_sequence'
    
    def __init__(
        self,
        df_pheno: pd.DataFrame, 
        vocab,
        antibiotics: list,
        specials: dict,
        max_seq_len: int,
        masking_method: str,
        mask_prob_pheno: float,
        num_known_ab: int,
        always_mask_replace: bool = False,
        random_state: int = 42,
        include_sequences: bool = False,
    ):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state) # creates a new generator 
        
        self.ds = df_pheno.reset_index(drop=True)
        assert all(self.ds['num_ab'] > 0), "Dataset contains isolates without phenotypes"
        
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.antibiotics = antibiotics
        self.num_ab = len(self.antibiotics)
        self.ab_to_idx = {ab: idx for idx, ab in enumerate(self.antibiotics)}
        self.enc_res = {'S': 0, 'R': 1}
        self.max_seq_len = max_seq_len
        self.CLS, self.PAD, self.AB_MASK = specials['CLS'], specials['PAD'], specials['AB_MASK']
        self.always_mask_replace = always_mask_replace
        
        self.masking_method = masking_method # 'random', 'num_known' or 'keep_one_class'
        self.mask_prob_pheno = mask_prob_pheno
        self.mask_prob_geno = 0 # not used in this dataset, required in trainer # TODO: use better solution
        self.no_genotype_masking = False # not used in this dataset, required in trainer
        self.num_known_ab = num_known_ab
        if self.masking_method == 'random':
            assert self.mask_prob_pheno, "mask_prob_pheno must be given if masking_method is 'random'"
        elif self.masking_method == 'num_known':
            assert self.num_known_ab, "num_known_ab must be given if masking_method is 'num_known'"
            self.ds = self.ds[self.ds['num_ab'] > self.num_known_ab].reset_index(drop=True)
        elif self.masking_method == 'keep_one_class':            
            self.ds = self.ds[self.ds['ab_classes'].apply(lambda x: len(set(x)) > 1)].reset_index(drop=True)
        self.num_samples = self.ds.shape[0]
                    
        self.include_sequences = include_sequences
        self.columns = [self.INDICES_MASKED, self.TARGET_RESISTANCES, self.TARGET_IDS, self.TOKEN_TYPES]
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
        target_ids = torch.tensor(item[self.TARGET_IDS], dtype=torch.long, device=device)
        
        if self.include_sequences:
            masked_sequence = item[self.MASKED_SEQUENCE]
            return input, target_res, target_ids, token_types, attn_mask, masked_sequence
        else:
            return input, target_res, target_ids, token_types, attn_mask   
    
    
    def prepare_dataset(self):
        pheno_sequences = deepcopy(self.ds['phenotypes'].tolist())
        years = self.ds['year'].astype(str).tolist()
        countries = self.ds['country'].tolist()
        genders = self.ds['gender'].tolist()
        ages = self.ds['age'].astype(int).astype(str).tolist()
        seq_starts = [[self.CLS, years[i], countries[i], genders[i], ages[i]] for i in range(self.num_samples)]
        
        if self.masking_method == "keep_one_class":
            ab_classes = deepcopy(self.ds['ab_classes'].tolist())
            masked_pheno_sequences, target_resistances, target_ids = self._mask_pheno_sequences(pheno_sequences, ab_classes)
        else:
            masked_pheno_sequences, target_resistances, target_ids = self._mask_pheno_sequences(pheno_sequences)   
        pheno_token_types = [[2]*len(seq) for seq in masked_pheno_sequences]
        
        masked_geno_sequences =[[] for _ in range(self.num_samples)]
        target_ids_geno = [[-1]*len(seq) for seq in masked_geno_sequences]
        geno_token_types = [[1]*len(seq) for seq in masked_geno_sequences]
        
        # combine sequences and pad
        target_ids = [[-1]*3 + target_ids_geno[i] + target_ids[i] for i in range(self.num_samples)]
        target_ids = [indices + [-1]*(self.max_seq_len - len(indices)) for indices in target_ids]
        
        masked_sequences = [seq_starts[i] + masked_geno_sequences[i] + masked_pheno_sequences[i] for i in range(self.num_samples)]
        masked_sequences = [seq + [self.PAD]*(self.max_seq_len - len(seq)) for seq in masked_sequences]
        indices_masked = [self.vocab.lookup_indices(seq) for seq in masked_sequences]
        
        token_types = [[0]*3 + geno_token_types[i] + pheno_token_types[i] for i in range(self.num_samples)]
        token_types = [seq + [2]*(self.max_seq_len - len(seq)) for seq in token_types]
        
        if self.include_sequences:
            rows = zip(indices_masked, target_resistances, target_ids, token_types, masked_sequences)
        else:
            rows = zip(indices_masked, target_resistances, target_ids, token_types)
        self.df = pd.DataFrame(rows, columns=self.columns)
    
    
    def _get_replace_token(self, mask_token, original_token): 
        if self.always_mask_replace:
            return mask_token
        else:                           ## BERT masking
            r = self.rng.random()
            if r < 0.8:
                return mask_token
            elif r < 0.9:
                return self.vocab.lookup_token(self.rng.integers(self.vocab_size))
            else:
                return original_token
    
    
    def _mask_pheno_sequences(self, pheno_sequences, ab_classes=None):
        masked_pheno_sequences = list()
        target_resistances = list()
        pheno_target_ids = list()

        if self.mask_prob_pheno:
            for pheno_seq in pheno_sequences:
                seq_len = len(pheno_seq)
                token_mask = self.rng.random(seq_len) < self.mask_prob_pheno
                target_res = [-1]*self.num_ab
                target_ids = np.array([-1]*seq_len)
                if not token_mask.any():
                    idx = self.rng.integers(seq_len)
                    target_ids[idx] = self.vocab[pheno_seq[idx]]
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]  
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx]) 
                else:
                    target_ids[token_mask] = self.vocab.lookup_indices([pheno_seq[i] for i in token_mask.nonzero()[0]])
                    for idx in token_mask.nonzero()[0]:
                        ab, res = pheno_seq[idx].split('_')
                        target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                        pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())
        elif self.num_known_ab:
            for pheno_seq in pheno_sequences:
                seq_len = len(pheno_seq)
                target_res = [-1]*self.num_ab
                target_ids = np.array([-1]*seq_len)
                if self.num_known_ab == 0:
                    indices = range(seq_len)
                else:
                    indices = self.rng.choice(seq_len, seq_len - self.num_known_ab, replace=False)
                target_ids[indices] = self.vocab.lookup_indices([pheno_seq[i] for i in indices])
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())  
        elif self.masking_method == "keep_one_class":
            for i, pheno_seq in enumerate(pheno_sequences):
                classes = ab_classes[i]                # randomly choose one class to keep
                unique_classes, counts = np.unique(classes, return_counts=True)
                # freq = counts / counts.sum()
                # inv_freq = 1 / freq
                # prob = inv_freq / inv_freq.sum()
                # keep_class = self.rng.choice(unique_classes, p=prob) # less frequent classes are more likely
                keep_class = self.rng.choice(unique_classes) # all classes are equally likely
                # keep_class = self.rng.choice(classes) # more frequent classes are more likely
                seq_len = len(pheno_seq)
                target_ids = np.array([-1]*seq_len)
                target_res = [-1]*self.num_ab
                indices = [idx for idx in range(seq_len) if classes[idx] != keep_class] 
                target_ids[indices] = self.vocab.lookup_indices([pheno_seq[i] for i in indices])
                for idx in indices:
                    ab, res = pheno_seq[idx].split('_')
                    target_res[self.ab_to_idx[ab]] = self.enc_res[res]
                    pheno_seq[idx] = self._get_replace_token(self.AB_MASK, pheno_seq[idx])
                masked_pheno_sequences.append(pheno_seq)
                target_resistances.append(target_res)
                pheno_target_ids.append(target_ids.tolist())
        else:
            raise ValueError(f"Unknown masking method: {self.masking_method}")
        
        return masked_pheno_sequences, target_resistances, pheno_target_ids