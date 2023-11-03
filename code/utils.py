import numpy as np
import pandas as pd

def get_split_indices(size_to_split, split:list, random_state:int=42):
    indices = np.arange(size_to_split)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    assert sum(split) == 1, "Sum of split percentages must be 1."
    
    train_share, val_share, test_share = split[0], split[1], split[2]
    train_size = int(train_share * size_to_split)
    test_size = int(test_share * size_to_split)
    val_size = int(val_share * size_to_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    return train_indices, val_indices, test_indices


def filter_gene_counts(df, threshold_num):
    # get indices of samples with more than threshold_num genotypes
    indices = df[df['num_genotypes'] > threshold_num].index
    num_above = len(indices)
    # drop samples with more than threshold_num genotypes
    df.drop(indices, inplace=True)
    print(f"Dropping samples with more than {threshold_num} genotypes")
    print(f"Number of samples with more than {threshold_num} genotypes: {num_above:,}")
    return df


def impute_col(df, col, print_examples=False, random_state=42):
    print(f"Imputing column {col} from the distribution of non-NaN values")
    indices = df[df[col].isnull()].index
    examples_before = df.loc[indices, col][:5]
    if print_examples:
        print("Examples before imputation:")
        print(examples_before) 
    np.random.seed(random_state)
    sample = np.random.choice(df[col].dropna(), size=len(indices))
    df.loc[indices, col] = sample
    if print_examples:
        examples_after = df.loc[indices, col][:5]
        print(examples_after) 
    
    return df
    