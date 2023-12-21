import numpy as np
import pickle

def get_split_indices(size_to_split, val_share, random_state: int = 42):
    indices = np.arange(size_to_split)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_share = 1 - val_share
    
    train_size = int(train_share * size_to_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    return train_indices, val_indices


def get_multimodal_split_indices(sizes: list[int], val_share, random_state:int=42):
    train_indices = []
    val_indices = []
    np.random.seed(random_state)
    for size in sizes:
        assert isinstance(size, int), "sizes must be a list of integers"
        indices = np.arange(size)
        np.random.shuffle(indices)
        train_size = int((1 - val_share) * size)
        train_indices.append(indices[:train_size])
        val_indices.append(indices[train_size:])
    
    return train_indices, val_indices
    


def filter_gene_counts(df, threshold_num):
    # get indices of samples with more than threshold_num genotypes
    indices = df[df['num_genotypes'] > threshold_num].index
    num_above = len(indices)
    # drop samples with more than threshold_num genotypes
    df.drop(indices, inplace=True)
    print(f"Dropping {num_above} isolates with more than {threshold_num} genotypes")
    return df


def impute_col(df, col, random_state=42):
    print(f"Imputing column {col} from the distribution of non-NaN values")
    indices = df[df[col].isnull()].index
    np.random.seed(random_state)
    sample = np.random.choice(df[col].dropna(), size=len(indices))
    df.loc[indices, col] = sample
    
    return df


def export_results(results, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {savepath}")