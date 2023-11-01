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
