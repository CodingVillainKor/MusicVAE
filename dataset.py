import torch
from torch.utils.data import DataLoader

import pickle as pkl

def col_fn(batch_list):
    batch = torch.stack([torch.tensor(item) for item in batch_list])
    return batch

def get_dataset():
    with open("dataset.pkl", "rb") as fr:
        dataset = pkl.load(fr)
    print(f"len(dataset) = {len(dataset):,}")
    
    return dataset