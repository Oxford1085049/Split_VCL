import torch
import numpy as np

class SubsetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, digits):
        self.dataset = dataset
        self.digits = digits

        self.indices = []
        for idx, (img, label) in enumerate(self.dataset):
            if label in self.digits:
                self.indices.append(idx)

    def __getitem__(self, idx):
        ori_idx = self.indices[idx]
        img, label = self.dataset[ori_idx]
        return img, label

    def __len__(self):
        return len(self.indices)

    def split_train_test(self, train_ratio=0.8):
        num_train = int(len(self) * train_ratio)
        num_test = len(self) - num_train

        train_indices = self.indices[:num_train]
        test_indices = self.indices[num_train:]

        train_subset = SubsetDataset(self.dataset, self.digits)
        train_subset.indices = train_indices

        test_subset = SubsetDataset(self.dataset, self.digits)
        test_subset.indices = test_indices

        return train_subset, test_subset

def compute_kl_divergence(mu_current, logvar_current, mu_previous, logvar_previous, reduction='sum'):
    kl_divergence = 1/2*(logvar_previous - logvar_current) + (torch.exp(logvar_current) + (mu_current - mu_previous)**2) / (2* (torch.exp(logvar_previous))) - .5
    if reduction == 'sum':
        return kl_divergence.sum()
    elif reduction == 'mean':
        return kl_divergence.mean()
    else:
        raise ValueError(f"Invalid reduction type: {reduction}")

import random

def create_coresets(train_datasets, coreset_size=200):
    coreset_datasets = []
    for train_dataset in train_datasets:
        coreset_indices = random.sample(range(len(train_dataset)), k=coreset_size)
        coreset = torch.utils.data.Subset(train_dataset, coreset_indices)
        coreset_datasets.append(coreset)
        # Remove coreset points from the original train set
        train_dataset = torch.utils.data.Subset(train_dataset, list(set(range(len(train_dataset))) - set(coreset_indices)))
    return coreset_datasets, train_datasets


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)