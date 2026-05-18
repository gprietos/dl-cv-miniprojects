import torch

class SubsetWrapper(torch.utils.data.Dataset):
    """
    Wraps a Dataset subset to apply a specific transform to it,
    overriding the base dataset's original transform.
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # Grab the raw image and label from the underlying dataset
        # by bypassing the base transform temporarily
        path, label = self.subset.dataset.samples[self.subset.indices[index]]
        sample = self.subset.dataset.loader(path)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

    def __len__(self):
        return len(self.subset)