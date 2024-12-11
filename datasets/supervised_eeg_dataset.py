import numpy as np
import torch
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SupervisedEEGDataset(Dataset):
    """
    A PyTorch Dataset class for supervised EEG data.

    This dataset handles EEG signals and their corresponding labels for training and evaluation.
    """
    def __init__(self, eeg_signals: dict, transform: callable = None):
        """
        Initializes the SupervisedEEGDataset.

        Parameters:
        - eeg_signals (dict): Dictionary containing EEG signals per class.
                              Keys are labels, and values are lists of signals.
        - transform (callable, optional): Optional transform to be applied on each sample.
        """
        logger.debug(f"Creating SupervisedEEGDataset with {len(eeg_signals)} classes")
        data = np.concatenate([signals for signals in eeg_signals.values()])
        labels = np.concatenate([[label] * len(signals) for label, signals in eeg_signals.items()])
        self.data, self.labels = data, labels
        self.transform = transform
        logger.info(f"SupervisedEEGDataset created with {len(self.data)} samples")
        
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Number of samples.
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple:
        """
        Fetches a single sample from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - x (torch.Tensor): EEG signal as a PyTorch tensor with shape (1, Length).
        - y (torch.Tensor): Label as a PyTorch tensor.
        """
        x = self.data[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(y, dtype=torch.long)
        return x, y