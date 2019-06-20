import numpy as np
import torch
from torch.utils import data
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.labels = labels
        self.data = data

    def __getitem__(self, index):
        row = self.data[index]
        row = torch.from_numpy(np.array(row, dtype='float32'))
        label = self.labels[index]
        label = torch.from_numpy(np.array(label))
        return row, label, torch.from_numpy(np.array(index))

    def __len__(self):
        n, _ = self.data.shape
        return n

