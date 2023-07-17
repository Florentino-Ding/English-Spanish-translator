import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

from data_perprocess import data_preprocessing, tokenlize, vocablize


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        x = self.embedding(x)
        x, (h, c) = self.lstm(x)
        return x, (h, c)


class TranslationDataset(Dataset):
    def __init__(self, file_dir: str, max_length=100):
        super(TranslationDataset, self).__init__()
        if os.path.exists(file_dir + "/processed_data"):
            pass
        else:
            eng_spa = pd.read_csv(
                file_dir + "/eng-spa.tsv", sep="\t", header=None, on_bad_lines="warn"
            )
            spa_eng = pd.read_csv(
                file_dir + "/spa-eng.tsv", sep="\t", header=None, on_bad_lines="warn"
            )
            eng_spa = data_preprocessing(eng_spa)
            spa_eng = data_preprocessing(spa_eng)
            spa_eng.rename(columns={3: "English", 1: "Spanish"}, inplace=True)
            eng_spa.rename(columns={1: "English", 3: "Spanish"}, inplace=True)
            data = pd.concat([eng_spa, spa_eng], axis=0, ignore_index=True)
            data = tokenlize(data)

        self.max_length = max_length

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    translationDataset = TranslationDataset(file_dir="data")
