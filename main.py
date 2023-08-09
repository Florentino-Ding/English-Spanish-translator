import os
import sys
import re
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter

import data_perprocess


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        **kwargs
    ):
        super(Encoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            self.num_layers,
            dropout=dropout,
        )

    def forward(self, input_data, hidden_state):
        input_data = self.embedding(input_data)
        input_data = input_data.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        input_data, (h, c) = self.lstm(input_data, hidden_state)
        return input_data, (h, c)

    def init_state(self, batch_size, device=torch.device("cpu")):
        return (
            torch.randn(
                (
                    self.lstm.num_layers,
                    batch_size,
                    self.lstm.hidden_size,
                ),
                device=device,
            ),
            torch.randn(
                (
                    self.lstm.num_layers,
                    batch_size,
                    self.lstm.hidden_size,
                ),
                device=device,
            ),
        )


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layer: int = 1,
        dropout: float = 0.0,
        **kwargs
    ) -> None:
        super(Decoder, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            self.num_layers,
            dropout=dropout,
        )

    def forward(self, input_data, context):
        input_data = self.embedding(input_data)
        input_data = input_data.permute(1, 0, 2)  # (seq_len, batch_size, embedding_dim)
        input_data, (h, c) = self.lstm(input_data, context)

        return input_data, (h, c)


class translationModel:
    def __init__(
        self,
        encoder,
        decoder,
        epochs: int,
        encoder_lr: float = 0.01,
        decoder_lr: float = 0.01,
        clipping_theta: float = 1,
        loss_func=nn.CrossEntropyLoss,
        optim=torch.optim.Adam,
        log_dir="logs",
        device=torch.device("cpu"),
    ) -> None:
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.epochs = epochs
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.clipping_theta = clipping_theta
        self.loss_func = loss_func()
        self.optim = optim(
            [
                {"params": self.encoder.parameters(), "lr": self.encoder_lr},
                {"params": self.decoder.parameters(), "lr": self.decoder_lr},
            ]
        )
        self.log_dir = log_dir
        self.device = device

    def _save_data(self, version: int, model_dir=""):
        if model_dir:
            torch.save(
                self.encoder.state_dict(), "%s/encoder_%d.pth" % (model_dir, version)
            )
            torch.save(
                self.decoder.state_dict(), "%s/decoder_%d.pth" % (model_dir, version)
            )
            print("[INFO] Saving to %s_%s.pth" % (model_dir, version))

    def _load_data(self, version, model_dir=""):
        if model_dir:
            self.encoder.load_state_dict(
                torch.load(
                    "%s/encoder_%d.pth" % (model_dir, version), map_location=self.device
                )
            )
            self.decoder.load_state_dict(
                torch.load(
                    "%s/decoder_%d.pth" % (model_dir, version), map_location=self.device
                )
            )
            print("[INFO] Loading from %s_%s.pth" % (model_dir, version))

    def train(
        self,
        train_loader,
        model_dir: str = "model",
        save_frequency: int = 100,
        model_version: int = 0,
    ) -> None:
        writer = SummaryWriter(self.log_dir)
        if model_version:
            self._load_data(model_version, model_dir)

        for epoch in trange(self.epochs, desc="Epoch", file=sys.stdout):
            loader = tqdm(train_loader, leave=False, desc="Batch")
            for eng, spa in loader:
                # 准备数据
                eng, spa = eng.to(self.device), spa.to(self.device)
                self.optim.zero_grad()
                # 准备隐状态
                hidden_state = self.encoder.init_state(eng.shape[0])
                with autocast():
                    _, hidden_state = self.encoder(eng, hidden_state)
                    output, (h, c) = self.decoder(spa, hidden_state)
                    loss = self.loss_func(output, spa)
                scaler = GradScaler()
                scaler.scale(loss).backward()  # type: ignore
                scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(
                    [self.encoder.parameters(), self.decoder.parameters()],
                    self.clipping_theta,
                )
                scaler.step(self.optim)
                scaler.update()
                writer.add_scalar("loss", loss.item(), epoch)
            if (epoch + 1) % save_frequency == 0:
                self._save_data(epoch + 1, model_dir)


class TranslationDataset(Dataset):
    def __init__(self, file_dir: str, max_length=100):
        super(TranslationDataset, self).__init__()
        if os.path.exists(file_dir + "/data.npz"):
            data = np.load(file_dir + "/data.npz", allow_pickle=True)
            print("\n[INFO] Load the processed data successfully!")
            print("[INFO] The processed data is as follows:")
            print(data.files)
            self.Eng_data, self.Spa_data = data["Eng_data"], data["Spa_data"]
            self.Eng_word2idx, self.Spa_word2idx = (
                data["Eng_word2idx"].item(),
                data["Spa_word2idx"].item(),
            )
            self.Eng_idx2word, self.Spa_idx2word = (
                data["Eng_idx2word"].item(),
                data["Spa_idx2word"].item(),
            )
        else:
            eng_spa = pd.read_csv(
                file_dir + "/eng-spa.tsv", sep="\t", header=None, on_bad_lines="warn"
            )
            spa_eng = pd.read_csv(
                file_dir + "/spa-eng.tsv", sep="\t", header=None, on_bad_lines="warn"
            )
            eng_spa = data_perprocess.data_preprocessing(eng_spa)
            spa_eng = data_perprocess.data_preprocessing(spa_eng)
            spa_eng.rename(columns={3: "English", 1: "Spanish"}, inplace=True)
            eng_spa.rename(columns={1: "English", 3: "Spanish"}, inplace=True)
            data = pd.concat([eng_spa, spa_eng], axis=0, ignore_index=True)
            Eng_data, Spa_data = data_perprocess.tokenlize(data)
            self.Eng_word2idx, self.Eng_idx2word = data_perprocess.vocablize(Eng_data)
            self.Spa_word2idx, self.Spa_idx2word = data_perprocess.vocablize(Spa_data)
            self.Eng_data = data_perprocess.string2idx(Eng_data, self.Eng_word2idx)
            self.Spa_data = data_perprocess.string2idx(Spa_data, self.Spa_word2idx)

            data_perprocess.save_data(
                [self.Eng_data, self.Spa_data],
                [self.Eng_word2idx, self.Spa_word2idx],
                [self.Eng_idx2word, self.Spa_idx2word],
                file_dir + "/data",
            )
            print("\n[INFO] Save the processed data successfully!")
        self.max_length = max_length

    def __getitem__(self, idx):
        assert idx < len(self.Eng_data)
        return self.Eng_data[idx], self.Spa_data[idx]

    def __len__(self):
        return len(self.Eng_data)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

    translationDataset = TranslationDataset(file_dir="data")
    train_loader = DataLoader(
        translationDataset, batch_size=32, shuffle=True, num_workers=8
    )

    encoder = Encoder(
        vocab_size=len(translationDataset.Eng_word2idx),
        embedding_dim=256,
        hidden_dim=256,
    )
    decoder = Decoder(
        vocab_size=len(translationDataset.Spa_word2idx),
        embedding_dim=256,
        hidden_dim=256,
    )
    model = translationModel(encoder, decoder, epochs=100)

    model.train(train_loader, model_dir="model", save_frequency=10)
