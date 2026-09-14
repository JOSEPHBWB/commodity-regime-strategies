from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReturnLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 24, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, _ = self.lstm(x)
        return self.output(values[:, -1, :]).squeeze(-1)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    hidden_size: int = 24,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 7,
) -> ReturnLSTM:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ReturnLSTM(x_train.shape[-1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    x_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_tensor, y_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()
            optimizer.step()

    return model


def predict(model: ReturnLSTM, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        values = model(torch.tensor(x, dtype=torch.float32))
    return values.cpu().numpy()
