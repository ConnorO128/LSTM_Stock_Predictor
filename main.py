import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# Download Apple stock data and create features
data = yf.download("AAPL", start="2015-01-01", end="2024-01-01", auto_adjust=True)
data['Return'] = data['Close'].pct_change()
data['Volatility'] = data['Return'].rolling(20).std()
data['MA10'] = data['Close'].rolling(10).mean()
data = data.dropna()

# normalize features
features = ['Return', 'Volatility', 'MA10']
scaler = StandardScaler()
scaled = scaler.fit_transform(data[features])

# Create sequences of 30 days of features with up/down labels
x, y = [], []
days = 30
for i in range(len(scaled) - days):
    x.append(scaled[i:i+days])
    y.append(1 if data['Return'].iloc[i+days] > 0 else 0)

# Convert arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# 80/20 train-test split (important for time series)
split = int(0.8 * len(x_tensor))
x_train, x_test = x_tensor[:split], x_tensor[split:]
y_train, y_test = y_tensor[:split], y_tensor[split:]

# DataLoader for batching
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class LSTMModel(nn.Module):
    """
    LSTM model for binary classification of stock return direction.

    Args:
        input_dim (int): Number of features per timestep.
        hidden_dim (int): Number of hidden units in LSTM.
        num_layers (int): Number of stacked LSTM layers.
        output_dim (int): Number of output classes.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass of the LSTM.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim).
        
        Returns:
            torch.Tensor: Logits for each class.
        """
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last state
        out = self.fc(out)
        return out

#parameters
input_dim = 3
hidden_dim = 64
num_layers = 2
output_dim = 2
epochs = 60
lr = 0.001

# initialize model, loss, and optimizer
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# loop for training
for epoch in range(epochs):
    for xb, yb in train_loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} / {epochs}, Loss: {loss.item():.4f}")

# Evaluate accuracy
with torch.no_grad():
    preds = model(x_test)
    preds_labels = torch.argmax(preds, dim=1)
    accuracy = (preds_labels == y_test).float().mean().item()
print(f"Test Accuracy: {accuracy:.2f}")

# Backtest strategy
signals = preds_labels.numpy()
returns = data['Return'].iloc[-len(signals):].values
strategy_returns = signals * returns
cumulative = (1 + strategy_returns).cumprod()

# Compute Sharpe ratio
sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe:.2f}")
