import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model.lsttn_model import LSTTNModel
from utils.data_loader import load_metrla_npz
from config.config import Config

def train():
    cfg = Config()

    print(f"Loading real traffic data from: {cfg.data_path}")
    X, y = load_metrla_npz(cfg.data_path)  # Load full dataset (no slicing!)
    print("DEBUG y shape:", y.shape)
    print(f"Input shape: {X.shape}, Target shape: {y.shape}")

    input_dim = X.shape[-1]
    #output_dim = y.shape[-1]
    output_dim = y.shape[1] * y.shape[2]  # 6 × 207 = 1242

    model = LSTTNModel(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=output_dim
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Training loop with progress bar
    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        # Progress bar over each sample (optional: use batches in future)
        with tqdm(total=X.shape[0], desc=f"Epoch {epoch+1}/{cfg.epochs}", unit="sample") as pbar:
            outputs = model(X)
            y = y.reshape(y.shape[0], -1)  # (batch, 1242)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            pbar.update(X.shape[0])  # All samples at once

        print(f"✅ Epoch [{epoch+1}/{cfg.epochs}], Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "saved_model.pt")
    print("✅ Model saved to saved_model.pt")

if __name__ == "__main__":
    train()
