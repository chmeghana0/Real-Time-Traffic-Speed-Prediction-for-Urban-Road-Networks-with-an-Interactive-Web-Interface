import torch
import numpy as np
import pandas as pd
from model.lsttn_model import LSTTNModel
from utils.data_loader import load_metrla_npz
from config.config import Config

def save_predictions_to_csv(sample_count=100):
    cfg = Config()

    # Load validation data (or test if available)
    val_path = cfg.data_path.replace("train.npz", "val.npz")
    X, y = load_metrla_npz(val_path)
    X = X[:sample_count]
    y = y[:sample_count]

    input_dim = X.shape[-1]
    output_dim = y.shape[-1]

    # Load trained model
    model = LSTTNModel(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=output_dim
    )
    model.load_state_dict(torch.load("saved_model.pt"))
    model.eval()

    with torch.no_grad():
        predictions = model(X).numpy()
    y_true = y.numpy()

    # Prepare DataFrame
    records = []
    for i in range(sample_count):
        row = {'sample_id': i}
        for sensor_id in range(output_dim):
            row[f'sensor_{sensor_id}_actual'] = y_true[i, sensor_id]
            row[f'sensor_{sensor_id}_pred'] = predictions[i, sensor_id]
        records.append(row)

    df = pd.DataFrame(records)
    df.to_csv("predictions.csv", index=False)
    print("✅ Saved predictions to predictions.csv")

if __name__ == "__main__":
    save_predictions_to_csv()
