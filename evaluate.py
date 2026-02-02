import torch
import numpy as np
from model.lsttn_model import LSTTNModel
from utils.data_loader import load_metrla_npz
from config.config import Config
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model():
    cfg = Config()

    # Load validation data
    val_path = cfg.data_path.replace("train.npz", "val.npz")
    X_val, y_val = load_metrla_npz(val_path)

    print(f"Validation set shape: {X_val.shape}, Targets: {y_val.shape}")

    input_dim = X_val.shape[-1]
    output_dim = y_val.shape[-1]

    # Load trained model
    model = LSTTNModel(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        output_dim=output_dim
    )
    model.load_state_dict(torch.load("saved_model.pt"))
    model.eval()

    # Predict
    with torch.no_grad():
        predictions = model(X_val).numpy()

    # Convert true values
    y_true = y_val.numpy()

    # Flatten all sensors and time steps
    y_true_flat = y_true.flatten()
    y_pred_flat = predictions.flatten()

    # Compute metrics
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    #rmse = mean_squared_error(y_true_flat, y_pred_flat, squared=False)
    mse = mean_squared_error(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mse)

    print(f"\n✅ Validation MAE: {mae:.4f}")
    print(f"✅ Validation RMSE: {rmse:.4f}")

if __name__ == "__main__":
    evaluate_model()