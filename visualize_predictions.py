import torch
import matplotlib.pyplot as plt
from model.lsttn_model import LSTTNModel
from utils.data_loader import load_metrla_npz
from config.config import Config

def plot_predictions(sensor_id=0, step=0, sample_count=100):
    cfg = Config()

    # Load validation data
    X, y = load_metrla_npz(cfg.data_path.replace("train", "val"))
    X = X[:sample_count]
    y = y[:sample_count]

    input_dim = X.shape[-1]
    output_dim = y.shape[1] * y.shape[2]  # 6 * 207 = 1242

    # Load trained model
    model = LSTTNModel(input_dim=input_dim, hidden_dim=cfg.hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load("saved_model.pt"))
    model.eval()

    with torch.no_grad():
        predictions = model(X).view(-1, 6, 207)

    # Plot actual vs predicted for the given time step and sensor
    y_true = y[:, step, sensor_id].numpy()
    y_pred = predictions[:, step, sensor_id].numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted", linestyle='--')
    plt.title(f"Sensor {sensor_id} - Step {step+1} (e.g. {5*(step+1)} min ahead)")
    plt.xlabel("Sample")
    plt.ylabel("Normalized Traffic Speed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_predictions(sensor_id=0, step=1)  # Step=0 means next 5 mins
