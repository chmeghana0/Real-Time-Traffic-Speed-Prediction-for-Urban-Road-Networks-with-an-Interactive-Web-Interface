import os
import torch
import numpy as np
from flask import Flask, request, render_template
from model.lsttn_model import LSTTNModel
from config.config import Config

app = Flask(__name__)
cfg = Config()

# --- Load model ---
input_dim = 207
output_dim = 6 * 207  # 6 future steps × 207 sensors
model = LSTTNModel(input_dim, cfg.hidden_dim, output_dim)
model.load_state_dict(torch.load("saved_model.pt"))
model.eval()

# ✅ Load mean and std for denormalization
stats = np.load("data/METR-LA/norm_stats.npz")
mean = stats["mean"]  # shape: (207,)
std = stats["std"]    # shape: (207,)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file.filename == '':
        return 'No file uploaded.'

    path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(path)

    data = np.load(path)
    X = torch.tensor(data["x"]).float()
    y_true = torch.tensor(data["y"]).float()
    pred = model(X).view(-1, 6, 207)  # (batch, steps, sensors)
    y_true = y_true.view(-1, 6, 207)

    # ✅ Display: 5 samples × 3 sensors × 6 steps
    rows = []
    for i in range(min(5, X.shape[0])):
        for j in range(3):  # sensors 0,1,2
            actuals = [(y_true[i, step, j].item() * std[j] + mean[j]) for step in range(6)]
            preds = [(pred[i, step, j].item() * std[j] + mean[j]) for step in range(6)]
            row = {
                "sample": i,
                "sensor": j,
                "actuals": [f"{v:.2f}" for v in actuals],
                "preds": [f"{v:.2f}" for v in preds]
            }
            rows.append(row)

    return render_template("index.html", rows=rows)

if __name__ == "__main__":
    app.run(debug=True)
