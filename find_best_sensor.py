import pandas as pd
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("predictions.csv")

sensor_count = (len(df.columns) - 1) // 2  # excluding sample_id

best_sensor = None
lowest_mae = float('inf')

for i in range(sensor_count):
    actual = df[f'sensor_{i}_actual']
    pred = df[f'sensor_{i}_pred']
    mae = mean_absolute_error(actual, pred)

    if mae < lowest_mae:
        lowest_mae = mae
        best_sensor = i

print(f"✅ Best performing sensor: sensor_{best_sensor} with MAE = {lowest_mae:.4f}")
