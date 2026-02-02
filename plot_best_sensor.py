import pandas as pd
import matplotlib.pyplot as plt

# Replace this with the sensor number from Step 1
best_sensor = 65 # 👈 update this!

df = pd.read_csv("predictions.csv")

plt.figure(figsize=(10, 4))
plt.plot(df[f'sensor_{best_sensor}_actual'], label='Actual', linewidth=2)
plt.plot(df[f'sensor_{best_sensor}_pred'], label='Predicted', linestyle='--', linewidth=2)

plt.title(f"Sensor {best_sensor} - Actual vs Predicted")
plt.xlabel("Sample")
plt.ylabel("Normalized Traffic Speed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
