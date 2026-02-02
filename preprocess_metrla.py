import pandas as pd
import numpy as np
import os
def normalize(data, mean=None, std=None):
    if mean is None or std is None:
        mean = data.mean(axis=0)  # ⚠️ per sensor
        std = data.std(axis=0) + 1e-6  # avoid division by zero
    data = (data - mean) / std
    return data, mean, std



def generate_npz(h5_path, save_dir, seq_len=12, pred_len=6):
    df = pd.read_hdf(h5_path)
    data = df.values  # shape: (timesteps, sensors)

    print("Raw data shape:", data.shape)

    # Normalize
    data, mean, std = normalize(data)

    # Create input-output sequences
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len):
        x_seq = data[i:i+seq_len]
        y_seq = data[i+seq_len:i+seq_len+pred_len]
        X.append(x_seq)
        Y.append(y_seq)

    X = np.stack(X)  # shape: (samples, seq_len, sensors)
    Y = np.stack(Y)  # shape: (samples, pred_len, sensors)

    print("Input shape:", X.shape)
    print("Target shape:", Y.shape)

    # Split: 70% train, 10% val, 20% test
    num_samples = X.shape[0]
    num_train = int(num_samples * 0.7)
    num_val = int(num_samples * 0.1)
    num_test = num_samples - num_train - num_val

    np.savez_compressed(os.path.join(save_dir, 'train.npz'),
                        x=X[:num_train], y=Y[:num_train])
    np.savez_compressed(os.path.join(save_dir, 'val.npz'),
                        x=X[num_train:num_train+num_val], y=Y[num_train:num_train+num_val])
    np.savez_compressed(os.path.join(save_dir, 'test.npz'),
                        x=X[-num_test:], y=Y[-num_test:])
    # Save mean and std for denormalization
    np.savez_compressed(os.path.join(save_dir, 'norm_stats.npz'), mean=mean, std=std)
    np.savez_compressed(os.path.join(save_dir, 'mean_std.npz'), mean=mean, std=std)

    print(f"Saved to: {save_dir}")

if __name__ == "__main__":
    h5_path = 'data/METR-LA/metr-la.h5'  # adjust if needed
    save_dir = 'data/METR-LA'
    generate_npz(h5_path, save_dir)
