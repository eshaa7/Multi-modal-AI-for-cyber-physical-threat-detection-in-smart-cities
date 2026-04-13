import pandas as pd
import torch
import numpy as np

def get_network_data(path):
    df = pd.read_csv(path)
    df = df.select_dtypes(include=['number'])

    data = df.values

    if len(data) < 30:
        data = np.pad(data, ((0,30-len(data)),(0,0)))
    else:
        data = data[:30]

    if data.shape[1] < 40:
        data = np.pad(data, ((0,0),(0,40-data.shape[1])))
    else:
        data = data[:, :40]

    return torch.tensor(data, dtype=torch.float32)