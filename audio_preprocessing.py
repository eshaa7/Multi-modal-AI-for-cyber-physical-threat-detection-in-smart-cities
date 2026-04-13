import librosa
import numpy as np
import torch

def get_spectrogram(path):
    y, sr = librosa.load(path, sr=22050)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel)

    if mel.shape[1] < 50:
        mel = np.pad(mel, ((0,0),(0,50-mel.shape[1])))
    else:
        mel = mel[:, :50]

    return torch.tensor(mel.T, dtype=torch.float32)