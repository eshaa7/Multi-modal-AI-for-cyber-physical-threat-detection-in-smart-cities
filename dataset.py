from torch.utils.data import Dataset
import torch

from audio_preprocessing import get_spectrogram
from video_preprocessing import get_video_frames
from network_preprocessing import get_network_data


class MultiModalDataset(Dataset):
    def __init__(self, video_paths, audio_paths, network_paths, labels):
        self.video_paths = video_paths
        self.audio_paths = audio_paths
        self.network_paths = network_paths
        self.labels = labels

        # use single network file (as you only have one)
        self.network_path = network_paths[0]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        # ✅ consistent pairing (NO randomness)
        video_path = self.video_paths[idx]
        audio_path = self.audio_paths[idx % len(self.audio_paths)]

        # 🔹 Load data
        v = get_video_frames(video_path)      # [T, C, H, W]
        a = get_spectrogram(audio_path)       # [T, 128]
        n = get_network_data(self.network_path)  # [30, features]

        # 🔹 Label
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)

        return v, a, n, label