import torch
import torch.nn as nn
import os

from video_encoder import VideoEncoder
from audio_encoder import AudioEncoder
from network_encoder import NetworkEncoder
from fusion import CrossModalFusion


class UMTDNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ✅ Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Encoders
        self.video_encoder = VideoEncoder()
        self.audio_encoder = AudioEncoder()
        self.network_encoder = NetworkEncoder()

        # ✅ Load weights safely
        self._load_weights()

        # ✅ Freeze encoders
        self._freeze_encoders()

        # ✅ Fusion + Classifier
        self.fusion = CrossModalFusion()

        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    # 🔥 SAFE WEIGHT LOADING
    def _load_weights(self):
        base = os.path.dirname(__file__)

        video_path = os.path.join(base, "weights", "video.pth")
        audio_path = os.path.join(base, "weights", "audio.pth")
        network_path = os.path.join(base, "weights", "network.pth")

        try:
            self.video_encoder.load_state_dict(
                torch.load(video_path, map_location=self.device),
                strict=False
            )
            print("✅ Video weights loaded")
        except Exception as e:
            print("⚠️ Video weights NOT loaded:", e)

        try:
            self.audio_encoder.load_state_dict(
                torch.load(audio_path, map_location=self.device),
                strict=False
            )
            print("✅ Audio weights loaded")
        except Exception as e:
            print("⚠️ Audio weights NOT loaded:", e)

        try:
            self.network_encoder.load_state_dict(
                torch.load(network_path, map_location=self.device),
                strict=False
            )
            print("✅ Network weights loaded")
        except Exception as e:
            print("⚠️ Network weights NOT loaded:", e)

    # 🔥 FREEZE ENCODERS
    def _freeze_encoders(self):
        for param in self.video_encoder.parameters():
            param.requires_grad = False

        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        for param in self.network_encoder.parameters():
            param.requires_grad = False

    # 🔥 FORWARD PASS
    def forward(self, v, a, n):
        v = self.video_encoder(v)
        a = self.audio_encoder(a)
        n = self.network_encoder(n)

        fused = self.fusion(v, a, n)

        return self.classifier(fused)