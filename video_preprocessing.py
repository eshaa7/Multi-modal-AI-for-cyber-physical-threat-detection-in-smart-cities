import os
import cv2
import torch

def get_video_frames(folder_path, num_frames=10):
    frames = []

    images = sorted(os.listdir(folder_path))[:num_frames]

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        frame = cv2.resize(frame, (64, 64))
        frame = torch.tensor(frame).permute(2, 0, 1) / 255.0
        frames.append(frame)

    # pad if less frames
    while len(frames) < num_frames:
        frames.append(torch.zeros(3, 64, 64))

    return torch.stack(frames)