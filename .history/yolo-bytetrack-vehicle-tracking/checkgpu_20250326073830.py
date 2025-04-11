import torch

ckpt = torch.load("best12epoch.pt", map_location="cpu")
print(ckpt.keys())  # Kiểm tra các thành phần trong file .pt
