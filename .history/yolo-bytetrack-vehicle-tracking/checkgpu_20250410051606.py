import torch
state_dict = torch.load("best_v8_CBAM.pt", map_location="cpu")
print(state_dict.keys())