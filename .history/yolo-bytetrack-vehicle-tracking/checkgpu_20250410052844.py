import torch
import ultralytics.nn.modules.conv
import torch.nn as nn

# Định nghĩa các lớp cần thiết cho CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ResBlock_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_CBAM, self).__init__()
        self.cbam = CBAM(in_channels)

    def forward(self, x):
        out = self.cbam(x)
        return out

# Đăng ký ResBlock_CBAM vào ultralytics.nn.modules.conv
setattr(ultralytics.nn.modules.conv, "ResBlock_CBAM", ResBlock_CBAM)

# Load file mô hình
state_dict = torch.load("best_v8_CBAM.pt", map_location="cpu")
print("Top-level keys:", state_dict.keys())

# Truy cập state dict từ đối tượng DetectionModel
if 'model' in state_dict:
    model_state = state_dict['model']
    # Lấy state dict từ đối tượng DetectionModel
    model_state_dict = model_state.state_dict()
    print("\nModel state dict keys:")
    for key in model_state_dict.keys():
        print(key)