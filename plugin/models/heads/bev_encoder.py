import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import torch.nn.functional as F

from mmdet.models import HEADS

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        if (x1.shape[-2:] != x2.shape[-2:]):
            x1 = F.interpolate(x1, size=(x2.shape[-2:]))

        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

@HEADS.register_module()
class BevEncode(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(BevEncode, self).__init__()
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, out_channels, kernel_size=1, padding=0),
        )

    def forward(self, bev_feature):
        x2, x1 = self.encoder(bev_feature)

        x = self.up1(x2, x1)
        x = self.up2(x)

        return x
    
    def encoder(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x2 = self.layer3(x)
        
        return x2, x1


def onehot_encoding(logits, dim=1):
    max_idx = torch.argmax(logits, dim, keepdim=True)
    one_hot = logits.new_full(logits.shape, 0)
    one_hot.scatter_(dim, max_idx, 1)
    return one_hot
