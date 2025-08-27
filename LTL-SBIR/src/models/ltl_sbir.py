
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LearnableTransformLayer(nn.Module):
    """Depthwise 3x3 -> Pointwise 1x1 (+BN/ReLU) -> optional SE -> 1x1 back + residual."""
    def __init__(self, channels: int, pw_channels: int = 256, dw_kernel: int = 3, use_se: bool = True):
        super().__init__()
        pad = dw_kernel // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=dw_kernel, padding=pad, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, pw_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(pw_channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.se = SEBlock(pw_channels) if use_se else nn.Identity()
        self.out = nn.Conv2d(pw_channels, channels, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        y = self.act(self.bn2(self.pw(y)))
        y = self.se(y)
        y = self.act(self.bn3(self.out(y)))
        return x + y

class EmbeddingHead(nn.Module):
    def __init__(self, in_channels: int, emb_dim: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, emb_dim, bias=False)
        self.bn = nn.BatchNorm1d(emb_dim)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        x = self.bn(x)
        return F.normalize(x, p=2, dim=1)

def make_backbone(name: str, pretrained: bool = True):
    name = name.lower()
    if name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim
    if name == "vgg16":
        net = models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None)
        feat_dim = 512
        net.classifier = nn.Identity()
        return net, feat_dim
    raise ValueError(f"Unsupported backbone: {name}")

class SiameseLTLNet(nn.Module):
    def __init__(self, backbone="resnet50", emb_dim=512, shared_backbone=False,
                 ltl_pw_channels=256, ltl_dw_kernel=3, ltl_use_se=True):
        super().__init__()
        self.shared = shared_backbone
        self.bb_img, feat_dim = make_backbone(backbone, pretrained=True)
        if shared_backbone:
            self.bb_sketch = self.bb_img
        else:
            self.bb_sketch, _ = make_backbone(backbone, pretrained=True)

        self.ltl_img = LearnableTransformLayer(feat_dim, ltl_pw_channels, ltl_dw_kernel, ltl_use_se)
        self.ltl_sketch = LearnableTransformLayer(feat_dim, ltl_pw_channels, ltl_dw_kernel, ltl_use_se)
        self.head_img = EmbeddingHead(feat_dim, emb_dim)
        self.head_sketch = EmbeddingHead(feat_dim, emb_dim)

    def _forward_backbone(self, x, bb):
        if isinstance(bb, models.ResNet):
            x = bb.conv1(x); x = bb.bn1(x); x = bb.relu(x); x = bb.maxpool(x)
            x = bb.layer1(x); x = bb.layer2(x); x = bb.layer3(x); x = bb.layer4(x)
            return x
        elif isinstance(bb, models.VGG):
            return bb.features(x)
        else:
            raise NotImplementedError("Backbone not supported.")

    def forward_branch(self, x, branch="image"):
        bb = self.bb_img if (branch == "image" or self.shared) else self.bb_sketch
        fmap = self._forward_backbone(x, bb)
        ltl = self.ltl_img if branch == "image" else self.ltl_sketch
        fmap = ltl(fmap)
        head = self.head_img if branch == "image" else self.head_sketch
        return head(fmap)

    def forward(self, sketch, image):
        e_s = self.forward_branch(sketch, "sketch")
        e_i = self.forward_branch(image, "image")
        return e_s, e_i
