from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .resnet50 import resnet50_SSD300
from .resnet_fpn import resnet50_fpn

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'resnet50_SSD300', 'resnet50_fpn']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
