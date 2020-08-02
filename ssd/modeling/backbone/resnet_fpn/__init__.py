from ssd.modeling import registry
from .fpn import FPN
from .resnet import ResNet
from torchvision.models import resnet

@registry.BACKBONES.register('resnet50_fpn')
def resnet50_fpn(cfg, pretrained=True):
    # Build Model now
    model = FPN(
        ResNet(
            layers=[3, 4, 6, 3],
            bottleneck=resnet.Bottleneck,
            outputs=[3, 4, 5],
            url=resnet.model_urls['resnet50']
        )
    )
    # init weights
    model.initialize()
    return model
