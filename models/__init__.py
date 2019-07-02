from .resnet import *
from .vgg import *

import torchvision.models as models

def GoogleNet(num_classes, pretrained=False):
    return models.googlenet(num_classes, pretrained)