import torchvision.models as models
  
def ResNet(num_classes, depth, pretrained=False):
    if depth == 18:
        return models.resnet18(pretrained, num_classes=num_classes)
    elif depth == 34:
        return models.resnet34(pretrained, num_classes=num_classes)
    elif depth == 50:
        return models.resnet50(pretrained, num_classes=num_classes)
    elif depth == 101:
        return models.resnet101(pretrained, num_classes=num_classes)
    elif depth == 152:
        return models.resnet152(pretrained, num_classes=num_classes)
