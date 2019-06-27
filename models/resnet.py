import torchvision.models as models

def ResNet(num_classes, depth, pretrained=False):
    if depth == 18:
        return models.resnet18(pretrained, num_classes=num_classes)


# resnet18 = models.resnet18(pretrained=False)