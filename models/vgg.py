import torchvision.models as models

def vgg11(num_classes, pretrained=False):
    return models.vgg11(pretrained=pretrained, num_classes=num_classes)

def vgg13(pretrained=False):
    return models.vgg13(pretrained=pretrained)

def vgg16(pretrained=False):
    return models.vgg16(pretrained=pretrained)

def vgg19(pretrained=False):
    return models.vgg19(pretrained=pretrained)

def vgg11_bn(num_classes, pretrained=False):
    return models.vgg11_bn(pretrained=pretrained, num_classes=num_classes)

def vgg13_bn(pretrained=False):
    return models.vgg13_bn(pretrained=pretrained)

def vgg16_bn(pretrained=False):
    return models.vgg16_bn(pretrained=pretrained)

def vgg19_bn(pretrained=False):
    return models.vgg19_bn(pretrained=pretrained)
