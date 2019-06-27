from torchvision import datasets, transforms

normaliz = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = transforms.Compose([])

train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normaliz)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normaliz])

def CIFAR10(dataroot):
    train_dataset = datasets.CIFAR10(root=dataroot,
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(root=dataroot,
                                    train=False,
                                    transform=test_transform,
                                    download=True)
    num_classes = 10

    return train_dataset, test_dataset, num_classes

def CIFAR100(dataroot):
    train_dataset = datasets.CIFAR100(root=dataroot,
                                     train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR100(root=dataroot,
                                    train=False,
                                    transform=test_transform,
                                    download=True)
    num_classes = 100
    
    return train_dataset, test_dataset, num_classes
