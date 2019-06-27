import numpy as np
from torchvision import datasets, transforms

normaliz = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                     std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
train_transform = transforms.Compose([])

train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normaliz)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normaliz])

def SVHN(dataroot):
    train_dataset = datasets.SVHN(root=dataroot,
                                  split='train',
                                  transform=train_transform,
                                  download=True)

    extra_dataset = datasets.SVHN(root=dataroot,
                                  split='extra',
                                  transform=train_transform,
                                  download=True)

    # Combine both training splits (https://arxiv.org/pdf/1605.07146.pdf)
    data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
    labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
    train_dataset.data = data
    train_dataset.labels = labels

    test_dataset = datasets.SVHN(root=dataroot,
                                 split='test',
                                 transform=test_transform,
                                 download=True)
    num_classes = 10

    return train_dataset, test_dataset, num_classes
