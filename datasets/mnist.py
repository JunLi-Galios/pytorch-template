from torchvision import datasets, transforms

def MNIST(dataroot):

    train_dataset = datasets.MNIST(root=dataroot, train=True, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    test_dataset = datasets.MNIST(root=dataroot, train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]))

    num_classes = 10

    return train_dataset, test_dataset, num_classes
