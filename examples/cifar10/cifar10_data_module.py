import torch
import torchvision
import torchvision.transforms as transforms


class Cifar10DataModule:

    feature_type = "ImageFeature"
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    @staticmethod
    def setup():
        transformer = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transformer)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=False, transform=transformer)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=2)
        return trainloader, testloader

    @staticmethod
    def get_batch_data():
        from captum.insights import Batch
        trainloader, testloader = Cifar10DataModule.setup()
        loader = iter(testloader)
        while True:
            inp_data = next(loader)
            yield Batch(inputs=inp_data[0], labels=inp_data[1])