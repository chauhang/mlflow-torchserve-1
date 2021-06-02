import json
import os
from argparse import ArgumentParser

import mlflow
import torch.nn as nn
import torch.optim as optim

import torch
import torchvision.transforms as transforms

from cifar10_data_module import Cifar10DataModule

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def baseline_func(input):
    return input * 0


class Cifar10Classifier(nn.Module):
    def __init__(self):
        super(Cifar10Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def training(trainloader):
        net = Cifar10Classifier()
        USE_PRETRAINED_MODEL = dict_args["use_pretrained_model"]
        if USE_PRETRAINED_MODEL:
            print("Using existing trained model")
            net.load_state_dict(torch.load('cifar_torchvision.pt'))
        else:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(net.parameters(), lr=dict_args["lr"], momentum=0.9)
            num_epochs = dict_args["max_epochs"]
            for epoch in range(num_epochs):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # get the inputs
                    inputs, labels = data
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    if i % 2000 == 1999:  # print every 2000 mini-batches
                        print('[%d, %5d] loss: %.3f' %
                              (epoch + 1, i + 1, running_loss / 2000))
                        running_loss = 0.0

            print('Finished Training')
            torch.save(net.state_dict(), 'cifar_torchvision.pt')
        return net


if __name__ == "__main__":

    parser = ArgumentParser(description="Titanic Captum Example")

    parser.add_argument(
        "--use_pretrained_model",
        type=bool,
        default=False,
        help="Use pretrained model or train from the scratch",
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        metavar="N",
        help="Number of epochs to be used for training",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate (default: 0.1)",
    )

    args = parser.parse_args()
    dict_args = vars(args)

    mlflow.autolog()
    dm = Cifar10DataModule()

    trainloader, testloader = dm.setup()

    net = Cifar10Classifier.training(trainloader)
    #########################################################################
    import cloudpickle
    model_pickle = cloudpickle.dumps(net)
    feature_type = "ImageFeature"
    baseline = cloudpickle.dumps(baseline_func)
    transform_pickle = cloudpickle.dumps(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    json_content = {}
    json_content["model"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'cifar_torchvision.pt')
    json_content["feature_type"] = feature_type
    json_content["baseline"] = baseline.decode('ISO-8859-1')
    json_content["transform"] = transform_pickle.decode('ISO-8859-1')
    json_content["classes"] = classes

    with open("cifar10_data.json", "w") as f:
        json.dump(json_content, f)
