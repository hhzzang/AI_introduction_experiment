import torch
import torchvision  # using torchvision, we can eaily download MNIST dataset
import torchvision.transforms as transforms  # to transform MNIST "images" to "tensor"
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn

bs = 10000
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_data = torchvision.datasets.CIFAR10(root='./datasets',
                                         train=True,
                                         transform=transform,
                                         download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=bs,
                                          shuffle=True)


class Test(nn.Module):
    def __init__(self, out_dim):
        super(Test, self).__init__()
        self.max_pool_kernel = 2
        self.num_class = out_dim
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1, padding=2),  # same = round((k-s)/2)
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_kernel)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_kernel)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(self.max_pool_kernel)
        )
        self.fc1 = nn.Linear(4 * 4 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        print(x.shape)
        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = x.reshape(x.size(0), -1)
        print(x.shape)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = F.relu(self.fc2(x))
        print(x.shape)
        x = F.softmax(self.fc3(x))
        print(x.shape)
        return x


test_model = Test(out_dim=10)
test_model.to(device)
test_model.load_state_dict(torch.load('./model.pth'))
test_model.eval()
with torch.no_grad():  # auto_grad off
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = test_model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images {}%'.format(len(test_loader) * bs, 100 * correct / total))
