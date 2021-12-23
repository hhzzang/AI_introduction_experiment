import torch
import torchvision #using torchvision, we can eaily download MNIST dataset
import torchvision.transforms as transforms #to transform MNIST "images" to "tensor"
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
bs= 64
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data = torchvision.datasets.CIFAR10(root='./datasets',
                                        train=True,
                                        transform=transform,
                                        download=True)
test_data = torchvision.datasets.CIFAR10(root='./datasets',
                                       train=False,
                                       transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=bs,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=bs,
                                          shuffle=True)
out_dim = len(train_data.__dict__['classes'])


class Assignment(nn.Module):
    def __init__(self, out_dim):
        super(Assignment, self).__init__()
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
        self.init_weights()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

    def init_weights(self, init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
            if classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

model = Assignment(out_dim=out_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
num_epochs = 20
total_step = len(train_loader) * bs
loss_list = []

# Train
model.to(device)
for epoch in range(num_epochs):
    i = 0
    for images, labels in train_loader:
        i += bs
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss)

        # Print Loss for Tracking Training
        if i % (bs * 100) == 0:
            acc = 0
            test_image, test_label = next(iter(test_loader))
            _, test_predicted = torch.max(model(test_image.to(device)).data, 1)

            for (pred, ans) in zip(test_predicted, test_label):
                if pred == ans:
                    acc += 1
            acc = acc / len(test_predicted)

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.1f}%'.format(epoch + 1, num_epochs, i + 1,
                                                                                        total_step, loss.item(),
                                                                                        acc * 100))

    if acc * 100 >= 60.0:
        torch.save(model.state_dict(), f'./model{epoch}_{acc * 1000:.1f}.pth')