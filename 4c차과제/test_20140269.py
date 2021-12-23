import numpy as np
import torch
import os
import glob
import torchvision.transforms as transforms
import torchvision
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch import nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyDataset(Dataset):
    def __init__(self, npy_dir,tfms=None):
        self.dir_path = npy_dir
        self.to_tensor = transforms.ToTensor()

        # all npy path
        self.npy_path = glob.glob(os.path.join(npy_dir, '*','*.npy')) 
        self.tfms = tfms
    def __getitem__(self, index):
        # load data
        single_data_path = self.npy_path[index]
        data = np.load(single_data_path, allow_pickle=True)
        
        image = Image.fromarray(data[0])
        if self.tfms:
            image = self.tfms(image)
        else:
            image = self.to_tensor(image)
        label = data[1]
       
        return (image, label)

    def __len__(self):
        return len(self.npy_path)
def valid(model, data_loader, criterion):
    epoch_loss, acc = 0, 0
    total = 0
    model.eval()

    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image = image.to(device)
            label = label.to(device)

            out = model(image)
            loss = criterion(out, label)
            _, pred = torch.max(out.data, 1)
            acc += (pred==label).sum()
            epoch_loss += loss.item()
            total += float(len(image))

    return epoch_loss, acc /total

class MyNet(nn.Module):

    def __init__(self, block, layers, num_classe):
        super(MyNet, self).__init__()
        self._norm_layer = nn.BatchNorm2d

        self.dilation = 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3,
                               bias=False)
        
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)

        
        self.downsample1 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(64),
            )
        self.downsample2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(128),
            )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(4608, num_classe)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        ident = self.downsample1(x)
        x = self.conv2(x)
        x = self.bn2(x) + ident
        x = self.relu2(x)
        ident = self.downsample2(x)
        x = self.conv3(x)
        x = self.bn3(x) + ident
        x = self.relu3(x)
        
        x = torch.flatten(x,1)
#         print(x.shape)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

tttfm = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.6647), (0.73155))])

# load dataset 
valid_data = MyDataset("./Font_npy_90_val",tttfm)
batch_size = 512
valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size,
                                           )

model4 = MyNet(None, [1,1],52)
# model4.load_state_dict(torch.load('./model.pth'))
model4.load_state_dict(torch.load('./20140269.pth')) # 22epoch에서 제일 잘나왔을 때
model4.to(device)
criterion = nn.CrossEntropyLoss()

valid_loss, valid_acc = valid(model4,valid_loader, criterion)
print(valid_loss, valid_acc)
