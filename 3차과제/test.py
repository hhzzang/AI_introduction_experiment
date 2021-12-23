
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 128
num_epochs = 4
input_size = 28
sequence_length = 28
out_node = 10 

transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.497))])

ttdata = torchvision.datasets.FashionMNIST(root='./datasets',
                                        train=False,
                                        transform=transform,
                                        download=True)

tt_loader = torch.utils.data.DataLoader(dataset=ttdata,
                                          batch_size=batch_size,
                                          shuffle=False)

class MY_RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10,bi=True):
        super(MY_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi = bi
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True) #1,28,28,50을 원하지 않음
        if bi == True:
            self.fc = nn.Linear(2*sequence_length*hidden_size, 512)
        else:
            self.fc = nn.Linear(sequence_length*hidden_size, 512)
        self.drops = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512,num_classes)
    def forward(self, x):
        out, _  = self.rnn(x) # output: tensor [batch_size, seq_length, hidden_size]
        if self.bi:
            out = out.reshape(-1,2*sequence_length*self.hidden_size)
        else:
            out = out.reshape(-1,sequence_length*self.hidden_size)
        out = F.relu(self.drops(self.fc(out)))
        out = self.fc2(out)
        return out

input_size = 28
num_layers = 1
hidden_dim = 128

model = MY_RNN(input_size, hidden_dim, num_layers, 10).to(device)
criterion = nn.CrossEntropyLoss()
class MY_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=10):
        super(MY_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True) #1,28,28,50을 원하지 않음
        # self.fc = nn.Linear(2*sequence_length*hidden_size, 512)
        self.fc = nn.Linear(hidden_size, 10)
        # self.drops = nn.Dropout(0.1)
        # self.fc2 = nn.Linear(512, 10)
    def forward(self, x):
        out, _  = self.gru(x) # output: tensor [batch_size, seq_length, hidden_size]
        out = out[:,-1,:]
        # out = out.reshape(-1,2*sequence_length*self.hidden_size)

        # out = F.relu(self.drops(self.fc(out)))
        # out = self.fc2(out)
        # out = self.fc(out)

        return self.fc(out)

    def init_weights(self, init_gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
        self.apply(init_func)


num_layers = 2
hidden_dim = 128
model2 = MY_GRU(input_size, hidden_dim, num_layers, 10).to(device)
criterion2 = nn.CrossEntropyLoss()

def rnn_valid(model, data_loader, criterion):
    epoch_loss, acc = 0, 0
    total = 0
    model.eval()

    with torch.no_grad():
        for i, (image, label) in enumerate(data_loader):
            image = image.reshape(-1, sequence_length, input_size).to(device)
            label = label.to(device) # 64

            out = model(image) # 64,10
            loss = criterion(out, label)
            _, pred = torch.max(out.data, 1)
            acc += (pred==label).sum()
            epoch_loss += loss.item()
            total += float(len(image))

    return epoch_loss, acc /total

model.load_state_dict(torch.load('./rnn_20140269.pth'))
rnn_loss, rnn_acc = rnn_valid(model, tt_loader, criterion)
print("\n rnn's fashion Accuracy:{:.2f}% (Loss:{:.4f})".format(rnn_acc,rnn_loss))

model2.load_state_dict(torch.load('./gru_20140269.pth'))
rnn_loss, rnn_acc = rnn_valid(model2, tt_loader, criterion2)
print("\n gru's fashion Accuracy:{:.2f}% (Loss:{:.4f})".format(rnn_acc,rnn_loss ))
