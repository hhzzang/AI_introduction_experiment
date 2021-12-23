
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
learning_rate = 0.001
num_epochs = 4
input_size = 28
sequence_length = 28
out_node = 10 
 
valid_percent = 0.1  # train data를 train data, validation data 로 분리할 때의 비율 

##########plot & shape############
import matplotlib.pyplot as plt

train_dataset = torchvision.datasets.FashionMNIST(root='./datasets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
tr_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=16,
                                          shuffle=True)
for i in tr_loader:
    plt.imshow(i[0][0].view(28,28))
    print(i[0][0].shape)
    break
##################################

trtransform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.497))])
transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5), (0.497))])
train_dataset = torchvision.datasets.FashionMNIST(root='./datasets',
                                        train=True,
                                        transform=trtransform,
                                        download=True)
trdata, valdata = torch.utils.data.random_split(train_dataset, (54000,6000))

ttdata = torchvision.datasets.FashionMNIST(root='./datasets',
                                        train=False,
                                        transform=transform,
                                        download=True)

tr_loader = torch.utils.data.DataLoader(dataset=trdata,
                                          batch_size=batch_size,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=valdata,
                                          batch_size=batch_size,
                                          shuffle=False)
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


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
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)


def rnn_train(model, train_loader, optimizer, criterion):
    epoch_loss, acc = 0, 0
    total = 0
    model.train()

    for i, (image, label) in enumerate(train_loader):
        image = image.reshape(-1, sequence_length, input_size).to(device)
        label = label.to(device)
        # print(image.shape)
        # print(label.shape)
        out = model(image).to(device)   
        # print(out.shape)
        loss = criterion(out, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(out.data, 1)
        acc += (pred==label).sum()
        epoch_loss += loss.item()
        total += float(len(image))

    return epoch_loss, acc /total

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


print("____rnn____")
best_valid_loss = float('inf')
best_epoch = 0
for epoch in range(num_epochs):
    train_loss, train_acc = rnn_train(model, tr_loader, optimizer, criterion)
    valid_loss, valid_acc = rnn_valid(model,val_loader, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "./rnn_epoch_{}.pth".format(epoch))

    print("Epoch[{}/{}], Train Loss:{:.4f}, Train Acc:{:.2f}, Valid Loss:{:.4f}, Valid Acc:{:.2f}".format(epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))


print("____GRU____")
best_valid_loss = float('inf')
best_epoch = 0
for epoch in range(num_epochs+5):
    train_loss, train_acc = rnn_train(model2, tr_loader, optimizer2, criterion2)
    valid_loss, valid_acc = rnn_valid(model2,val_loader, criterion2)

    if valid_loss < best_valid_loss:
        print(epoch)
        best_valid_loss = valid_loss
        best_epoch = epoch
        torch.save(model2.state_dict(), "./gru_epoch_{}.pth".format(epoch))

    print("Epoch[{}/{}], Train Loss:{:.4f}, Train Acc:{:.2f}, Valid Loss:{:.4f}, Valid Acc:{:.2f}".format(epoch+1, num_epochs, train_loss, train_acc, valid_loss, valid_acc))
