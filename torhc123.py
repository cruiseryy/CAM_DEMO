import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from utils import sst_prcp_ds
from torch.utils.data import DataLoader
from torch import nn

cchannel = 1
batch_size = 12

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=cchannel, out_channels=32, kernel_size=(8,4))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=12, kernel_size=(4,2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(26400, 32)
        self.fc2 = nn.Linear(32, 1)
   

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # x = self.gap(x)
        # x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


training_data = sst_prcp_ds(channel=cchannel, lag=0, start=0, end=240)

test_data = sst_prcp_ds(channel=cchannel, lag=0, start=240, end=480)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using {device} device")

model = NeuralNetwork().to(device)
learning_rate = 0.01

epochs = 100

loss_fn = nn.BCELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

pause = 1
def train_loop(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        y = torch.reshape(y, (len(y), 1))
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += torch.sum((y>0.5)*(pred>0.5) + (y<0.5)*(pred<0.5)).item()
        
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size
    print(f"\n Accuracy: {(100*correct):>0.1f}%")


def test_loop(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            y = torch.reshape(y, (len(y), 1))
            test_loss += loss_fn(pred, y).item()
            correct += torch.sum((y>0.5)*(pred>0.5) + (y<0.5)*(pred<0.5)).item()
            pause = 1
    correct /= size
    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, optimizer)
    test_loop(test_dataloader, model)
pause = 1