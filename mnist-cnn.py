import torch
import torch.nn as nn
from torch.optim import SGD
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys
import random
from tabulate import tabulate


device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

root_directory = os.getcwd()

# Default values
epoch = 20
learningRate = 0.01
# Check for command-line arguments
if len(sys.argv) > 1:
    for arg in sys.argv[1:]:
        key, value = arg.split('=')
        if key == '--epoch':
            epoch = int(value)
        elif key == '--learningRate':
            learning_rate = float(value);
else:
    print("\nNo arguments given proceeding with default values.....")


runFolder = os.path.join(root_directory, "run")

# Check if the folder exists, if not, create it

if not os.path.exists(runFolder):
    os.mkdir(runFolder)

def PlotTrainData(epoch_data, loss_data, epochCount):
    # plt.plot(epoch_data, loss_data)
    # plt.xlabel('Epoch Number')
    # plt.ylabel('Cross Entropy')
    # plt.title('Cross Entropy (per batch)')
    epoch_data_avgd = epoch_data.reshape(epochCount,-1).mean(axis=1)
    loss_data_avgd = loss_data.reshape(epochCount,-1).mean(axis=1)
    plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
    plt.xlabel('Epoch Number')
    plt.ylabel('Cross Entropy')
    plt.title('Cross Entropy (avgd per epoch)')
    plt.savefig(f'{runFolder}/train-loss-stats.png');


# Dataset Object
class CTDataset(Dataset):
    def __init__(self, filepath):
        self.x, self.y = torch.load(filepath)
        self.x = self.x / 255.
        self.y = F.one_hot(self.y, num_classes=10).to(float)
    def __len__(self): 
        return self.x.shape[0]
    def __getitem__(self, ix): 
        return self.x[ix], self.y[ix]

trainDs = CTDataset("/Users/tamaldas/Documents/Learnnig-PyTorch/mnist-pytorch/MNIST-Dataset/MNIST/processed/training.pt")
testDs = CTDataset("/Users/tamaldas/Documents/Learnnig-PyTorch/mnist-pytorch/MNIST-Dataset/MNIST/processed/test.pt")

# Data Loader Object
train_dl = DataLoader(trainDs, batch_size=5)
test_dl = DataLoader(testDs, batch_size=5)

# Set Loss Function
LossFunc = nn.CrossEntropyLoss()

# Network
class MyMnistCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.Matrix1 = nn.Linear(28**2,100)
        self.Matrix2 = nn.Linear(100,50)
        self.Matrix3 = nn.Linear(50,10)
        self.R = nn.ReLU()
    def forward(self,x):
        x = x.view(-1,28**2)
        x = self.R(self.Matrix1(x))
        x = self.R(self.Matrix2(x))
        x = self.Matrix3(x)
        return x.squeeze();

# instance of network
cnn = MyMnistCNN().to(device=device)

xs, ys = trainDs[0:4]


# making first prediction and calculating the loss
pred = cnn(xs.to(device=device))
ys = ys.type(torch.float32)

initialLoss = LossFunc(pred, ys.to(device=device))


# Training Loop
def train_model(dl, f, n_epochs=20, lr=0.01):
    # Optimization
    opt = SGD(f.parameters(), lr)
    L = nn.CrossEntropyLoss()

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        N = len(dl)
        for i, (x, y) in enumerate(dl):
            # Update the weights of the network
            if device == "mps":
                y = y.type(torch.float32)
                y.to(device)
                x.to(device)
                x = x.type(torch.float32)
            elif device == "cuda":
                y.to(device)
                x.to(device)
            opt.zero_grad()
            loss_value = L(f(x), y) 
            loss_value.backward() 
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
        print(f'Epoch {epoch}: Loss: {loss_value}')
    return np.array(epochs), np.array(losses);


# starting training
print("----------------------------")
print("Starting training....")
print(f"Number of Epochs: {epoch}")
print(f"Learning Rate: {learningRate}")
print("----------------------------\n")
epoch_data, loss_data = train_model(train_dl, cnn, epoch, lr= learningRate)

PlotTrainData(epoch_data=epoch_data, loss_data=loss_data, epochCount= epoch)
print("----------------------------")
print("Training finished...")
print("Saving model...")
print("----------------------------")

# saving the model
MODEL_PATH = Path(f'{runFolder}/models/')
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "mnist-cnn.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

torch.save(cnn.state_dict(), MODEL_SAVE_PATH)
print(f"\nModel Saved at: {MODEL_SAVE_PATH}")
print("----------------------------\n")


# Evaluate the model
correct = 0
total = 0
xs, ys = testDs[:]
original_labels = torch.argmax(ys, dim=1)
finalPred = cnn(xs).argmax(axis=1)

fig, ax = plt.subplots(10,4,figsize=(10,15))
for i in range(40):
    randIndex = random.randint(1, xs.shape[0])
    plt.subplot(10,4,i+1)
    plt.imshow(xs[randIndex])
    plt.title(f'Predicted Digit: {finalPred[randIndex]}')
    if finalPred[randIndex] == original_labels[randIndex]:
        correct += 1;
    total += 1
    fig.tight_layout()
plt.savefig(f'{runFolder}/test-results.png')

accuracy = 100 * correct / total

totalSumary = [
    ["Number of epochs", epoch],
    ["Learning Rate", learningRate],
    ["Accuracy", accuracy],
    ["Results saved at", MODEL_SAVE_PATH]
]
print(tabulate(totalSumary, tablefmt="grid"))


