import torch
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import random
from pathlib import Path

# import the data
trainImgs, trainLabes = torch.load("/Users/tamaldas/Documents/Learnnig-PyTorch/mnist-pytorch/MNIST-Dataset/MNIST/processed/training.pt")
testImgs, testLabels = torch.load("/Users/tamaldas/Documents/Learnnig-PyTorch/mnist-pytorch/MNIST-Dataset/MNIST/processed/test.pt")
imgDim = "28x28"
trainDatasetLength = trainImgs.shape[0]

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"

datasetSummary = [["Total Data", int(trainImgs.shape[0]+testImgs.shape[0])],
                  ["Image size", imgDim],
                  ["Train Images", trainImgs.shape[0]],
                  ["Train Labels", trainLabes.shape[0]],
                  ["Test Images", testImgs.shape[0]],
                  ["Test Labels", testLabels.shape[0]],
                  ]

def VisualizeData(imgData, labelData, len):
    randomIndex = random.randint(1, len)
    plt.imshow(imgData[randomIndex].numpy())
    plt.title(f'Number is {labelData[randomIndex].numpy()}')
    plt.colorbar()
    plt.show();

VisualizeData(trainImgs, trainLabes, trainDatasetLength)
print(tabulate(datasetSummary, tablefmt="grid"))