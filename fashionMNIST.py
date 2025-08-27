# In this file we'll load the fashion mnist dataset from the ready folder
# we'll gather the data, define dataloader etc, create a linear model, train and test, improve

import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import torchinfo, torchmetrics
from torch import nn

# Torchvision
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor



# Check PyTorch access (should print out a tensor)
print(torch.randn(3, 3))

# Check for GPU (should return True)
print(torch.cuda.is_available())

# Check pytorch version
print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)



# Setup the training data, MAKE SURE YOU HAVE THIS FALE IN SAME DIR AS `fashion-MNIST`
train_data = datasets.FashionMNIST(
    root = "fashion-MNIST",
    train = True,
    download = True,
    transform = ToTensor(),
    target_transform = None
)

test_data = datasets.FashionMNIST(
    root = "fashion-MNIST",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = None
)


# Let's see how it looks!
print(len(test_data), len(train_data))
image, label = train_data[0]
print(image, label)
print()
print(f" Image shape {image.shape}, image label shape {image.shape}")
print(train_data.classes)
print("\n")

# Create some dicts
class_to_idx = train_data.class_to_idx
print(f"class to index dict {class_to_idx}")
idx_to_class = {value: key for key, value in class_to_idx.items()}
print(f"Index to class dict {idx_to_class}")
print("\n")

# Let's see an image!
print("The first image of the dataset")
plt.imshow(image.squeeze(), cmap="gray")
plt.title(idx_to_class[label])
plt.axis(False)
plt.show()
# Let's see a few more
# Plot more images
torch.manual_seed(42)
fig = plt.figure(figsize=(9, 9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size = [1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap = "gray")
    plt.title(idx_to_class[label])
    plt.axis(False)
plt.show()
print("\n")


