# In this file we'll load the fashion mnist dataset from the ready folder
# we'll gather the data, define dataloader etc, create a linear model, train and test, improve the model, test etc

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import torchinfo, torchmetrics
from torch import nn
from tqdm.auto import tqdm
from helperfunctions import accuracy_fn


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

# Device code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)



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


# Let's see how it looks
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
print("More images... looks difficult!")
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


# Let's prepare the dataloader
# Dataloader turns our data into python iterable and to minibatches

# Let's setup the batch size, 32 is recommended
BATCH_SIZE = 512
train_dataloader = DataLoader(dataset = train_data,
                                batch_size = BATCH_SIZE,
                                shuffle = True)

test_dataloader = DataLoader(dataset = test_data,
                            batch_size = BATCH_SIZE)

# Check out train loader
print("Check out train loader")
train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(train_features_batch.shape, train_labels_batch.shape)
print("\n")

# Checkout yet another image with dataloader now
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.axis(False)
plt.title(idx_to_class[label.item()])
print(f" Image shape: {img.shape}")
print(f" Image label: {label}")
plt.show()


# Time to crank out the first model, but first let's flatten the data shape to one long vector
flatten_model = nn.Flatten()

x = train_features_batch[0]
print(f"Shape before flattening: {x.shape}")
output = flatten_model(x)
print(f"Shape after flattening{output.shape}")
print("\n")

# First simple Linear model
# Creating a model
class FashionMNISTModelV0(nn.Module):
    def __init__(self, 
                input_shape: int,
                hidden_units: int,
                output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features = hidden_units),
            nn.Linear(in_features = hidden_units, out_features = output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

model_0 = FashionMNISTModelV0(
    input_shape = 784,
    hidden_units = 16,
    output_shape = 10
).to("cpu")

print(f"Check out the first model: {model_0}")
# Test that it works correctly
dummy_x = torch.rand([1, 1, 28, 28])
print(f"The first test with the model, should output 10 values {model_0(dummy_x)}")

# Time to setup the optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model_0.parameters(),
                             lr = 0.01)

# Create a function for timing our calculations:
from timeit import default_timer as timer
def print_train_time(start: float,
                     end: float,
                     device: torch.device = None):

    """ Prints difference between start and end time"""
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# Training and testing loop
train_time_start_on_cpu = timer()
epochs = 10

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n-----")

    ### Training
    train_loss = 0
    # Add a loop to batches
    for batch, (X, y) in enumerate(train_dataloader):
        model_0.train()
        y_preds = model_0(X)
        loss = loss_fn(y_preds, y)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print(f" Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

    # Training loss
    train_loss /= len(train_dataloader)
    
    
    ### Testing loop
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X, y in test_dataloader:
            test_pred = model_0(X)
            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
    print(f"Train loss {train_loss:.4f} Test loss: {test_loss:.4f} Test acc: {test_acc:.4f}")    

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(start=train_time_start_on_cpu,
                                            end= train_time_end_on_cpu,
                                            device=str(next(model_0.parameters()).device))


# Create a function for evaluating models
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):

    """Returns a dictionary containing the results of model predicting on data_loader"""
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader):
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred=y_pred.argmax(dim = 1))
        
        # Scale loss and acc to find average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}


# Calculate model 0 results on test dataset
model_0_results = eval_model(model = model_0,
                            data_loader = test_dataloader,
                            loss_fn = loss_fn,
                            accuracy_fn=accuracy_fn)

print(f"The results of first model {model_0_results}")
print("We have some room for improvement")
