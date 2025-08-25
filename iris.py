import pandas as pd
import numpy as np
import torch
import sklearn
import matplotlib
import torchinfo
import torchmetrics
from sklearn.datasets import load_iris
from torch import nn
from sklearn.model_selection import train_test_split
torch.manual_seed(42)
torch.cuda.manual_seed(42)


# Check PyTorch access (should print out a tensor)
print(torch.randn(3, 3))

# Check for GPU (should return True)
print(torch.cuda.is_available())



# Load and check the data
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(data=X, columns=iris.feature_names)
df['target'] = y

print(df)



# change data to tensors
X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.long)
print(X.shape, y.shape)



# 2. Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.3,
                                                    random_state = 42)

print(X_train[:5], y_test[:5], X_train.shape, y_test.shape)



# 3. move the data to device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
print(X_train[:5], y_test.dtype)



# 4. Creating the model

class multiclassificationV0(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 16):
        '''
        Initiallizes multi-class classification model.
        
        Args:
            input_features (int): Number of input features to the model
            output_feature (int): Number of output features to the model (number of output classes)
            hidden_units (int): Number of hidden units between layers, default = 8

        Return:
            Returns a discrete propability distribution, showing how much the model belies the input is a certain class

        Example:
        '''
        super().__init__()

        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features = hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)
    
# Create an instance of model and send it to target device
model_5 = multiclassificationV0(input_features = 4,
                                output_features = 3,
                                hidden_units = 16).to(device)
print(model_5)



# 5. Choose the optimizer and loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model_5.parameters(),
                             lr = 0.01)



# Training and testing loop
epochs = 650

for epoch in range(epochs):
    
    model_5.train()
    y_preds = model_5(X_train)
    loss = loss_fn(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward() # back propagation
    optimizer.step() # gradient descent
    
    if epoch % 10 == 0:
        model_5.eval()
        with torch.inference_mode():
            test_preds = model_5(X_test)
            test_loss = loss_fn(test_preds, y_test)
            
            y_pred_classes = torch.argmax(test_preds, dim=1)
            correct = (y_pred_classes == y_test).sum().item()
            accuracy = correct / len(y_test) * 100
            
            print(f"Epoch: {epoch} | "
                  f"Loss: {loss:.4f} | "
                  f"Test loss: {test_loss:.4f} | "
                  f"Test accuracy: {accuracy:.2f}%")
            

# We've now correctly predicted the iris
        