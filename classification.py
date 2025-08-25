# Import pytorch and matplotlib
import torch
from torch import nn
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helperfunctions import plot_predictions, plot_decision_boundary, accuracy_fn



# Check pytorch version
torch.__version__

# Check and setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Run in console for more information
# nvidia-smi



# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples,
                    noise = 0.03,
                    random_state = 42)

# make dataframe of circle data
import pandas as pd
circles = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})

# Visualize
plt.scatter(x = X[:, 0],
            y = X[:, 1],
            c = y,
            cmap = plt.cm.RdYlBu)
plt.show()
# This is a toy dataset, dataset that is small enough to experiment on, but sizeable enough to practise



# Turn the data into tensors and create a train test split
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size = 0.2,
                                                    random_state = 42)

# Move data to Device
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)



# Creating a model
class circleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        # 2. Create 3 nn Linear layers
        self.layer_1 = nn.Linear(in_features = 2, out_features = 8) # input layer into hidden layer
        self.layer_2 = nn.Linear(in_features = 8, out_features = 8) # hidden layer
        self.layer_3 = nn.Linear(in_features = 8, out_features = 1) # output layer
        self.ReLU = nn.ReLU()
    
    # Could as well use 'nn.sequential'

    # Define forward method
    def forward(self, x):
        return self.layer_3(self.ReLU(self.layer_2(self.ReLU(self.layer_1(x)))))

# 4. Instantiate instance and send to gpu
model_3 = circleModelV1().to(device)



# Loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
# BCE with Logits implements a sigmoid function in order to turn pure logits into propabilities, 
# then calculates the loss using logarhitmic scale to punish confident wrong guesses

optimizer = torch.optim.Adam(params = model_3.parameters(),
                             lr = 0.01)



# Creating a training and testing loop
torch.manual_seed(42)
torch.cuda.manual_seed(42)

epochs = 650

for epoch in range(epochs):
    model_3.train()

    y_logits = model_3(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits, y_train)

    optimizer.zero_grad()

    # Back propagation
    loss.backward()

    # Gradient descent
    optimizer.step()

    # testing
    if epoch % 10 == 0:
        model_3.eval()
        with torch.inference_mode():

            test_logits = model_3(X_test).squeeze()
            test_pred = torch.round(torch.sigmoid(test_logits))
            

            test_loss = loss_fn(test_logits,
                                y_test)
            test_acc = accuracy_fn(y_true = y_test,
                                    y_pred = test_pred)

        print(f"Epoch: {epoch}, Loss: {loss.item()}, Test loss: {test_loss}, Test accuracy: {test_acc}")



# Plot decision boundary of the model
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_test)
plt.show()