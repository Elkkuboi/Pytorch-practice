# Import pytorch and matplotlib
import torch
from torch import nn
import matplotlib.pyplot as plt

# Check pytorch version
torch.__version__

# Check and setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Run in console for more information
# nvidia-smi



# Create some data using the linear regression formula
weight = 0.2
bias = 22.4

# Create range values
start = 0
end = 10
step = 0.01

# Create X and y (feature and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias



# Split the data
train_split = int(0.8 * len(X))

# shuffle the indexes
shuffled_indices = torch.randperm(len(X))

# selects indexes
train_indices = shuffled_indices[:train_split]
test_indices = shuffled_indices[train_split:]

# pick indexes
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

len(X_train), len(y_test)



# Plot the data
def plot_predictions(train_data = X_train,
                     train_labels = y_train,
                     test_data = X_test,
                     test_labels = y_test,
                     predictions = None):

    """
    Plots training data, test data and compares predictions.
    """

    plt.figure(figsize=(10,7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label = "Training data")

    # Plot testing data in green
    plt.scatter(test_data, test_labels, c="y", s=4, label = "Test data")

    # Check for predictions
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", label = "Predictions")
    
    # Show legend
    plt.legend(prop={"size":14})

plot_predictions(X_train, y_train, X_test, y_test)
plt.show()



# Changing the data to gpu
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)



class LinearRegressionModel(nn.Module): # <- almost everything in python inherits from nn.Module
    def __init__(self):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(1,
                                                requires_grad = True,
                                                dtype=torch.float32))
        self.bias = nn.Parameter(torch.rand(1,
                                                requires_grad = True,
                                                dtype=torch.float32))
        # YOU COULD ASWELL DO
        # self.linear_layer = nn.Linear(in_features=1, out_features=1)
        # But I like to do it by hand for now
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
        # YOU COULD AS WELL
        # return self.linear_layer(x)




# Lets create a model and transfer it to gpu
torch.manual_seed(0)

model_1 = LinearRegressionModel() # At this point we generated the seed because __init__ is called
model_1.to(device)

print(model_1.state_dict())


# Let's setup a loss function
loss_fn = nn.MSELoss() # squared error

# Let's setup an optimizer
optimizer = torch.optim.Adam(params = model_1.parameters(), lr = 0.01)




# FOR THE TRAINING LOOP

# Looping trought the data
epochs = 5000

# Store values
epoch_count = []
loss_values = []
test_loss_values = []

for epoch in range(epochs):

    # Train mode
    model_1.train()

    # Forward pass
    y_preds = model_1(X_train)

    # Calculate the loss
    loss = loss_fn(y_preds, y_train)

    # Zero the gradients (because pytorch for some reason accumulates them otherwice)
    optimizer.zero_grad()

    # Backward pass eli back propagation
    loss.backward()

    # Update the values based on lr and gradients
    optimizer.step()

    # Testing time
    model_1.eval()

    if epoch % 10 == 0:
        with torch.inference_mode():
            preds = model_1(X_test)
            train_loss = loss
            test_loss = loss_fn(preds, y_test)
        
        epoch_count.append(epoch)
        loss_values.append(train_loss.item())
        test_loss_values.append(test_loss.item())

        print(f"Epoch: {epoch}, Train loss: {train_loss}, Test loss: {test_loss}")

# Lets visualize test and train loss:
plt.plot(epoch_count, loss_values, label = "Train loss")
plt.plot(epoch_count, test_loss_values, label = "test loss")
plt.title("Training and test loss curve")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()

