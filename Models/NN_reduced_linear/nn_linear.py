import sys
sys.path.append('../')
import LoadIntoDf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = LoadIntoDf.load_sqlite_data()
print('data loaded')

# Split data into features and labels
X = df.drop('difficulty', axis=1).values
y = df.difficulty.str.split('/').str[0]


categories = ['4a', '4b', '4c', 
'5a', '5b', '5c', 
'6a', '6a+', '6b', '6b+', '6c', '6c+', 
'7a', '7a+', '7b', '7b+', '7c', '7c+', 
'8a', '8a+', '8b', '8b+', '8c']

encoder = OrdinalEncoder(categories=[categories])

y = encoder.fit_transform(y.values.reshape(-1, 1))

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# Create PyTorch datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        self.layer4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.l2_reg = 0.001  # L2 regularization factor

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout(out)
        out = self.relu(self.layer2(out))
        out = self.dropout(out)
        out = self.relu(self.layer3(out))
        out = self.dropout(out)
        out = self.layer4(out)
        return out
    

input_size = X_train.shape[1]

model = NeuralNet(input_size).to(device)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), weight_decay=model.l2_reg)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


# Training the model
num_epochs = 2
losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
         # Move inputs and labels to GPU
        inputs, labels = inputs.to(device), labels.to(device)
      
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    
    epoch_loss = total_loss / len(train_loader)
    epoch_val_loss = val_loss / len(test_loader)
    losses.append(epoch_loss)
    val_losses.append(epoch_val_loss)
    scheduler.step(epoch_val_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')

    if epoch % 10 == 0:
        model_path = 'model_epoch_{}.pt'.format(epoch)
        torch.save(model.state_dict(), model_path)


plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig(f"nn_learning_curve_epoch_{epoch+1}.pdf")
plt.close()
# Evaluate the model
model.eval()
with torch.no_grad():
    y_true = []
    y_pred = []
    
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    test_mse = mean_squared_error(y_true, y_pred)
    test_r2 = r2_score(y_true, y_pred)
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test R2: {test_r2:.4f}')