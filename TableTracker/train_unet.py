import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data_loader import TableTennisDataset
from unet import UNet
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Set random seeds for reproducibility (optional)
torch.manual_seed(42)
np.random.seed(42)

# Create directories for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Instantiate the dataset
dataset = TableTennisDataset(data_root='data/')

# Split the dataset into training and validation sets
train_ratio = 0.95
train_size = int(train_ratio * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Instantiate the model
model = UNet(in_channels=1, out_channels=1)

# Move the model to the appropriate device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50  # Set the number of epochs

# Lists to store training and validation losses
train_losses = []
val_losses = []

best_val_loss = float('inf')  # Initialize best validation loss

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # Use tqdm for progress bar during training
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for images, labels in progress_bar:
        images = images.to(device)  # Shape: (batch_size, 1, H, W)
        labels = labels.to(device)  # Shape: (batch_size, 1, H, W)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        outputs = outputs.squeeze(1)  # Remove channel dimension if necessary
        labels = labels.squeeze(1)    # Remove channel dimension if necessary

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': loss.item()})

    epoch_loss = running_loss / train_size
    train_losses.append(epoch_loss)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            outputs = outputs.squeeze(1)
            labels = labels.squeeze(1)

            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

    val_loss = val_running_loss / val_size
    val_losses.append(val_loss)

    # Print epoch summary
    print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Save the current model
    torch.save(model.state_dict(), f'models/model_epoch_{epoch+1}.pth')

    # Save the best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pth')
        print(f'Best model saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}')

# Plot training and validation loss over epochs
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss Over Epochs')
plt.savefig('plots/loss_plot.png')
plt.show()
