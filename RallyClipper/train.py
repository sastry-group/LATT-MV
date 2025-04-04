import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from dataset import VideoFrameDataset
from cnn import CNNVideoFrameClassifier
from constants import *

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_frames, batch_labels in pbar:
            batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_frames)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                              'acc': f'{100. * train_correct / train_total:.2f}%'})

        epoch_loss = train_loss / len(train_loader)
        epoch_acc = 100. * train_correct / train_total
        epoch_time = time.time() - start_time

        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print(f'Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, Time: {epoch_time:.2f}s')

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for batch_frames, batch_labels in pbar:
                batch_frames, batch_labels = batch_frames.to(device), batch_labels.to(device)
                
                outputs = model(batch_frames)
                loss = criterion(outputs, batch_labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_labels.size(0)
                test_correct += (predicted == batch_labels).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                  'acc': f'{100. * test_correct / test_total:.2f}%'})

        test_loss = test_loss / len(test_loader)
        test_acc = 100. * test_correct / test_total

        # Calculate additional metrics
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')

        print(f'Testing  - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

        # Save checkpoint at the end of every epoch
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
        }
        torch.save(checkpoint, f'ckpts/checkpoint_epoch_{epoch+1}.pth')
        print(f'Checkpoint saved for epoch {epoch+1}')

        # Save the best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'ckpts/best_model.pth')
            print(f'New best model saved with accuracy: {best_accuracy:.2f}%')

        print('-' * 60)

# Initalize the model and data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNVideoFrameClassifier(width=WIDTH, height=HEIGHT).to(device)
train_dataset = VideoFrameDataset(root_dir='train', n_samples=5000, frame_width=WIDTH, frame_height=HEIGHT)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataset = VideoFrameDataset(root_dir='test',  n_samples=1000, frame_width=WIDTH, frame_height=HEIGHT)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)

train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs=20, device=device)
