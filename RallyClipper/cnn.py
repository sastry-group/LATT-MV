import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNVideoFrameClassifier(nn.Module):
    def __init__(self, width, height):
        super(CNNVideoFrameClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(4, 4)
        
        # Calculate the size of the feature maps after convolutions and pooling
        self.feature_size = width // (4**3) * height // (4**3) * 128
        
        self.fc1 = nn.Linear(self.feature_size, 256)
        self.fc2 = nn.Linear(256, 2)  # 2 output classes
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.feature_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
if __name__ == "__main__":
    model = CNNVideoFrameClassifier(width=455, height=256)
    input_tensor = torch.randn(1, 1, 455, 256)  # (batch_size, channels, height, width)
    output = model(input_tensor)
    print(output.shape)  # Should print torch.Size([1, 2])