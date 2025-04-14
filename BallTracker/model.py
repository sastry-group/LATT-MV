import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2DBlock(nn.Module):
    """ Conv2D + BN + ReLU """
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
class Double2DConvWithSkip(nn.Module):
    """ Conv2DBlockWithSKip x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConvWithSkip, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x) + x
        return x
    
class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TrackNet, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                       # (N,   64,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   64,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,  128,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,  128,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  256,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  256,   36,    64)
        x = self.bottleneck(x)                                          # (N,  512,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  768,   72,   128)
        x = self.up_block_1(x)                                          # (N,  256,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  384,  144,   256)
        x = self.up_block_2(x)                                          # (N,  128,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,  192,  288,   512)
        x = self.up_block_3(x)                                          # (N,   64,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x

    
class Conv1DBlock(nn.Module):
    """ Conv1D + LeakyReLU"""
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Double1DConv(nn.Module):
    """ Conv1DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double1DConv, self).__init__()
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2)                                   # (N,   L,   3)
        x = x.permute(0, 2, 1)                                         # (N,   3,   L)
        x1 = self.down_1(x)                                            # (N,  16,   L)
        x2 = self.down_2(x1)                                           # (N,  32,   L)
        x3 = self.down_3(x2)                                           # (N,  64,   L)
        x = self.buttleneck(x3)                                        # (N,  256,  L)
        x = torch.cat([x, x3], dim=1)                                  # (N,  384,  L)
        x = self.up_1(x)                                               # (N,  128,  L)
        x = torch.cat([x, x2], dim=1)                                  # (N,  192,  L)
        x = self.up_2(x)                                               # (N,   64,  L)
        x = torch.cat([x, x1], dim=1)                                  # (N,   96,  L)
        x = self.up_3(x)                                               # (N,   32,  L)
        x = self.predictor(x)                                          # (N,   2,   L)
        x = self.sigmoid(x)                                            # (N,   2,   L)
        x = x.permute(0, 2, 1)                                         # (N,   L,   2)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.activation(out)
        # out = out + residual
        # out = self.activation(out)
        return out

class KeypointNet(nn.Module):
    def __init__(self, seq_len, num_classes, hidden_dim=32, hidden_layers=1, dropout_prob=0.25):
        super(KeypointNet, self).__init__()
        self.in_dim  = seq_len * 2
        self.out_dim = num_classes
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        self.fc_in = nn.Linear(self.in_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_prob)
        
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(hidden_layers)]
        )
        
        self.fc_out = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x):
        if len(x.size()) == 2:
            batch_size = len(x)
        else:
            batch_size = 1
        x = self.fc_in(x)
        x = self.activation(x)
        x = self.dropout(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.fc_out(x)
        return x
    
# class KeypointNet(nn.Module):
#     def __init__(self, in_channels=4, num_classes=5):
#         super(KeypointNet, self).__init__()
        
#         # Convolutional layers to featurize each 512x288x4 image
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
#         # Transformer encoder
#         self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=16)
#         self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=2)
        
#         # Fully connected layer for classification
#         self.fc = nn.Linear(512, num_classes)
#         self.softmax = nn.Softmax(dim=-1)
    
#     def forward(self, x):
#         # Ensure the input tensor is of type float
#         x = x.float()
        
#         # Reshape to (sequence_length * batch_size, channels, height, width)
#         seq_len, batch_size, channels, height, width = x.size()
#         x = x.reshape(seq_len * batch_size, channels, height, width)
        
#         # Apply convolutional layers with ReLU and max pooling
#         x = F.relu(self.conv1(x))
#         x = F.max_pool2d(x, kernel_size=4, stride=4)
        
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d(x, kernel_size=4, stride=4)
        
#         x = F.relu(self.conv3(x))
#         x = F.max_pool2d(x, kernel_size=4, stride=4)
        
#         x = F.relu(self.conv4(x))
#         x = F.max_pool2d(x, kernel_size=4, stride=4)

#         # Flatten the feature map into a sequence token
#         x = x.reshape(seq_len, batch_size, -1)
        
#         # Apply the transformer encoder
#         x = self.transformer_encoder(x)
        
#         # Average pooling and classification
#         x = self.fc(x).squeeze()
        
#         return x