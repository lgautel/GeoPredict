import torch.nn as nn
import torch.nn.functional as F


class VoxelDecoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512):
        super().__init__()
        self.feature_proj = nn.Linear(input_dim, hidden_dim)
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim
        
        self.upconv1 = nn.ConvTranspose3d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)  # 16x16x10
        self.norm1 = nn.BatchNorm3d(hidden_dim)
        
        self.upconv2 = nn.ConvTranspose3d(hidden_dim, 256, kernel_size=4, stride=2, padding=1)         # 32x32x20
        self.norm2 = nn.BatchNorm3d(256)
        
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=3, stride=1, padding=1)          # 32x32x20
        self.norm3 = nn.BatchNorm3d(128)
        
        self.final_conv = nn.Conv3d(128, 14*4, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # x: [B, 320, 2048]
        B = x.shape[0]
        
        x = self.feature_proj(x)  # [B, 320, 512]
        x = self.feature_norm(x)  # Add LayerNorm after feature projection
        x = x.reshape(B, 8, 8, 5, self.hidden_dim)  # [B, X, Y, Z, 512]
        x = x.permute(0, 4, 1, 2, 3)  # [B, 512, X, Y, Z], [B, 512, 8, 8, 5]

        x = self.upconv1(x)  # [B, 512, 16, 16, 10]
        x = self.norm1(x)    # Add BatchNorm3d
        x = F.relu(x)
        
        x = self.upconv2(x)  # [B, 256, 32, 32, 20]
        x = self.norm2(x)    # Add BatchNorm3d
        x = F.relu(x)
        
        x = self.upconv3(x)  # [B, 128, 32, 32, 20]
        x = self.norm3(x)    # Add BatchNorm3d
        x = F.relu(x)
        
        x = F.interpolate(x, size=(40, 40, 25), mode='trilinear', align_corners=False)
        save_features = x  # [B, 128, 40, 40, 25]
        
        x = self.final_conv(x)  # [B, 14*4, 40, 40, 25]
        
        return x.float(), save_features
