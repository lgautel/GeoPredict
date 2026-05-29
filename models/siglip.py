import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class Encoder1DBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MlpBlock(dim, mlp_dim)
    
    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x, _ = self.mha(x, x, x, need_weights=False)
        x = residual + x
        
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


class SigLIP(nn.Module):
    def __init__(
        self, 
        num_classes=2048,
        patch_size=14,
        width=1152,
        depth=27,
        mlp_dim=4304,
        num_heads=16,
        img_size=224,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.img_size = img_size
        
        self.patch_embed = nn.Conv2d(
            3, self.width, kernel_size=self.patch_size, stride=self.patch_size
        )
        
        h = w = self.img_size // self.patch_size
        self.pos_embedding = nn.Parameter(torch.zeros(1, h * w, self.width))
        nn.init.normal_(self.pos_embedding, std=1/math.sqrt(self.width))
        
        self.blocks = nn.ModuleList([
            Encoder1DBlock(self.width, self.num_heads, self.mlp_dim)
            for _ in range(self.depth)
        ])
        
        self.norm = nn.LayerNorm(self.width)
        
        self.projection = nn.Linear(self.width, self.num_classes)
    
    def forward(self, x, pool_type="none"):
        x = self.patch_embed(x)
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).permute(0, 2, 1)  # [n, h*w, c]
        x = x + self.pos_embedding

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        if pool_type == "none":
            pass
        elif pool_type == "gap":
            x = x.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown pool type: '{pool_type}'")
        
        x = self.projection(x)
        
        return x  # [n, h*w, num_classes]
