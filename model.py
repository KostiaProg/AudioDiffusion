import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange #pip install einops
from typing import List
import math

# consts
EMBEDDING_DIM = 128
NUM_GROUPS = 32
NUM_HEADS = 8
TIME_STEPS = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# embedding
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int = EMBEDDING_DIM):
        super().__init__()
        t = torch.arange(time_steps).unsqueeze(1).float()
        omega = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim)) # e^(-i*ln(10000) / d)

        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(t * omega) # sin/cos(t*w)
        embeddings[:, 1::2] = torch.cos(t * omega) # odd
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings[t].to(device)
        return embeds[:, :, None, None]


# Transformer
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int, dp: float):
        super().__init__()
        
        self.first = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Dropout(p=dp, inplace=True)
        )
        self.second = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, embedding):
        x += embedding[:, :x.shape[1], :, :]
        r = self.second(self.first(x))
        return r + x
    

class Attention(nn.Module):
    def __init__(self, in_channels: int, num_heads:int , dp: float):
        super().__init__()

        self.proj1 = nn.Linear(in_channels, in_channels*3)
        self.proj2 = nn.Linear(in_channels, in_channels)
        self.num_heads = num_heads
        self.dropout_prob = dp

    def forward(self, x):
        h, w = x.shape[2:]

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)

        # Calculate attention
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)

        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)

        return rearrange(x, 'b h w C -> b C h w')


# UNET
class UNETLayer(nn.Module):
    def __init__(self, attention: bool, num_groups: int, num_heads: int, in_channels: int, dp: float):
        super().__init__()

        self.res_block1 = ResidualBlock(in_channels=in_channels, num_groups=num_groups, dp=dp)
        self.res_block2 = ResidualBlock(in_channels=in_channels, num_groups=num_groups, dp=dp)

        self.conv = nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(in_channels=in_channels, num_heads=num_heads, dp=dp)

        self.attention = attention

    def forward(self, x, embeddings):
        x = self.res_block1(x, embeddings)
        if self.attention:
            x = self.attention_layer(x)
        x = self.res_block2(x, embeddings)

        return self.conv(x), x
    
class UNET(nn.Module):
    def __init__(self, channels: List, attention: List, num_groups: int=NUM_GROUPS, num_heads: int=NUM_HEADS, dp: float = 0.1, in_channels: int=1, out_channels: int=1, t: int=TIME_STEPS):
        super().__init__()
        
        self.num_layers = len(channels)
        out_channels = (channels[-1]//2)+channels[0]

        self.embeddings = SinusoidalEmbeddings(time_steps=t, embed_dim=max(channels))
        self.shallow_conv = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)

        self.output = nn.Sequential(
            nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels//2, out_channels, kernel_size=1)
        )

        for i in range(self.num_layers):
            layer = UNETLayer(
                attention=attention[i],
                num_groups=num_groups,
                dropout_prob=dp,
                C=channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t):
        x = self.shallow_conv(x)
        residuals = []

        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i+1}')
            embeddings = self.embeddings(x, t)
            _, r = layer(x, embeddings)
            residuals.append(r)

        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, embeddings)[0], residuals[self.num_layers-i-1]), dim=1)

        return self.output(x)
    

# scheduler
class DDPM_Scheduler(nn.Module):
    def __init__(self, t: int=TIME_STEPS, start: float=0.0001, end: float=0.02):
        super().__init__()

        self.beta = torch.linspace(start, end, t, requires_grad=False)
        self.alpha = torch.cumprod(1 - self.beta, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]