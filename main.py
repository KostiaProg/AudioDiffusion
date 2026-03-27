import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader

from pathlib import Path
import matplotlib.pyplot as plt
import math

from IPython.display import Audio, display

# consts
EMBEDDING_DIM = 128
NUM_GROUPS = 8
NUM_HEADS = 8

NUM_DOWN_LAYERS = 2
NUM_MID_LAYERS = 2
NUM_UP_LAYERS = 2

TIME_STEPS = 1000
SAMPLING_FACTOR = 4

WAVE_LENGTH = 16384
SAMPLE_RATE = 16000

device = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
MODEL_DIR_PATH = Path("models")
MODEL_PATH = MODEL_DIR_PATH / "audio_diffusion_1.0.pth"

# embedding
class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim: int = EMBEDDING_DIM):
        super().__init__()

        omega = 1.0 / (10000 ** (torch.arange(0, embed_dim, 2).float() / embed_dim))
        t = torch.arange(time_steps).unsqueeze(1).float()

        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(t * omega) # sin/cos(t*w)
        embeddings[:, 1::2] = torch.cos(t * omega) # odd
        self.register_buffer('embeddings', embeddings)

    def forward(self, t):
        return self.embeddings[t].to(device).unsqueeze(-1)
    
class TimeEmbedding(nn.Module):
    def __init__(self, out_channel: int,  t_emb_dim: int=EMBEDDING_DIM):
        super().__init__()
        
        self.te_block = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(t_emb_dim, out_channel),
            nn.SiLU(), 
            nn.Linear(out_channel, out_channel)
        )
        
    def forward(self, t_emb):
        return self.te_block(t_emb.transpose(1, 2)).transpose(1, 2)


# custom layers
class NormActConv(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, num_groups:int = NUM_GROUPS, padding_dilation: int=1):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding_dilation, dilation=padding_dilation)
        )
        
    def forward(self, x):
        return self.conv(x)

class Attention(nn.Module):
    def __init__(self, num_channels: int, num_groups: int=NUM_GROUPS, num_heads: int=NUM_HEADS):
        super().__init__()
        
        self.g_norm = nn.GroupNorm(num_groups, num_channels)
        self.attention = nn.MultiheadAttention(num_channels, num_heads, batch_first=True)
        
    def forward(self, x):
        res_x = x

        x = self.g_norm(x).transpose(1, 2) # [B, C, L] -> [B, L, C]
        x, _ = self.attention(x, x, x)
        return x.transpose(1, 2) + res_x

# UNET
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int=EMBEDDING_DIM, k: int=SAMPLING_FACTOR, downsample: bool = True):
        super().__init__()

        self.conv1 = NormActConv(in_channels, out_channels)
        self.conv2 = NormActConv(out_channels, out_channels, padding_dilation=2)
        self.te_block = TimeEmbedding(out_channels, t_emb_dim)
        self.attention_block = Attention(out_channels)

        self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

        self.down = nn.Conv1d(out_channels, out_channels, kernel_size=k*2, stride=k, padding=k//2) if downsample else nn.Identity()

    def forward(self, x, t_emb):
        res = self.res_proj(x)

        x = self.conv1(x)
        x = x + self.te_block(t_emb)
        x = self.conv2(x)
        x = x + res

        x = x + self.attention_block(x)
    
        return x, self.down(x) # return x so we add it as a skip connection during upsampling
    
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int=EMBEDDING_DIM, k: int=SAMPLING_FACTOR, upsample: bool = True):
        super().__init__()
        
        self.conv1 = NormActConv(in_channels, out_channels)
        self.conv2 = NormActConv(out_channels, out_channels, padding_dilation=2)
        self.te_block = TimeEmbedding(out_channels, t_emb_dim)
        self.attention_block = Attention(out_channels)

        self.res_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        self.up = nn.Sequential(
                nn.Upsample(scale_factor=k, mode='nearest'),
                nn.Conv1d(in_channels, in_channels // 2, kernel_size=3, padding=1)
            ) if upsample else nn.Identity()

    def forward(self, x, down_out, t_emb):
        # Upsampling
        x = self.up(x)
        if x.shape[-1] != down_out.shape[-1]: # Ensure temporal length matches the skip connection exactly
            x = F.interpolate(x, size=down_out.shape[-1], mode='linear', align_corners=False)
        x = torch.cat([x, down_out], dim=1) # skip connection

        res = self.res_proj(x)

        x = self.conv1(x)
        x = x + self.te_block(t_emb)
        x = self.conv2(x)
        x = x + res
            
        return self.attention_block(x)


class UNET(nn.Module):
    def __init__(self,
                 t_emb_dim: int = EMBEDDING_DIM,
                 t: int = TIME_STEPS,
                ):
        super().__init__()
        
        self.sin_embedding = SinusoidalEmbeddings(t, t_emb_dim)

        # Initial Convolution
        self.init_conv = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        
        # Down Path
        self.down1 = DownBlock(64, 128, EMBEDDING_DIM)
        self.down2 = DownBlock(128, 256, EMBEDDING_DIM)
        
        # Mid Block
        self.mid_conv = NormActConv(256, 256)
        
        # Up Path
        self.up1 = UpBlock(256, 128, EMBEDDING_DIM)
        self.up2 = UpBlock(128, 64, EMBEDDING_DIM)
        
        self.final_conv = nn.Conv1d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        t_emb = self.sin_embedding(t)
        
        x = self.init_conv(x)
        skip1, x = self.down1(x, t_emb)
        skip2, x = self.down2(x, t_emb)
        
        x = self.mid_conv(x)
        
        x = self.up1(x, skip2, t_emb)
        x = self.up2(x, skip1, t_emb)
        
        return self.final_conv(x)


# add and remove noise
class DiffusionProcess:
    def __init__(self, beta_start: float=1e-4, beta_end: float=0.02, t: int=TIME_STEPS):

        self.beta = torch.linspace(beta_start, beta_end, t).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
    def add_noise(self, x, noise, t):
        sqrt_alpha_bar = self.alpha_bar[t].sqrt()[:, None, None]
        sqrt_one_minus_bar = (1. - self.alpha_bar[t]).sqrt()[:, None, None]

        return sqrt_alpha_bar * x + sqrt_one_minus_bar * noise
    
    def remove_noise(self, xt, noise_pred, t):
        a = self.alpha[t][:, None, None]
        a_bar = self.alpha_bar[t][:, None, None]
        b = self.beta[t][:, None, None]
        
        z = torch.randn_like(xt) if t[0] > 0 else 0 # No noise added at the final step
            
        mean = (1/a.sqrt()) * (xt - ( (1-a)/((1-a_bar).sqrt()) ) * noise_pred)
        sigma = b.sqrt() # Simplest variance schedule
        return mean + sigma * z


# TRAINING CODE
# save and load model
def save_model(model):
    MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model, f=MODEL_PATH)
    
def get_saved_model():
    loaded_generator = UNET()

    if not MODEL_PATH.is_file():
        print("First have to train model")
        train()

    loaded_generator.load_state_dict(torch.load(f=MODEL_PATH, map_location=torch.device(device)))
    return loaded_generator

# training
def train():
    torch.seed(67)

    #train_dataset = datasets.MNIST(root='./data', train=True, download=False,transform=transforms.ToTensor())
    train_loader = DataLoader()

    scheduler = model.DDPM_Scheduler(num_time_steps=TIME_STEPS)
    model = UNET().to(device)

    lr=0.00002
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dfp = DiffusionProcess()

    best_eval_loss = float('inf')
    epochs = 50
    losses = []
    for epoch in range(epochs):
        n_batches = 0
        total_loss = 0

        model.train()
        for _, x in enumerate(train_loader, 0):
            x = x.to(device)
            
            # Generate noise and timestamps
            noise = torch.randn_like(x).to(device)
            t = torch.randint(0, TIME_STEPS, (x.shape[0],)).to(device)
            
            # Add noise to the images using Forward Process
            noisy_imgs = dfp.add_noise(x, noise, t)
            
            # Avoid Gradient Accumulation
            optimizer.zero_grad()
            
            # Predict noise using U-net Model
            noise_pred = model(noisy_imgs, t)
            
            # Calculate Loss
            loss = loss_fn(noise_pred, noise)
            total_loss += loss.item()
            
            # Backprop + Update model params
            loss.backward()
            optimizer.step()

            n_batches += 1
        
        total_loss /= n_batches
        losses.append(total_loss)

        print(f'Epoch: {epoch} | loss: {total_loss}')
        if total_loss < best_eval_loss:
            best_eval_loss = total_loss
            save_model(model)

    # plot loss curves
    plt.plot(range(len(losses)), losses)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()


# TOO CALL
def generate():
    drp = DiffusionProcess()
    model = get_saved_model()

    # noise
    xt = torch.randn(1, 1, WAVE_LENGTH).to(device)

    model.eval()
    with torch.inference_mode():
        for t in reversed(range(TIME_STEPS)):
            t_tensor = torch.as_tensor([t], device=device).long()
            noise_pred = model(xt, t_tensor)
            xt, x0 = drp.remove_noise(xt, noise_pred, t_tensor)

    # postprocess
    xt = torch.clamp(xt, -1., 1.).detach().cpu()

    audio_waveform = xt[0, 0].numpy()
    display(Audio(audio_waveform, rate=SAMPLE_RATE))