import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from timm.utils import ModelEmaV3 #pip install timm 

from pathlib import Path
import matplotlib.pyplot as plt

import model

# consts
BATCH_SIZE = 64
NUM_TIME_STEPS = 1000 
MODEL_DIR_PATH = Path("models")
MODEL_PATH = MODEL_DIR_PATH / "audio_diffusion_1.0.pth"

# save and load model
def save_model(model):
    MODEL_DIR_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=model, f=MODEL_PATH)
    



# training
def train(batch_size: int=BATCH_SIZE, num_time_steps: int=NUM_TIME_STEPS):
    torch.seed(67)

    #train_dataset = datasets.MNIST(root='./data', train=True, download=False,transform=transforms.ToTensor())
    train_loader = DataLoader()

    scheduler = model.DDPM_Scheduler(num_time_steps=num_time_steps)

    channels = [64, 128, 256, 512, 512, 384]
    attention = [False, True, False, False, False, True]
    model = model.UNET(channels, attention).to(model.device)

    lr=0.00002
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='mean')

    decay = 0.9999
    ema = ModelEmaV3(model, decay=decay)

    epochs = 15
    for i in range(epochs):
        total_loss = 0
        for _, x in enumerate(train_loader, 0):
            x = F.pad(x.to(model.device), (2,2,2,2))
            
            t = torch.randint(0,num_time_steps,(batch_size,))
            e = torch.randn_like(x, requires_grad=False)
            a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()

            x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)

            output = model(x, t)

            optimizer.zero_grad()
            loss = loss_fn(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            ema.update(model)
        print(f'Epoch {i+1} | Loss {total_loss / (60000/batch_size):.5f}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict()
    }
    save_model(checkpoint, 'checkpoints/ddpm_checkpoint')
    # show