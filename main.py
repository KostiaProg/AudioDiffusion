import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WIDTH = 32
HEIGHT = 16
BATCH_SIZE = 64
PIXELS = WIDTH*HEIGHT # 1600x1200 - 32x16

MODELS_PATH = Path("models")
DISCRIMINATOR_PATH = MODELS_PATH / "image_discriminator.1.1.pth"
GENERATOR_PATH = MODELS_PATH / "image_generator.1.1.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

COLOR_CHANNELS = 3
FEATURE_MULTIPLIER = 64
HIDDEN_DIM = 150
LABEL_SMOOTHING = 0.8 # 1.0 if no

# load data
def load_data() -> DataLoader:
    data_path = Path("/root/.cache/kagglehub/datasets/shyambhu/hands-and-palm-images-dataset/versions/2")
    transform = transforms.Compose([
        transforms.Resize((HEIGHT, WIDTH)),
        transforms.RandomVerticalFlip(1.0), # because training dataset is flipped
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    torch.manual_seed(67)
    data = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader


# models
def weights_init(m): # initialize random weights
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Discriminator(nn.Module):
    def __init__(self, feature_multiplier=FEATURE_MULTIPLIER, color_channels=COLOR_CHANNELS):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.Conv2d(color_channels, feature_multiplier, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_multiplier),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(feature_multiplier, feature_multiplier*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*2),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(feature_multiplier*2, feature_multiplier*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*4),
            nn.LeakyReLU(0.2, True),
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(feature_multiplier*4, feature_multiplier*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*8),
            nn.LeakyReLU(0.2, True),
        )
        self.out = nn.Conv2d(feature_multiplier*8, 1, (1, 2), 1, 0, bias=False)

        self.seq = nn.Sequential(
            self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.out
        )

    def forward(self, x):
        return self.seq(x)

class Generator(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM, feature_multiplier=FEATURE_MULTIPLIER, color_channels=COLOR_CHANNELS):
        super().__init__()

        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, feature_multiplier*8, (1, 2), 1, 0, bias=False),
            nn.BatchNorm2d(feature_multiplier*8),
            nn.ReLU(True),
        )
        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*8, feature_multiplier*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*4),
            nn.ReLU(True),
        )
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*4, feature_multiplier*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier*2),
            nn.ReLU(True),
        )
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier*2, feature_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_multiplier),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(feature_multiplier, color_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.seq = nn.Sequential(
            self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.out
        )

    def forward(self, x):
        return self.seq(x)


# training fns
def train_discriminator(discriminator, optimizer, loss_fn, real_data, label_smoothing, fake_data, batch_size):
    # real data
    pred_real = discriminator(real_data).view(-1)
    loss_real = loss_fn(pred_real, torch.full_like(pred_real, label_smoothing))

    # fake data
    pred_fake = discriminator(fake_data).view(-1)
    loss_fake = loss_fn(pred_fake, torch.zeros_like(pred_fake))

    # accuracy monitoring
    d_real_acc = (torch.sigmoid(pred_real) > 0.5).float().mean().item()
    d_fake_acc = (torch.sigmoid(pred_fake) < 0.5).float().mean().item()
    print(f"  D_real_acc: {d_real_acc:.3f} | D_fake_acc: {d_fake_acc:.3f} | D_loss: {d_loss:.4f}")

    optimizer.zero_grad()
    (loss_real + loss_fake).backward()
    optimizer.step()

    return (loss_real + loss_fake).item()

def train_generator(discriminator, optimizer, loss_fn, fake_data, batch_size):
    pred = discriminator(fake_data).view(-1)
    loss = loss_fn(pred, torch.ones_like(pred))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def train(epochs, dataloader, discriminator, generator, d_optim, g_optim, loss_fn, n_critic):
    results = { # for plotting
        "d_loss": [],
        "g_loss": [],
    }

    # training
    for epoch in range(epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for n_batch, data in enumerate(dataloader, 0):
            # discriminator
            real_data = data[0].to(device)
            real_data = real_data + torch.randn_like(real_data) * 0.02 # add noise
            real_data = torch.clamp(real_data, -1.0, 1.0)

            batch_size = real_data.size(0)

            for _ in range(n_critic):
                noise_d = torch.randn(batch_size, HIDDEN_DIM, 1, 1, device=device)
                fake_data = generator(noise_d).detach() # detach, so gradients are not calculated for generator
                d_loss = train_discriminator(discriminator, d_optim, loss_fn, real_data, LABEL_SMOOTHING, fake_data, batch_size)
                epoch_d_loss += d_loss

            # generator
            noise_g = torch.randn(batch_size, HIDDEN_DIM, 1, 1, device=device)
            fake_data = generator(noise_g).to(device)
            g_loss = train_generator(discriminator, g_optim, loss_fn, fake_data, batch_size)
            epoch_g_loss += g_loss;

            num_batches += 1

        # save loss and print loss
        avg_d = epoch_d_loss / num_batches
        avg_g = epoch_g_loss / num_batches
        results["d_loss"].append(float(avg_d))
        results["g_loss"].append(float(avg_g))

        print(f"Discriminator loss: {avg_d} / Generator loss: {avg_g}")
        print("")

    # plot curves
    plt.plot(range(len(results["d_loss"])), results["d_loss"], label="Discriminator loss")
    plt.plot(range(len(results["g_loss"])), results["g_loss"], label="Generator loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # save model
    save_models(discriminator, generator, DISCRIMINATOR_PATH, GENERATOR_PATH)


# create and save models
def save_models(discriminator: nn.Module, generator: nn.Module, discriminator_path: Path, generator_path: Path):
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(obj=discriminator.state_dict(), f=discriminator_path)
    torch.save(obj=generator.state_dict(), f=generator_path)

def create_model():
    dataloader = load_data()

    # models
    discriminator = Discriminator().to(device)
    generator = Generator().to(device)

    # random weights
    discriminator.apply(weights_init)
    generator.apply(weights_init)

    # optim and loss
    d_lr = 0.00005
    g_lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999
    d_optim = optim.Adam(discriminator.parameters(), lr=d_lr, betas=(beta1, beta2))
    g_optim = optim.Adam(generator.parameters(), lr=g_lr, betas=(beta1, beta2))

    loss_fn = nn.BCEWithLogitsLoss()

    epochs = 80
    n_critic = 2
    train(epochs=epochs, dataloader=dataloader, discriminator=discriminator, generator=generator, d_optim=d_optim, g_optim=g_optim, loss_fn=loss_fn, n_critic=n_critic)

def get_saved_model() -> nn.Module:
    loaded_generator = Generator()

    if not GENERATOR_PATH.is_file():
        print("First have to train model")
        create_model()

    loaded_generator.load_state_dict(torch.load(f=GENERATOR_PATH, map_location=torch.device(device)))

    return loaded_generator


# TOO CALL
def generate_image():
    generator = get_saved_model().to(device)
    generator.eval()

    noise = torch.randn(1, HIDDEN_DIM, 1, 1, device=device)
    with torch.inference_mode():
      image = generator(noise)

    image = image.detach().cpu()
    image = image[0].permute(1, 2, 0).numpy() # [64, 128, 3]

    # Denormalize from [-1, 1] back to [0, 1]
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)

    plt.imshow(image)
    plt.title("Generated image")
    plt.axis("off")
    plt.show()

generate_image()