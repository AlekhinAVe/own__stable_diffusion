import torch
import numpy as np
from tqdm import tqdm
from ddpm_train import DDPMTrainer
import torch.nn.functional as F
from torch.optim import Adam
from clip import CLIP
from torchvision import transforms
import gc
from simple_unet import SimpleUnet
from diffusion import Diffusion
from pre_diffusion import Network
from simple_unet import SimpleUnet
from own_trainer import DDPMSampler
from sr_unet import UNet

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

#we can train clip/encoder/decoder/diffusion

def train(
        prompt,
        data=None,
        strength=1,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
        batch_size=1,
        epochs=20,
        num_training_steps=1000,
        checkpoint_1=None,
        checkpoint_2=None
):
    if not 0 < strength <= 1:
        raise ValueError("strength must be between 0 and 1")

    if idle_device:
        to_idle = lambda x: x.to(idle_device)
    else:
        to_idle = lambda x: x

    # Initialize random number generator according to the seed specified
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    # SET THE TRAINER
    trainer = DDPMSampler()

    latents_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    encoder = models["encoder"]
    encoder.to(device)

    if checkpoint_1:
        network = Network().to(device)
        network.load_state_dict(checkpoint_1['model_state_dict'], strict=True)
        network.to(device)

    for param in network.parameters():
        param.requires_grad = True

    if checkpoint_2:
        diffusion = Diffusion().to(device)
        diffusion.load_state_dict(checkpoint_2['model_state_dict'], strict=True)
        diffusion.to(device)
    else:
        diffusion = Diffusion().to(device)
        diffusion = models["diffusion"]


    diffusion = UNet().to(device)

    print('Model created')

    for param in diffusion.parameters():
        param.requires_grad = True

    for name, param in diffusion.named_parameters():
        if 'final' in name or 'unet.decoders.10' in name or 'unet.decoders.11' in name:
            param.requires_grad = True

        diffusion_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
        print(diffusion_params)

    #for name, param in diffusion.named_parameters():
        #print(f'Layer: {name}, Parameters: {param.numel()}')

    context = torch.zeros(batch_size, 77, 768)
    context = context.to(device)

    optimizer_1 = Adam(network.parameters(), lr=0.001)
    optimizer_2 = Adam(diffusion.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"Эпоха: {epoch}")
        network.train()
        diffusion.train()
        for step, batch in enumerate(data):

            # creating condition
            batch_low = black_white(batch).cuda()

            batch = batch.cuda()

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            encoder_noise = encoder_noise.cuda()

            to_idle(encoder)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            with torch.no_grad():  # Не вычисляем градиенты для encoder
                inf_imgs = encoder(batch_low, encoder_noise)
                hr_imgs = encoder(batch, encoder_noise)
                timesteps = torch.randint(0, num_training_steps, (batch_size,), device=device).long()
                noisy_samples, noise = trainer.add_noise(hr_imgs, timesteps)
                noisy_samples = noisy_samples.to(device)
                noise = noise.to(device)
                time_embedding = get_time_embedding(timesteps, batch_size).to(device)
                model_input = torch.cat([noisy_samples, inf_imgs], dim=1)
                model_input = network(noisy_samples, inf_imgs)


            model_output = diffusion(model_input, time_embedding)

            loss = F.l1_loss(noise, model_output)

            optimizer_2.zero_grad()
            optimizer_1.zero_grad()

            loss.backward()

            optimizer_1.step()
            optimizer_2.step()

        print(f"Epoch {epoch} | Loss: {loss.item()} ")

    checkpoint_1 = {
         'model_state_dict': network.state_dict(),
         'optimizer_state_dict': optimizer_1.state_dict(),
     }

    checkpoint_2 = {
        'model_state_dict': diffusion.state_dict(),
        'optimizer_state_dict': optimizer_2.state_dict(),
    }

    # Сохранение файла .ckpt
    torch.save(checkpoint_1, '../data/model_checkpoint_1.ckpt')
    torch.save(checkpoint_2, '../data/model_checkpoint_2.ckpt')
    to_idle(diffusion)


def low_batch(batch, IMG_SIZE_LOW = 64, IMG_SIZE = 512):
    data_transforms = [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE_LOW, IMG_SIZE_LOW)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)

    for step, sample in enumerate(batch):
        x_low = data_transform(batch[step])
        x_low = x_low.unsqueeze(0)
        if step == 0:
            x_in = x_low
        else:
            x_in = torch.cat([x_in, x_low], dim=0)
    return x_in


def black_white(batch):
    data_transforms = [
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),  # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
    ]

    data_transform = transforms.Compose(data_transforms)

    for step, sample in enumerate(batch):
        x_low = data_transform(batch[step])
        x_low = x_low.unsqueeze(0)
        if step == 0:
            x_in = x_low
        else:
            x_in = torch.cat([x_in, x_low], dim=0)
    return x_in



def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timesteps, batch_size):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)

    for i in range(batch_size):
        xi = torch.tensor([timesteps[i]], dtype=torch.float32)[:, None] * freqs[None]
        if i == 0:
            x = xi
        else:
            x = torch.cat([x, xi], dim=0)

    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
