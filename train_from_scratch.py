import torch
import numpy as np
from tqdm import tqdm
from ddpm_train import DDPMTrainer
import torch.nn.functional as F
from torch.optim import Adam
from clip import CLIP
from own_diffusion_1 import Diffusion
from torchvision import transforms
import gc


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

#we can train clip/encoder/decoder/diffusion

def train(
        prompt,
        data=None,
        strength=0.1,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
        batch_size=1,
        epochs=20,
        num_training_steps=200,
        checkpoint=None
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
    trainer = DDPMTrainer(generator)

    latents_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    encoder = models["encoder"]
    encoder.to(device)

    if checkpoint:
        diffusion = Diffusion().to(device)
        diffusion.load_state_dict(checkpoint['model_state_dict'], strict=True)
        diffusion.to(device)
    else:
        diffusion = Diffusion().to(device)
        # diffusion = models["diffusion"]
        # diffusion.load_state_dict(models["diffusion"])
        diffusion.to(device)

    decoder = models["decoder"]
    decoder.to(device)

    print('Model created')

    diffusion_params = sum(p.numel() for p in diffusion.parameters())
    print(diffusion_params)

    context = torch.rand(batch_size, 77, 768)
    context = context.to(device)

    # timesteps = tqdm(trainer.timesteps)

    optimizer = Adam(diffusion.parameters(), lr=0.001)

    for epoch in range(epochs):
        print(f"Эпоха: {epoch}")
        diffusion.train()
        for step, batch in enumerate(data):

            optimizer.zero_grad()

            batch_low = low_batch(batch).cuda()

            batch = batch.cuda()

            # нулевой шум для кодирования целевых изображений
            encoder_noise_zero = torch.zeros(latents_shape, device=device)
            encoder_noise_zero = encoder_noise_zero.cuda()

            trainer.set_strength(strength=strength)

            to_idle(encoder)

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            with torch.no_grad():  # Не вычисляем градиенты для encoder
                targets = encoder(batch, encoder_noise_zero)
                timesteps = torch.randint(0, num_training_steps, (batch_size,), device=device).long()
                noisy_samples, noise = trainer.add_noise(targets, timesteps)
                noisy_samples = noisy_samples.to(device)
                noise = noise.to(device)
                latents = encoder(batch_low, noisy_samples)
                time_embedding = get_time_embedding(timesteps, batch_size).to(device)

            model_output = diffusion(latents, context, time_embedding)

            loss = F.mse_loss(noise, model_output)

            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch} | Loss: {loss.item()} ")

    checkpoint = {
        'model_state_dict': diffusion.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    # Сохранение файла .ckpt
    torch.save(checkpoint, '../data/model_checkpoint.ckpt')
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
