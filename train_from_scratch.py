import torch
import numpy as np
from tqdm import tqdm
from ddpm_train import DDPMTrainer
import torch.nn.functional as F
from torch.optim import Adam
from clip import CLIP
from encoder import VAE_Encoder
from own_diffusion import Diffusion

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

#we can train clip/encoder/decoder/diffusion

def train(
        prompt,
        data=None,
        strength=1.0,
        models={},
        seed=None,
        device=None,
        idle_device=None,
        tokenizer=None,
        batch_size=64,
        epochs=1000,
        num_training_steps=1000
):
    with torch.no_grad():
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


        #SET THE TRAINER
        trainer = DDPMTrainer(generator)

        latents_shape = (batch_size, 4, LATENTS_HEIGHT, LATENTS_WIDTH)


        encoder = VAE_Encoder()
        encoder.to(device)

        diffusion = Diffusion()
        diffusion.to(device)

        print('Model created')

        diffusion_params = sum(p.numel() for p in diffusion.parameters())
        print(diffusion_params)

        context = torch.rand(1, 77, 768)

        #timesteps = tqdm(trainer.timesteps)

        optimizer = Adam(diffusion.parameters(), lr=0.001)

        for epoch in range(epochs):
            print(f"Эпоха: {epoch}")
            for step, batch in enumerate(data):
                optimizer.zero_grad()

                batch = batch.cuda()

                # (Batch_Size, 4, Latents_Height, Latents_Width)
                encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

                encoder_noise = encoder_noise.cuda()

                # (Batch_Size, 4, Latents_Height, Latents_Width)
                latents = encoder(batch, encoder_noise)

                # Add noise to the latents (the encoded input image)
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                trainer.set_strength(strength=strength)

                to_idle(encoder)

                timesteps = torch.randint(0, num_training_steps, (batch_size,), device=device).long()
                noisy_samples, noise = trainer.add_noise(latents, timesteps)
                time_embedding = get_time_embedding(timesteps).to(device)
                model_output = diffusion(noisy_samples, context, time_embedding)
                loss = F.l1_loss(noise, model_output)
                loss.backward()
                optimizer.step()

        to_idle(diffusion)


def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timesteps):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # Shape: (batch_size, 160)
    x = torch.tensor([timesteps], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (batch_size, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)