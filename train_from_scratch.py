import torch
import numpy as np
from tqdm import tqdm
from ddpm_train import DDPMTrainer
import torch.nn.functional as F
from torch.optim import Adam
from clip import CLIP
from diffusion import Diffusion
from torchvision import transforms
import gc
from encoder import VAE_Encoder
from decoder import VAE_Decoder

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

    encoder = VAE_Encoder()
    #encoder = models["encoder"]
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

    #decoder = models["decoder"]
    decoder = VAE_Decoder()
    decoder.to(device)

    print('Model created')

    diffusion_params = sum(p.numel() for p in diffusion.parameters())
    print(diffusion_params)

    context = torch.rand(batch_size, 77, 768)
    context = context.to(device)

    # timesteps = tqdm(trainer.timesteps)

    optimizer = Adam(diffusion.parameters(), lr=0.001)

    for param in encoder.parameters():
        param.requires_grad = False

    for param in decoder.parameters():
        param.requires_grad = False

    for param in diffusion.parameters():
        param.requires_grad = True

    for epoch in range(epochs):
        print(f"Эпоха: {epoch}")
        diffusion.train()
        for step, batch in enumerate(data):

            # Установка флага requires_grad для всех параметров модели diffusion
            for param in diffusion.parameters():
                param.requires_grad = True

            optimizer.zero_grad()

            batch_low = low_batch(batch).cuda()

            batch = batch.cuda()

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

            encoder_noise = encoder_noise.cuda()

            # (Batch_Size, 4, Latents_Height, Latents_Width)
            with torch.no_grad():  # Не вычисляем градиенты для encoder
                latents = encoder(batch_low, encoder_noise)

            #old_weights = [param.clone() for param in diffusion.parameters()]

            # Add noise to the latents (the encoded input image)
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            trainer.set_strength(strength=strength)

            to_idle(encoder)

            timesteps = torch.randint(0, num_training_steps, (batch_size,), device=device).long()
            noisy_samples, noise = trainer.add_noise(latents, timesteps)
            noisy_samples = noisy_samples.to(device)
            noise = noise.to(device)
            time_embedding = get_time_embedding(timesteps, batch_size).to(device)
            model_output = diffusion(latents, context, time_embedding)

            # with torch.no_grad():  # Не вычисляем градиенты для decoder
            # output = decoder(model_output)

            loss = F.l1_loss(noise, model_output)

            gc.collect()
            torch.cuda.empty_cache()

            #loss.requires_grad = True
            loss.backward()

            gc.collect()
            torch.cuda.empty_cache()

            optimizer.step()

            print('all_right')

            #new_weights = list(diffusion.parameters())

            #lists_equal = all(torch.equal(a, b) for a, b in zip(old_weights, new_weights))
            #print(f"Параметры изменились: {not lists_equal}")


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
