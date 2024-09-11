import model_loader
import pipeline
import train_decoder
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from diffusion import Diffusion

ALLOW_CUDA = True
ALLOW_MPS = False

DEVICE1 = "cpu"
print(torch.cuda.is_available())

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

DEVICE = "cuda"

tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")
model_file = "../data/v1-5-pruned-emaonly.ckpt"
model_file_1 = "../data/model_checkpoint_1.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

checkpoint_1 = torch.load(model_file_1, DEVICE)


model_file_2 = "../data/model_checkpoint_2.ckpt"

checkpoint_2 = torch.load(model_file_2, DEVICE)

## TEXT TO IMAGE

# prompt = "A dog with sunglasses, wearing comfy hat, looking at camera, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
prompt = "draw a dog"
uncond_prompt = ""  # Also known as negative prompt
do_cfg = False
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
image_path = "../images/0_5_inf.png"
input_image = Image.open(image_path)

#image_path1 = "../images/0_5_hr.png"
#input_image1 = Image.open(image_path1)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 1

## SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
    checkpoint_1=checkpoint_1,
    checkpoint_2=checkpoint_2
)

# Combine the input image and the output image into a single image.
result = Image.fromarray(output_image)
result.save('result12.jpg')
