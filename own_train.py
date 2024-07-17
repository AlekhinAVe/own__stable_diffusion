import model_loader
import train_from_scratch
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from data import DATA

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")


tokenizer = CLIPTokenizer("../data/vocab.json", merges_file="../data/merges.txt")


#PROMPT

prompt = "increased image resolution"

#IMAGE TO IMAGE

strength = 1

#SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

#CREATE THE DATA
path = 'C:/Users/Andrey/faces/faces'
data = DATA(path)
data.create_data()
data.create_dataloader()

#SET PARAMETERS OF TRAINING

batch_size = 64
epochs = 1000

train_from_scratch.train(
        prompt,
        data=data.data,
        strength=1.0,
        models=None,
        seed=seed,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
        batch_size=64,
        epochs=1000,
        num_training_steps=1000
)
