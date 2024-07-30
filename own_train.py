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


model_file = "/content/drive/MyDrive/models/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
#models = torch.load(model_file, DEVICE)

#model_file_1 = "../data/model_checkpoint.ckpt"
#checkpoint = torch.load(model_file_1, DEVICE)

#models['diffusion'] = checkpoint['model_state_dict']
#del models['diffusion']

#torch.save(models, '../data/full_model.ckpt')

#PROMPT

prompt = "increased image resolution"

#IMAGE TO IMAGE

strength = 0.1

#SAMPLER

sampler = "ddpm"
num_inference_steps = 50
seed = 42

#CREATE THE DATA
path = '/content/own__stable_diffusion/celebahq-resized-256x256/celeba_hq_256'
data = DATA(path)
data.create_data()
data.create_dataloader()

#SET PARAMETERS OF TRAINING

batch_size = 1
epochs = 20

train_from_scratch.train(
        prompt,
        data=data.data,
        strength=0.1,
        models=models,
        seed=seed,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=None,
        batch_size=batch_size,
        epochs=10,
        num_training_steps=200,
        checkpoint=False
)

