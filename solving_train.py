import model_loader
import solving_train
import sr_training
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from data import DATA
import train_decoder

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")


model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)
models = torch.load(model_file, DEVICE)

model_file_1 = "../data/model_checkpoint_1.ckpt"
checkpoint_1 = torch.load(model_file_1, DEVICE)

model_file_2 = "../data/model_checkpoint_2.ckpt"
checkpoint_2 = torch.load(model_file_2, DEVICE)

#models['diffusion'] = checkpoint['model_state_dict']
#del models['diffusion']

torch.save(models, '../data/full_model.ckpt')

#PROMPT

prompt = "increased image resolution"

#IMAGE TO IMAGE

strength = 1

#SAMPLER

sampler = "ddpm"
num_inference_steps = 1000
seed = 42

#CREATE THE DATA
path = 'C:/Users/Andrey/faces/faces'
data = DATA(path)
data.create_data()
data.create_dataloader()

#SET PARAMETERS OF TRAINING

batch_size = 1
epochs = 5

solving_train.train(
        prompt,
        data=data.data,
        strength=1,
        models=models,
        seed=seed,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=None,
        batch_size=batch_size,
        epochs=epochs,
        num_training_steps=1000,
        checkpoint_1=False,
        checkpoint_2=False
)
