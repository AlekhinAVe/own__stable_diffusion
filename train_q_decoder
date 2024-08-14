import time
import pennylane as qml
import torch
import torch.nn as nn
import q_decoder
from data import DATA
import torch.optim as optim
import model_loader
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

#HYPERPARAMETERES
n_qubits = 4                # Number of qubits
step = 0.0004               # Learning rate
batch_size = 4              # Number of samples for each training step
num_epochs = 3              # Number of training epochs
q_depth = 6                 # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1    # Learning rate reduction applied every 10 epochs.
q_delta = 0.01              # Initial spread of random quantum weights
start_time = time.time()    # Start of the computation timer
BATCH_SIZE = 1
q_strength = 0.1

#setting device
dev = qml.device("default.qubit", wires=n_qubits)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#CREATE THE DATA
path = 'C:/Users/Andrey/faces/faces'
data = DATA(path)
data.create_data()
data.create_dataloader()

#LOAD CLASSICAL ENCODER AND DECODER FROM STABLE DIFFUSION
model_file = "../data/v1-5-pruned-emaonly.ckpt"
models = model_loader.preload_models_from_standard_weights(model_file, device)
encoder = models["encoder"]
encoder.to(device)
decoder = models["decoder"]
decoder.to(device)


#CREATE MODEL
model_hybrid = q_decoder.Hybrid(n_qubits, q_delta, q_depth, device)
model_hybrid = model_hybrid.to(device)

#SET LOSS
criterion = nn.CrossEntropyLoss()

#SET OPTIMIZER
optimizer_hybrid = optim.Adam(model_hybrid.parameters(), lr=step)

epochs = 10 # Try more!
losses = []
lepochs = []

#CREATE ENCODER NOISE
latents_shape = (1, 4, 64, 64)
generator = torch.Generator(device=device)
generator.manual_seed(42)
encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
encoder_noise = encoder_noise.cuda()


for epoch in range(epochs):
    for step, batch in enumerate(data.data):
        optimizer_hybrid.zero_grad()
        batch = batch.cuda()
        x = encoder(batch, encoder_noise)
        y = decoder(x)
        model_output = model_hybrid(x)
        loss = criterion(batch, model_output*q_strength + y*(1-q_strength))
        loss.backward()
        optimizer_hybrid.step()

    print(f"Epoch {epoch} | Loss: {loss.item()} ")
