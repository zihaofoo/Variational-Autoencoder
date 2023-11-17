import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from utils_public import *
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from CP3_sub import *
from hyperopt import hp, tpe, Trials, fmin
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

## Loading input data
topologies = np.load("topologies_train.npy")
constraints = np.load("constraints_train.npy", allow_pickle=True)

## Applying masking over data
masked_topologies = []
for i in trange(len(topologies)):
    mask = random_n_masks(np.array((64,64)), 4, 7).astype(bool)
    topology = topologies[i]
    masked_topology = topology*(1-mask) + 0.5*(mask)
    masked_topologies.append(masked_topology)
masked_topologies = np.stack(masked_topologies)

masked_constraints = mask_constraints(constraints)

## Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if gpu/tpu is available
data_in_tensor = torch.from_numpy(masked_topologies).float().to(device)
data_out_tensor = torch.from_numpy(topologies).float().to(device)

# expand dims of tensor in channel 1
data_in_tensor = data_in_tensor.unsqueeze(1)
data_out_tensor = data_out_tensor.unsqueeze(1)

input_channels = 1
image_size = (64, 64)

latent_dim = 10
hidden_size = 64
num_layers = 3
kernel_size = 3
stride = 2
num_epochs = 10
batch_size = 64

model = VAE(input_channels, hidden_size, num_layers, latent_dim, image_size, kernel_size, stride).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
summary(model, input_size=(input_channels, image_size[0], image_size[1]))

# Training loop
for epoch in range(1, num_epochs + 1):
    train(epoch, model, optimizer, data_in_tensor, data_out_tensor, batch_size)

originals = np.random.choice(np.arange(len(masked_topologies)), size=5, replace=False) #Select 5 random indices
reconstructions = reconstruct_from_vae(model, masked_topologies[originals], device) #Reconstruct
plot_reconstruction(topologies[originals], masked_topologies[originals], reconstructions) #Compare

## Testing and scoring
topologies_test = np.load("topologies_test.npy")
masked_topologies_test = np.load("masked_topologies_test.npy")
reconstructions_test = reconstruct_from_vae(model, masked_topologies_test, device) #Reconstruct
score = evaluate_score(masked_topologies_test, topologies_test, reconstructions_test)
print(f"Final Accuracy: {score:.5f}")


## Submission 
masked_topologies_submission = np.load("masked_topologies_submission.npy")
random_indices = np.random.choice(range(len(masked_topologies_submission)), 2)
plot_n_topologies(masked_topologies_submission[random_indices])
reconstructions_submission = reconstruct_from_vae(model, masked_topologies_submission, device) #Reconstruct
reconstructions_submission = np.round(reconstructions_submission).astype(bool)
assert reconstructions_submission.shape == (1200,64,64)
assert reconstructions_submission.dtype == bool
np.save("CP3_final_submission.npy", reconstructions_submission)