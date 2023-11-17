from hyperopt import hp, tpe, Trials, fmin
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import torch
from CP3_sub import *

## Constraining random seeds
seed_value = 42

torch.manual_seed(seed_value)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    # Additional lines to further enforce determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
np.random.seed(seed_value)

# Assuming you already have the VAE class as defined in your script
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if gpu/tpu is available
image_size = (64, 64)
kernel_size = 3
stride = 2

# Define the hyperparameter search space
space = {
    'latent_dim': hp.choice('latent_dim', [5, 10, 20, 50, 100]),
    'hidden_size': hp.choice('hidden_size', [32, 64, 128, 256]),
    'num_layers': hp.choice('num_layers', [1, 2, 3, 4]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
    'num_epochs': hp.choice('num_epochs', [10, 20, 50, 100]),
    'input_channels': hp.choice('input_channels', [1, 3, 5])  # For example, for grayscale, RGB, etc.
}

def objective(params):
    # Load your data here
    train_data = ...  # replace with your data loading logic
    train_loader = DataLoader(train_data, batch_size=int(params['batch_size']), shuffle=True)

    # Initialize the model with input_channels
    model = VAE(int(params['input_channels']), params['hidden_size'], params['num_layers'], params['latent_dim'], image_size, kernel_size, stride).to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop with num_epochs
    for epoch in range(int(params['num_epochs'])):
        model.train()
        total_loss = 0
        for data_in, data_out in train_loader:
            data_in, data_out = data_in.to(device), data_out.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data_in)
            loss = loss_function(recon_batch, data_out, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        avg_loss = total_loss / len(train_loader)

    return avg_loss


# Run the optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=50,  # You can change this number based on how long you want to search
    trials=trials
)

print("Best hyperparameters:", best)
