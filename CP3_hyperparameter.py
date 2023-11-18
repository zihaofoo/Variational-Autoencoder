from CP3_sub import *
from hyperopt import hp, fmin, tpe, Trials

# Setting a seed value
seed_value = 42

# Set the seed for NumPy
np.random.seed(seed_value)

# Set the seed for PyTorch
torch.manual_seed(seed_value)

# For PyTorch to ensure reproducibility when using CUDA
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
data_in_tensor = data_in_tensor.unsqueeze(1)
data_out_tensor = data_out_tensor.unsqueeze(1)

space = {
    'lr': hp.loguniform('lr', np.log(0.0001), np.log(0.01)),
    'latent_dim': hp.choice('latent_dim', [8, 16, 32, 64, 128, 256]),
    'hidden_size': hp.choice('hidden_size', [32, 64, 128, 256, 512]),
    'num_layers': hp.choice('num_layers', [2, 3, 4, 5, 6]),
    'kernel_size': hp.choice('kernel_size', [3, 5, 7]),
    'stride': hp.choice('stride', [1, 2]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
    'num_epochs': hp.choice('num_epochs', [20, 30, 40, 50, 60, 70, 80, 90])
}

obj_func = lambda params: objective(params, data_in_tensor, data_out_tensor)
best = fmin(fn = obj_func,
            space = space,
            algo = tpe.suggest,
            max_evals = 50,  # You can adjust this based on computational resources
            trials = Trials())

print("Best hyperparameters:", best)
