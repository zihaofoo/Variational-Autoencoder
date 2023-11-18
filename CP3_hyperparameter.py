from CP3_sub import *
from hyperopt import hp, fmin, tpe, Trials

def objective(params):
    # Initialize the VAE model with hyperparameters
    print(params)
    input_channels = 1
    image_size = (64, 64)
    num_epochs = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #Check if gpu/tpu is available

    model = VAE(input_channels, params['hidden_size'], params['num_layers'], 
                params['latent_dim'], image_size, params['kernel_size'], 
                params['stride']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    # Training loop (assumed validation set is available)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        num_batches = len(data_in_tensor) // params['batch_size']

        for batch_idx in range(num_batches): #Loop over batch indices
            start_idx = batch_idx * params['batch_size']
            end_idx = (batch_idx + 1) * params['batch_size']
            data_in = data_in_tensor[start_idx:end_idx] #Gather corresponding data
            data_out = data_out_tensor[start_idx:end_idx] #Gather corresponding data

            optimizer.zero_grad() #Set up optimizer
            recon_batch, mu, logvar = model(data_in) #Call model
            loss = loss_function(recon_batch, data_out, mu, logvar) #Call loss function
            loss.backward() #Get gradients of loss
            train_loss += loss.item() #Append to total loss
            optimizer.step() #Update weights using optimizeer

    ## Testing and scoring
    topologies_test = np.load("topologies_test.npy")
    masked_topologies_test = np.load("masked_topologies_test.npy")
    reconstructions_test = reconstruct_from_vae(model, masked_topologies_test, device) #Reconstruct
    score = evaluate_score(masked_topologies_test, topologies_test, reconstructions_test)
    print(params, score)
    return score
    # return {'loss': score, 'status': STATUS_OK}

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
    'hidden_size': hp.choice('hidden_size', [32, 64, 128, 256]),
    'num_layers': hp.choice('num_layers', [2, 3, 4, 5]),
    'kernel_size': hp.choice('kernel_size', [2, 3, 4]),
    'stride': hp.choice('stride', [1, 2]),
    'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
}

best = fmin(fn = objective,
            space = space,
            algo = tpe.suggest,
            max_evals = 50,  # You can adjust this based on computational resources
            trials = Trials())

print("Best hyperparameters:", best)
