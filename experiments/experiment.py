import sys
import os
sys.path.append(os.path.abspath('/Users/dharv/VAE/VAE')) #Home Directory for VAE project
print(sys.path)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.data_loader import load_mnist
from src.utils import plot_results
from src.models.basic_vae import ImprovedVAE
from src.training.train_vae import train_improved_vae

# Set up device and hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
input_dim = 784
hidden_dim = 512
latent_dim = 32
batch_size = 128
epochs = 50     
lr = 1e-2
weight_decay = 1e-5  # Added L2 regularization

# Main training loop
model = ImprovedVAE(input_dim, hidden_dim, latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

train_loader = load_mnist(batch_size=batch_size, train=True)

for epoch in range(1, epochs + 1):
    kl_weight = min(1.0, epoch / 10)  # KL annealing
    train_improved_vae(model, device, train_loader, optimizer, scheduler, epoch, input_dim)

# Save the trained model
torch.save(model.state_dict(), 'improved_vae_model.pth')

# Visualize results
test_loader = load_mnist(batch_size=10, train=False)
test_data = next(iter(test_loader))[0].to(device)
plot_results(model, test_data)