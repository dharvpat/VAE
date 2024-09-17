import torch
import torch.optim as optim
from tqdm import tqdm
from ..data_loader import load_mnist
from ..vae import VAE, vae_loss
import torch.nn as nn

# Improved loss function with KL annealing
def loss_function(recon_x, x, mu, logvar,input_dim, kl_weight=1.0):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + kl_weight * KLD

def train_vae(input_dim, hidden_dim, latent_dim, device, batch_size, epochs, lr):
    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    train_loader = load_mnist(batch_size=batch_size, train=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

    return model

# Improved training function
def train_improved_vae(model, device, train_loader, optimizer, scheduler, epoch, input_dim):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data.view(-1, input_dim), mu, logvar, input_dim)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f}')
    scheduler.step(avg_loss)
    return avg_loss