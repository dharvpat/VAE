import torch
import matplotlib.pyplot as plt
from .models import BasicVAE  # or ConvVAE

def generate_images(model, num_images, device):
    model.eval()
    with torch.no_grad():
        sample = torch.randn(num_images, model.fc_mu.out_features).to(device)
        sample = model.decode(sample).cpu()
        return sample

def plot_generated_images(images, nrow, ncol):
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(images[i].view(28, 28), cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the trained model
    model = BasicVAE(input_dim=784, hidden_dim=400, latent_dim=20).to(device)
    model.load_state_dict(torch.load('vae_model.pth'))

    # Generate images
    num_images = 16
    generated_images = generate_images(model, num_images, device)

    # Plot the generated images
    plot_generated_images(generated_images, 4, 4)

if __name__ == "__main__":
    main()