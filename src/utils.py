import matplotlib.pyplot as plt
import torch

def plot_results(model, test_data, num_examples=10):
    model.eval()
    with torch.no_grad():
        test_data = test_data.to(next(model.parameters()).device)
        recon, _, _ = model(test_data)

    fig, axes = plt.subplots(2, num_examples, figsize=(num_examples * 2, 4))
    for i in range(num_examples):
        axes[0, i].imshow(test_data[i].cpu().view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap='gray')
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original')
    axes[1, 0].set_ylabel('Reconstructed')
    plt.tight_layout()
    plt.show()

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, **model_params):
    model = model_class(**model_params)
    model.load_state_dict(torch.load(path))
    return model