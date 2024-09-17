from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # We remove the Normalize transform as ToTensor already scales to [0, 1]
    ])
    
    dataset = datasets.MNIST('../datasets/mnist', train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)
    
    return loader