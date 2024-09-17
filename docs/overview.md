# VAE Image Generation Project Overview

This project implements a Variational Autoencoder (VAE) for generating handwritten digits using the MNIST dataset. The VAE is trained to learn a compressed latent representation of the input images and can generate new, similar images from this latent space.

## Project Structure

The project is organized as follows:

- `src/`: Contains the main source code for the project.
  - `models/`: Implementations of VAE architectures.
  - `training/`: Scripts for training the VAE.
  - `data_loader.py`: Utilities for loading the MNIST dataset.
  - `generate_images.py`: Functions for generating new images using the trained VAE.
  - `utils.py`: Miscellaneous utility functions.

- `experiments/`: Contains code for training and image generation experiments.

- `docs/`: Project documentation.

## Key Components

1. **Data Loading**: The MNIST dataset is loaded using PyTorch's `torchvision` library.

2. **VAE Models**: Two VAE architectures are implemented:
   - `BasicVAE`: A simple fully-connected VAE.
   - `ConvVAE`: A convolutional VAE for potentially better performance.

3. **Training**: The `train_vae.py` script handles the training process, including loss calculation and optimization.

4. **Image Generation**: The `generate_images.py` script provides functions for generating new images from the trained VAE.

5. **Visualization**: Utility functions in `utils.py` help visualize the original, reconstructed, and generated images.

## Getting Started

1. Install the required dependencies listed in `requirements.txt`.
2. Run the training notebook in `experiments/notebooks/VAE_mnist_training.ipynb` to train the VAE.
3. Use the generation notebook in `experiments/notebooks/VAE_mnist_generation.ipynb` to generate new images.

For more detailed information on the VAE architecture and implementation, refer to `vae_image_generation.md`.