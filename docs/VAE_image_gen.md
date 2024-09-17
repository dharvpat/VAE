# Variational Autoencoder (VAE) for Image Generation

This document provides a detailed explanation of the Variational Autoencoder (VAE) architecture and its implementation in this project for generating MNIST-like handwritten digits.

## VAE Architecture

A VAE consists of two main components: an encoder and a decoder.

1. **Encoder**: Compresses the input data into a latent space representation.
2. **Decoder**: Reconstructs the input data from the latent space representation.

The key feature of a VAE is the introduction of a probabilistic latent space, which allows for generating new, unseen data points.

### Encoder

The encoder maps the input `x` to two vectors: `μ` (mu) and `log(σ^2)` (log-variance). These represent the mean and log-variance of the latent space distribution for the input.

### Reparameterization Trick

To allow for backpropagation through the sampling process, we use the reparameterization trick:

```python
z = μ + σ * ε
```

where `ε` is a random vector sampled from a standard normal distribution.

### Decoder

The decoder takes a point `z` from the latent space and reconstructs the input data.

## Loss Function

The VAE loss function consists of two terms:

1. **Reconstruction Loss**: Measures how well the VAE can reconstruct the input data.
2. **KL Divergence**: Ensures that the latent space distribution is close to a standard normal distribution.

```python
loss = reconstruction_loss + KL_divergence
```

## Implementation Details

### ImprovedVAE

The `ImprovedVAE` class implements a simple fully-connected VAE:

- Encoder: Two fully-connected layers
- Decoder: Two fully-connected layers

### ConvVAE

The `ConvVAE` class implements a convolutional VAE:

- Encoder: Two convolutional layers followed by fully-connected layers
- Decoder: Fully-connected layer followed by two transposed convolutional layers

## Training Process

1. Forward pass: Encode input, sample from latent space, decode
2. Compute loss: Reconstruction loss + KL divergence
3. Backpropagate and update model parameters

## Image Generation

To generate new images:

1. Sample points from the standard normal distribution
2. Pass these points through the decoder

## Latent Space Visualization

Interpolating between two points in the latent space and decoding the intermediate points can provide insights into the learned representations and the smoothness of the generative process.

For more details on the implementation, refer to the source code in the `src/` directory.
