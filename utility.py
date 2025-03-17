import torch.nn.functional as F

# Define VAE loss function
def vae_loss(recon_x, x, mu, logvar, beta):
    # Reconstruction loss: sum over features, mean over batch
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=1).mean()
    # KL divergence: sum over latent dimensions, mean over batch
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return recon_loss + beta * kl_loss

