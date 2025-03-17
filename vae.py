import torch
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, original_dim, latent_dim, intermediate_dim, beta):
        super(VAE, self).__init__()
        self.beta = beta
        # Encoder layers
        self.fc1 = nn.Linear(original_dim, intermediate_dim)
        self.fc21 = nn.Linear(intermediate_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(intermediate_dim, latent_dim)  # Log-variance
        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, intermediate_dim)
        self.fc4 = nn.Linear(intermediate_dim, original_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar