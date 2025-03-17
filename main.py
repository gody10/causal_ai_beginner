import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from vae import VAE
from utility import vae_loss
from classifier import Classifier

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and Preprocess MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),              # Convert images to tensors (C, H, W)
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to 784-dimensional vector
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Instantiate VAE
original_dim = 784  # 28x28 flattened
latent_dim = 10     # Number of latent variables
intermediate_dim = 256  # Hidden layer size
beta = 4.0          # β parameter for KL weighting
vae = VAE(original_dim, latent_dim, intermediate_dim, beta).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Train VAE
print("Training β-VAE...")
for epoch in range(80):
    vae.train()
    train_loss = 0
    for data, _ in train_loader:  # Ignore labels for VAE
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = vae_loss(recon_batch, data, mu, logvar, beta)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print(f"VAE Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}")

classifier = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

# Train Classifier
print("\nTraining Classifier...")
for epoch in range(20):
    classifier.train()
    train_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = classifier(data)
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(train_loader.dataset)
    print(f"Classifier Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.6f}, Accuracy: {accuracy:.4f}")

# Causal Inference - Compute Mean Latent Values
z_means = []
labels = []
vae.eval()
with torch.no_grad():
    for data, target in train_loader:
        data = data.to(device)
        mu, _ = vae.encode(data)
        z_means.append(mu.cpu())
        labels.append(target)
z_means = torch.cat(z_means, dim=0)
labels = torch.cat(labels, dim=0)

# Compute mean latent vector for each digit
digit_means = [z_means[labels == d].mean(dim=0) for d in range(10)]
digit_means = torch.stack(digit_means)

# Assume latent indices (simplified assumption; in practice, analyze variance)
digit_latent_idx = 0  # Latent dimension for digit identity
style_latent_idx = 1  # Latent dimension for style
digit_value = digit_means[5, digit_latent_idx]  # Mean value for digit 5

# Generate Intervened Images
def generate_intervened_images(digit_value, style_idx, style_range, num_samples=10):
    z_sample = torch.zeros(num_samples, latent_dim)
    z_sample[:, digit_latent_idx] = digit_value  # Fix digit latent variable
    style_values = np.linspace(style_range[0], style_range[1], num_samples)
    z_sample[:, style_idx] = torch.tensor(style_values, dtype=torch.float32)  # Vary style
    # Set other dimensions to random noise
    for i in range(latent_dim):
        if i not in [digit_latent_idx, style_idx]:
            z_sample[:, i] = torch.randn(num_samples)
    z_sample = z_sample.to(device)
    with torch.no_grad():
        images = vae.decode(z_sample)
    return images, style_values

style_range = (-3, 3)  # Range to vary style variable
intervened_images, style_values = generate_intervened_images(digit_value, style_latent_idx, style_range)

# ### Step 6: Classify Intervened Images
classifier.eval()
with torch.no_grad():
    predictions = classifier(intervened_images)
    prob_5 = F.softmax(predictions, dim=1)[:, 5].cpu().numpy()  # Probability for digit 5

# Visualize Results
# Plot causal effect
plt.figure(figsize=(10, 5))
plt.plot(style_values, prob_5, marker='o')
plt.xlabel('Style Latent Variable Value')
plt.ylabel('Probability of Being Classified as 5')
plt.title('Causal Effect of Style on Classification')
plt.grid(True)
#plt.show()
plt.savefig('causal_effect.png')

# Display intervened images
plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(1, 10, i+1)
    image = intervened_images[i].cpu().view(28, 28).numpy()
    plt.imshow(image, cmap='gray')
    plt.title(f"Style: {style_values[i]:.2f}")
    plt.axis('off')
plt.tight_layout()
#plt.show()
plt.savefig('causal_inference.png')