import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Verify PyTorch installation
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Load your dataset
df = pd.read_excel('1_Bare Soil_7bands.xlsx')  # Replace with your dataset file path
X = df.iloc[:, 1:].values  # Landsat 8 reflections (7 bands)
y = df.iloc[:, 0].values   # Ground truth SOC

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# Step 1: Transformer Model for Learning Complex Relationships
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return x[:, 0, :]  # Return the high-level representation

# Initialize the Transformer model
input_dim = X_tensor.shape[1]
model_dim = 64
num_heads = 4
num_layers = 2

transformer_model = TransformerModel(input_dim, model_dim, num_heads, num_layers)
optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Train the Transformer model
num_epochs = 50
training_losses = []

# Split data into training and validation sets
val_split = 0.2
val_size = int(len(X_tensor) * val_split)
train_size = len(X_tensor) - val_size

X_train_tensor = X_tensor[:train_size]
X_val_tensor = X_tensor[train_size:]

for epoch in range(num_epochs):
    transformer_model.train()
    optimizer.zero_grad()
    output = transformer_model(X_train_tensor.unsqueeze(1))
    optimizer.step()
    scheduler.step()
    training_losses.append(0)  # No loss calculation since we aren't reconstructing

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1}/{num_epochs}")

# Use Transformer to learn high-level representations
transformer_model.eval()
with torch.no_grad():
    learned_relationships = transformer_model(X_tensor.unsqueeze(1))

# Apply PCA to Transformer output
pca = PCA(n_components=3)
learned_relationships_pca = pca.fit_transform(learned_relationships.numpy())

# Combine PCA components with original L8 7 bands
combined_features = np.hstack((learned_relationships_pca, X_scaled))

# Step 2: Isolation Forest for Anomaly Detection
isolation_forest = IsolationForest(contamination=0.1)
isolation_forest.fit(combined_features)
anomalies = isolation_forest.predict(combined_features) == -1
noisy_samples = anomalies

noisy_samples_tensor = torch.tensor(noisy_samples, dtype=torch.bool)
noisy_indices = np.where(noisy_samples)[0]
print("\nUpdated indices of noisy samples after Isolation Forest detection:")
print(noisy_indices)
print("\nTotal number of noisy samples identified:")
print(len(noisy_indices))

# Step 3: Save noisy soil samples with SOC to an Excel file
df_noisy_samples = df.iloc[noisy_indices]
df_noisy_samples.to_excel('300noisy_samples_with_SOC.xlsx', index=False)
print("Noisy soil samples with SOC have been saved to '300noisy_samples_with_SOC.xlsx'.")

# Step 4: Conditional GAN for Data Reconstruction
class ConditionalGenerator(nn.Module):
    def __init__(self, soc_dim, noise_dim, output_dim):
        super(ConditionalGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(soc_dim + noise_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, noise, soc):
        input = torch.cat((noise, soc), dim=1)
        return self.fc(input)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, soc_dim, input_dim):
        super(ConditionalDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(soc_dim + input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, reflection, soc):
        input = torch.cat((reflection, soc), dim=1)
        return self.fc(input)

# Initialize the cGAN models
soc_dim = 1
noise_dim = 10
input_dim = X_tensor.shape[1]

generator = ConditionalGenerator(soc_dim, noise_dim, input_dim)
discriminator = ConditionalDiscriminator(soc_dim, input_dim)

# Loss function and optimizers
criterion = nn.BCELoss()
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

# Learning rate schedulers
g_scheduler = optim.lr_scheduler.StepLR(g_optimizer, step_size=50, gamma=0.5)
d_scheduler = optim.lr_scheduler.StepLR(d_optimizer, step_size=50, gamma=0.5)

# Labels for real and fake data with label smoothing
real_label = 0.9
fake_label = 0.1

# Train the Conditional GAN
num_epochs_gan = 100
batch_size = 32

def create_noise(batch_size, noise_dim):
    return torch.randn(batch_size, noise_dim)

for epoch in range(num_epochs_gan):
    for i in range(0, len(noisy_indices), batch_size):
        batch_indices = noisy_indices[i:i + batch_size]
        real_reflections = X_tensor[batch_indices]
        real_soc = y_tensor[batch_indices]
        real_labels = torch.full((real_reflections.size(0), 1), real_label)

        discriminator.zero_grad()
        output_real = discriminator(real_reflections, real_soc)
        d_loss_real = criterion(output_real, real_labels)
        d_loss_real.backward()

        noise = create_noise(real_reflections.size(0), noise_dim)
        fake_reflections = generator(noise, real_soc)
        fake_labels = torch.full((real_reflections.size(0), 1), fake_label)

        output_fake = discriminator(fake_reflections.detach(), real_soc)
        d_loss_fake = criterion(output_fake, fake_labels)
        d_loss_fake.backward()
        d_optimizer.step()

        generator.zero_grad()
        output_fake_for_g = discriminator(fake_reflections, real_soc)
        g_loss = criterion(output_fake_for_g, real_labels)
        g_loss.backward()
        g_optimizer.step()

    g_scheduler.step()
    d_scheduler.step()

    if epoch % 10 == 0 or epoch == num_epochs_gan - 1:
        print(f"Epoch {epoch + 1}/{num_epochs_gan}, D Loss: {d_loss_real.item() + d_loss_fake.item()}, G Loss: {g_loss.item()}")

# Generate reconstructed L8 reflections for noisy samples
generator.eval()
with torch.no_grad():
    noise = create_noise(len(noisy_indices), noise_dim)
    reconstructed_reflections = generator(noise, y_tensor[noisy_indices]).detach().numpy()

# Convert the reconstructed reflections back to the original scale
reconstructed_reflections = scaler.inverse_transform(reconstructed_reflections)

# Save the reconstructed reflections to an Excel file
reconstructed_df = pd.DataFrame(reconstructed_reflections, columns=df.columns[1:])
reconstructed_df.insert(0, 'SOC', y[noisy_indices])  # Insert SOC values
reconstructed_df.to_excel('300reconstructed_noisy_samples.xlsx', index=False)

print("Reconstructed noisy samples reflections have been saved to '300reconstructed_noisy_samples.xlsx'.")

# Step 5: Combine non-noisy samples with reconstructed noisy samples
# Identify non-noisy samples by excluding noisy indices
non_noisy_indices = np.setdiff1d(np.arange(len(df)), noisy_indices)

# Extract non-noisy samples
df_non_noisy = df.iloc[non_noisy_indices]

# Save the non-noisy samples to an Excel file
df_non_noisy.to_excel('300non_noisy_samples.xlsx', index=False)
print("Non-noisy samples have been saved to '300non_noisy_samples.xlsx'.")

# Combine non-noisy samples with reconstructed noisy samples
df_combined = pd.concat([df_non_noisy, reconstructed_df], ignore_index=True)

# Save the combined dataset to an Excel file
df_combined.to_excel('300combined_samples.xlsx', index=False)
print("Combined samples (non-noisy and reconstructed noisy samples) have been saved to '300combined_samples.xlsx'.")

# Step 6: Save non-noisy samples to an Excel file
df_non_noisy.to_excel('300non_noisy_samples.xlsx', index=False)
print("Non-noisy samples have been saved to '300non_noisy_samples.xlsx'.")

