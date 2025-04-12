import torch
import torch.nn as nn
import numpy as np

# Generate and preprocess data (same as before)
x1 = np.array([i for i in range(0, 10000, 100)])
x2 = np.array([i for i in range(0, 1000, 10)])

# Normalization
x1 = (x1 - x1.min()) / (x1.max() - x1.min())
x2 = (x2 - x2.min()) / (x2.max() - x1.min())

# Create target values
a, b, c = 5, 3, 1
y = a * x1 + b * x2 + c

# Convert to tensors and combine features
X = torch.stack([
    torch.from_numpy(x1).float(),
    torch.from_numpy(x2).float()
], dim=1)  # Shape: [100, 2]

y = torch.from_numpy(y).float().unsqueeze(1)  # Shape: [100, 1]

# Create model
model = nn.Linear(
    in_features=2,  # Two input features (x1 and x2)
    out_features=1,
    bias=True
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
for epoch in range(5000):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    loss = criterion(y_pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/5000], Loss: {loss.item():.4f}')

# Print learned parameters (matches your original equation)
print("\nLearned parameters:")
print(f"a (x1 weight): {model.weight[0, 0].item():.4f}")
print(f"b (x2 weight): {model.weight[0, 1].item():.4f}")
print(f"c (bias): {model.bias.item():.4f}")