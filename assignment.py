
import matplotlib.pyplot as plt
import numpy as np

# Generate data
x = np.array([i for i in range(0, 10000, 100)])
a = -3
b = 4
# Adding noise to y
noise = np.random.normal(0, 50, size=x.shape)  # Adding Gaussian noise
y = a * x + b + noise  # y now includes noise

# Normalize x for better convergence
x_min, x_max = x.min(), x.max()
x_normalized = (x - x_min) / (x_max - x_min)  # Scale x to [0, 1]

# Initial parameters
a0 = 0.5
b0 = 1

# Learning rate and epochs
lr = 0.01  # Increased learning rate
epochs = 50000  # Increased number of epochs

# Mean Squared Error function
def error_mse(y_pred, y_true):
    return np.sum((y_pred - y_true) ** 2) / len(y_true)

errors = []
for i in range(epochs):
    # Predictions
    prediction = a0 * x + b0
    e = error_mse(prediction, y)
    errors.append(e)

    # Calculate gradients
    da_df = np.clip(np.dot(2 * (prediction - y), x) / len(y),-1,1)
    db_df = np.clip(np.sum(2 * (prediction - y)) / len(y),-1,1)

    # Update parameters
    a0 -= lr * da_df
    b0 -= lr * db_df

    # Print for debugging
    if i % 100 == 0:  # Print every 100 epochs
        print(f"Epoch {i}: a0 = {a0:.4f}, b0 = {b0:.4f}, error = {e:.4f}")

# Final output
print(f"Final model: y = {a0:.2f} * (x normalized) + {b0:.2f}")

# Plotting results
plt.scatter(x, y, label='Noisy Data', color='blue')
plt.plot(x, a0 * x + b0, color="orange", label='Fitted Line')
plt.legend()
plt.title('Linear Regression Fit with Noise')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Plotting the error over epochs
plt.figure(figsize=(10, 5))
plt.plot(errors)
plt.title('Mean Squared Error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.yscale('log')  # Optional: Log scale for better visualization
plt.show()
