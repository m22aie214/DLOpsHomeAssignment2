import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoidfn(x):
    return 1 / (1 + np.exp(-x))

def relufn(x):
    return np.maximum(0, x)

def leaky_relufn(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def tanhfn(x):
    return np.tanh(x)

# Generate x values
x = np.linspace(-5, 5, 100)

# Calculate y values for each activation function
y_sigmoid = sigmoidfn(x)
y_relu = relufn(x)
y_leaky_relu = leaky_relufn(x)
y_tanh = tanhfn(x)

# Plot the activation functions
plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_tanh, label='Tanh')

plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Activation Functions')
plt.legend()
plt.grid(True)
plt.show()
