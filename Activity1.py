import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def sigmoidfns(x):
    return 1 / (1 + np.exp(-x))

def relufns(x):
    return np.maximum(0, x)

def leaky_relufns(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def tanhfns(x):
    return np.tanh(x)

# Generate x values
x = np.linspace(-5, 5, 100)

# Calculate y values for each activation function
y_sigmoid = sigmoidfns(x)
y_relu = relufns(x)
y_leaky_relu = leaky_relufns(x)
y_tanh = tanhfns(x)

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
