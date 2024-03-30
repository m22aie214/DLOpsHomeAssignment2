import numpy as np

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Given data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Apply ReLU, Leaky ReLU, and Tanh to the data and print outputs
for value in random_values:
    relu_value = relu(value)
    leaky_relu_value = leaky_relu(value)
    tanh_value = tanh(value)
    print(f"ReLU({value}) = {relu_value}, Leaky ReLU({value}) = {leaky_relu_value}, Tanh({value}) = {tanh_value}")
