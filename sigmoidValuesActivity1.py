import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Given data
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

# Calculate and print sigmoid values for each data point
for value in random_values:
    sigmoid_value = sigmoid(value)
    print(f"Sigmoid({value}) = {sigmoid_value}")