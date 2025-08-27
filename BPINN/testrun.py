# Create a very simple Neunral netowrk script that runs for 10 seconds

import time
import numpy as np

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the sigmoid derivative function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the input dataset
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the output dataset
outputs = np.array([[0], [1], [1], [0]])

# Seed the random number generator
np.random.seed(1)

# Initialize the weights
weights0 = 2 * np.random.random((2, 4)) - 1

weights1 = 2 * np.random.random((4, 1)) - 1

# Set the learning rate
learning_rate = 0.1

# Run the neural network for 10 seconds
start_time = time.time()

while time.time() - start_time < 10:
    # Feed forward through the network
    layer0 = inputs
    layer1 = sigmoid(np.dot(layer0, weights0))
    layer2 = sigmoid(np.dot(layer1, weights1))

    # Calculate the error
    layer2_error = outputs - layer2

    # Calculate the gradient
    layer2_gradient = layer2_error * sigmoid_derivative(layer2)

    # Calculate the error for layer1
    layer1_error = layer2_gradient.dot(weights1.T)

    # Calculate the gradient for layer1
    layer1_gradient = layer1_error * sigmoid_derivative(layer1)

    # Update the weights
    weights1 += layer1.T.dot(layer2_gradient) * learning_rate
    weights0 += layer0.T.dot(layer1_gradient) * learning_rate

# Print the final output
print("Output after training:")
print(layer2)

# Print the final weights
print("Final weights:")
print(weights0)
print(weights1)

# Print the time taken
print("Time taken:")
print(time.time() - start_time)
