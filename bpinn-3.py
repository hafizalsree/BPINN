# This is a Bayesian Neural Network implementation using TensorFlow for a simple y = sin(x) function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the parameters for the Burger's equation
nu = 0.2 / np.pi  # Viscosity
N = 1000  # Number of data points

# Generate the spatial domain
x = np.linspace(-1, 1, N)
t = np.linspace(0, 1, N)
X, T = np.meshgrid(x, t)

# Initial condition
u0 = -np.sin(np.pi * x)

# Generate the dataset
def burgers_exact(x, t, nu):
    return -2 * nu * np.pi * np.sin(np.pi * (x - 4 * nu * t)) * np.exp(-np.pi**2 * nu * t)
# Exact solution: u(x, t) = -2 * nu * pi * sin(pi * (x - 4 * nu * t)) * exp(-pi^2 * nu * t)
# If t = 0, u(x, 0) = -2 * nu * pi * sin(pi * x) = -2 * 0.01/pi * pi * sin(pi * x) = -0.02 * sin(pi * x)
# If t = 1, u(x, 1) = -2 * nu * pi * sin(pi * (x - 4 * nu)) = -2 * 0.01/pi * pi * sin(pi * (x - 4 * 0.01/pi)) = -0.02 * sin(pi * (x - 4 * 0.01/pi))

U_1D = burgers_exact(x, t, nu)
U = burgers_exact(X, T, nu)

# Plot sizes of all the data
print("x shape:", x.shape)
print("t shape:", t.shape)
print("u0:", u0.shape)
print("U_1D shape:", U_1D.shape)
print("X shape:", X.shape)
print("T shape:", T.shape)
print("U shape:", U.shape)

plt.figure(figsize=(8, 6))
plt.plot(x, U_1D)
plt.xlabel('x')
plt.ylabel('u')
plt.title('Initial Condition')
plt.show()

# Define the loss function (negative log posterior)
def negative_log_posterior(model, U, U_pred):
    # We minimize the negative log posterior
    return -model.log_posterior(U, U_pred)

def pde_residual(model, x, t, u, nu):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(x)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            #u = model.forward(tf.concat([x], axis=1))
            u = model.forward(x)
        u_x = tape.gradient(u, x)
    u_xx = tape2.gradient(u_x, x)
    #u_t = tape.gradient(u, t)
    residual = u * u_x - nu * u_xx
    return residual

def pinn_loss(model, x, t, u0, nu):
    # Initial condition loss
    #u_pred_initial = model.forward(tf.concat([x, tf.zeros_like(x)], axis=1))
    #loss_initial = tf.reduce_mean(tf.square(u_pred_initial - u0))

    # Boundary condition loss
    #u_pred_left = model.forward(tf.concat([tf.ones_like(t) * x[0], t], axis=1))
    #u_pred_right = model.forward(tf.concat([tf.ones_like(t) * x[-1], t], axis=1))
    #loss_boundary = tf.reduce_mean(tf.square(u_pred_left)) + tf.reduce_mean(tf.square(u_pred_right))
    u_pred_left = model.forward([x[0]])
    u_pred_right = model.forward([x[-1]])
    loss_boundary = tf.reduce_mean(tf.square(u_pred_left)) + tf.reduce_mean(tf.square(u_pred_right))

    '''# PDE residual loss
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        u_pred = model(tf.concat([x, t], axis=1))
    residual = pde_residual(u_pred, x, t, nu)
    loss_pde = tf.reduce_mean(tf.square(residual))'''
    # Total loss
    total_loss = loss_boundary 
    return total_loss

import time

# Define tic and toc functions
def tic():
    return time.time()

def toc(start_time):
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    return elapsed_time

Y = burgers_exact(x, t, nu)
X = x
Y = Y.reshape(-1, 1).astype(np.float32)
X = X.reshape(-1, 1).astype(np.float32)
hidden1 = N

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the neural network architecture
class BayesianNeuralNetwork:
    def __init__(self, input_dim, hidden1, hidden2, output_dim, prior_std):
        # Initialize weights and biases as TensorFlow variables
        initializer = tf.initializers.GlorotUniform()

        # Define trainable parameters (weights and biases)
        self.weights = [
            tf.Variable(initializer([input_dim, hidden1]), dtype=tf.float32),
            tf.Variable(initializer([hidden1, hidden2]), dtype=tf.float32),
            tf.Variable(initializer([hidden2, output_dim]), dtype=tf.float32)
        ]
        
        self.biases = [
            tf.Variable(tf.zeros([hidden1]), dtype=tf.float32),
            tf.Variable(tf.zeros([hidden2]), dtype=tf.float32),
            tf.Variable(tf.zeros([output_dim]), dtype=tf.float32)
        ]

        # Prior parameters (assuming Gaussian priors)
        self.prior_std = prior_std
        self.prior_vars = [
            tf.constant(self.prior_std**2, dtype=tf.float32) for _ in self.weights + self.biases
        ] # sigma**2. We assume that the prior is N(0, sigma**2).
        # for _ in self.weights + self.biases: create a list of prior_vars with the same length as weights and biases
        # So it creates a list of prior_vars for each weight and bias, total 6 elements in the list
        # The prior_std is the standard deviation of the prior, and we assume that the prior is N(0, sigma**2)

    def forward(self, X):
        # Layer 1 with tanh activation
        hidden1 = tf.matmul(X, self.weights[0]) + self.biases[0]
        hidden1_activation = tf.nn.tanh(hidden1)
        
        # Layer 2 with tanh activation
        hidden2 = tf.matmul(hidden1_activation, self.weights[1]) + self.biases[1]
        hidden2_activation = tf.nn.tanh(hidden2)
        
        # Output layer (linear activation)
        output = tf.matmul(hidden2_activation, self.weights[2]) + self.biases[2]
        return output

    def log_prior(self): # Role: Define initial beliefs about each parameter, acting as regularizers.
        # Compute log prior probability of weights and biases
        log_prior = 0.0
        for var, var_prior_var in zip(self.weights + self.biases, self.prior_vars): 
            # Assuming Gaussian prior: log p(w) = -0.5 * log(2πσ²) - w²/(2σ²)
            #log_prior += -0.5 * tf.reduce_sum(tf.math.log(2.0 * np.pi * var_prior_var**2)) # log(2πσ²)
            log_prior += -tf.reduce_sum(tf.square(var)) / (2.0 * var_prior_var**2) # -w²/(2σ²)
        log_prior_store.append(log_prior)
        return log_prior

    def log_likelihood(self, Y_true, Y_pred): # Role: Represents the probability of observing the data given the parameters, modeling how well the network fits the data.
        # Assuming Gaussian likelihood: p(Y|X,w) = N(Y_pred, σ²)
        # log p(Y|X,w) = -0.5 * log(2πσ²) - (Y - Y_pred)^2 / (2σ²)
        #log_likelihood = -0.5 * tf.cast(tf.size(Y_true), tf.float32) * tf.math.log(2.0 * np.pi * likelihood_std**2) # log(2πσ²)
        likelihood_std = np.std(Y_pred)
        log_likelihood = 0.0
        log_likelihood += -tf.reduce_sum(tf.square(Y_true - Y_pred)) / (2.0 * likelihood_std**2) # -(Y - Y_pred)^2 / (2σ²)
        log_likelihood_store.append(log_likelihood)
        return log_likelihood

    def log_posterior(self, Y_true, Y_pred): # Role: Combines priors and likelihood to update beliefs about parameters after observing the data.
        # log posterior ∝ log likelihood + log prior
        #print("Likelihood:   ", self.log_likelihood(Y_true, Y_pred))
        #print("Prior:   ", self.log_prior())
        log_posterior_store.append(self.log_likelihood(Y_true, Y_pred) + self.log_prior())
        return self.log_likelihood(Y_true, Y_pred) + self.log_prior()

# Initialize the Bayesian neural network
input_dim = 1
hidden1 = 30
hidden2 = 30
output_dim = 1
prior_std = np.sqrt(50/hidden1)  # Standard deviation for the prior
log_prior_store = []
log_likelihood_store = []
log_posterior_store = []

model = BayesianNeuralNetwork(input_dim, hidden1, hidden2, output_dim, prior_std)

# Define the loss function (negative log posterior)
def negative_log_posterior(model, Y_true, Y_pred):
    # We minimize the negative log posterior
    return -model.log_posterior(Y_true, Y_pred)

# Prepare the optimizer
learning_rate = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate)  # Adam often works better for Bayesian methods

# Training parameters
epochs = 20000
print_interval = 100
best_rmse = float('inf')
loss_all = []
loss_all_posterior = []
loss_all_pde = []
loss_all_conditions = []
modified_loss_all = []

# Convert data to TensorFlow tensors
X_tf = tf.convert_to_tensor(X)
Y_tf = tf.convert_to_tensor(Y)

print("GPU cuda is available: ", tf.test.is_built_with_cuda())

# Training loop
start_time = tic()
with tf.device('/GPU:0'):
    for epoch in range(1, epochs + 1):
    
        with tf.GradientTape() as tape:
            # Forward pass
            Y_pred = model.forward(X_tf)
            # Compute negative log posterior
            loss_posterior = negative_log_posterior(model, Y_tf, Y_pred)
            modified_loss = -1*(model.log_likelihood(Y_tf, Y_pred)/N + model.log_prior())
            residual = pde_residual(model, X_tf, T, Y_tf, nu)
            loss_pde = tf.reduce_mean(tf.square(residual))
            conditions = pinn_loss(model, X_tf, T, u0, nu)
            alpha_const = 0.010912768542766571
            alpha_const_inv = 1/alpha_const
            lambda_const = loss_posterior/(loss_pde + conditions)
            #loss = alpha_const*loss_posterior + loss_pde + conditions
            loss = loss_posterior + alpha_const_inv*(loss_pde + conditions)
            #loss = loss_posterior + loss_pde + conditions
            loss_all_posterior.append(loss_posterior)
            loss_all_pde.append(loss_pde)
            loss_all_conditions.append(conditions)
            loss_all.append(loss)
            modified_loss_all.append(modified_loss)
        
        # Compute gradients
        loss_chosen = loss
        gradients = tape.gradient(loss_chosen, model.weights + model.biases) # Purpose of gradient: To update the weights and biases in the direction that minimizes the loss function.
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.weights + model.biases))
        
        # Print loss every 'print_interval' epochs
        if epoch % print_interval == 0 or epoch == 1:
            print(f'Epoch {epoch} --- Anti Log Posterior = {np.exp(loss_chosen.numpy()):.6f} --- Modified Loss = {loss_chosen.numpy():.6f}')   
            rmse_train = tf.sqrt(tf.reduce_mean((Y_tf - Y_pred)**2))
            print(f"RMSE: {rmse_train.numpy():.10f} --- loss_pde: {loss_pde.numpy()/(2*np.std(Y_pred)**2):.10f} --- conditions: {conditions.numpy()/(2*np.std(Y_pred)**2):.10f}")
    
        # Save weights and biases only if lower rmse than before
        if rmse_train < best_rmse:
            best_rmse = rmse_train
            MAP_weights = model.weights
            MAP_biases = model.biases
            best_epoch = epoch
            if epoch % print_interval == 0 or epoch == 1:
                #print("Improvement in RMSE. Weights and biases saved.")
                print("----------------------------------------GPU cuda is available: ", tf.test.is_built_with_cuda())
        else:
            if epoch % print_interval == 0 or epoch == 1:
                print("No improvement in RMSE. Weights and biases not saved.")
                print("----------------------------------------GPU cuda is available: ", tf.test.is_built_with_cuda())
                
elapsed_time = toc(start_time)

# After training, plot the model predictions
print(f"Best Epoch at: {best_epoch}")
plt.figure(figsize=(8, 6))
plt.plot(X, Y, alpha=1, label='True Data')
# Sort X for better visualization of the prediction curve
sorted_indices = np.argsort(X.squeeze())
X_sorted = X[sorted_indices]
Y_pred_sorted = Y_pred.numpy()[sorted_indices]
plt.plot(X_sorted, Y_pred_sorted, color='red', label='Model Prediction', linewidth=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Bayesian Neural Network Fit to sin(X)")
plt.legend()
plt.show()

# Print RMSE
model.weights = MAP_weights
model.biases = MAP_biases
U_pred = model.forward(X_tf)
rmse = tf.sqrt(tf.reduce_mean(tf.square(Y_tf - U_pred)))
print("Best epoch at: ", best_epoch)
print(f"RMSE: {rmse.numpy():.8f}")

# Convert loss to anti-log
print(f"Loss: {(loss.numpy()):.8f}")
print(f"Anti Log Posterior = {np.exp(loss.numpy()):.6f}")

# Corrected modified loss
print(f"Corrected Modified Loss: {modified_loss.numpy():.8f}")

# Plot the loss
plt.figure(figsize=(8, 6))
plt.plot(loss_all, linewidth=5)
plt.plot(loss_all_pde)
plt.plot(loss_all_conditions)
plt.plot(loss_all_posterior)
plt.plot(modified_loss_all)
plt.axhline(y=0, color='b', linestyle='-')
plt.text(0, 0, 'Elapsed Time: ' + str(elapsed_time), fontsize=12, color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Negative Log Posterior")
plt.legend(["Total Loss", "PDE Loss", "Conditions Loss", "Negative Log Posterior", "Modified Loss"])
plt.yscale('log')
plt.savefig('Loss_3.png')
plt.show()


# Plot true vs predicted
plt.figure(figsize=(8, 6))
plt.plot(x, Y_tf, alpha=1, label='True Data')
plt.plot(x, U_pred, color='red', label='Model Prediction', linewidth=2)
plt.xlabel("True Y")
plt.ylabel("Predicted Y")
plt.title("True vs Predicted Y")
plt.legend()
plt.savefig('True_vs_Predicted_3.png')
plt.show()

# Save all data in a file
import pickle
with open('BPINN_1.pkl', 'wb') as f:
    pickle.dump(MAP_weights, f)
    pickle.dump(MAP_biases, f)