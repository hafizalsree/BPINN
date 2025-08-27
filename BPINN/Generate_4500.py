# Load data
import pickle

num_leapfrog_steps = 4500
file = 'BPINN_leapfrog_' + str(num_leapfrog_steps) + '.pkl'

with open('../../Results/HMC/' + file, 'rb') as f:
    samples = pickle.load(f)
    momentum_norm_total = pickle.load(f)
    alphas = pickle.load(f)
    proposed_H_store = pickle.load(f)
    current_H = pickle.load(f)
    grad_list_total = pickle.load(f)
    negative_log_posterior_list_total = pickle.load(f)
    acceptance_list = pickle.load(f)
    MAP_weights = pickle.load(f)
    MAP_biases = pickle.load(f)

import numpy as np
import matplotlib.pyplot as plt

# From the file name, we know that the model is trained with 100 leapfrog steps so set n_steps using the file name
num_samples = len(acceptance_list)

# Find number of accepted samples
num_accepted = np.sum(acceptance_list)
num_rejected = num_samples - num_accepted
print('No of Accepted: ', num_accepted)
print('No of Rejected: ', num_rejected)

# find the index of the first accepted sample
first_accepted = 0
for i in range(num_samples):
    if acceptance_list[i] == 1:
        first_accepted = i
        break
    else:
        first_rejected = 0
print('First accepted sample: ', first_accepted, 'First rejected sample: ', first_rejected)

remove = []
remove = [6, 88, 89, 26] # For 1500 leapfrog steps
remove = [6, 65, 2, 44, 68, 3] # For 3000 leapfrog steps
remove = [70, 20, 54] # For 3750 leapfrog steps
remove = [19, 0, 12] # For 4500 leapfrog steps

# Plot energy per sample
plt.figure(figsize=(35, 30))
plt.subplot(5, 2, 1)
# bar colours based on acceptance rate color map
plt.bar(range(1, num_samples+1), proposed_H_store, color=plt.cm.Greens(np.array(alphas)))
plt.axhline(y=current_H, color='r', linestyle='-')
plt.xlabel("Sample")
plt.ylabel("Energy")
plt.title("Energy per Sample")
plt.ylim(0, 1200)

plt.subplot(5, 2, 2)
# minus proposal_H_store - current_H for all
acceptance_alphas = [min(1, np.exp(proposed_H - current_H)) for proposed_H in proposed_H_store]
plt.bar(range(1, num_samples+1),alphas)
plt.ylim(-0.2,1.2)
plt.axhline(y=1, color='r', linestyle='-')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('Sample')
plt.ylabel('Alpha (Acceptance Probability)')

# Create an array of 100 colours
import matplotlib.cm as cm
colors = [cm.viridis(x) for x in np.linspace(0, 1, num_samples)]
# create array of sample 1, sample 2, sample 3, etc. for legend
samples_array = [f'Sample {i+1}' for i in range(num_samples)]
plt.subplot(5, 2, 3)
for j in range(num_samples):
    if j not in remove:
        if acceptance_list[j] == 1:  # Only plot if sample was accepted
            for i in range(num_leapfrog_steps):
                if j != first_accepted:  # Only label the first point of each accepted sample
                    plt.plot(i, momentum_norm_total[j][i][1], 'o', color=colors[j])
                plt.plot(i, momentum_norm_total[first_accepted][i][1], 'o', color='red', markersize=15, label=samples_array[j])                    
plt.xlabel('Leapfrog Steps')
plt.ylabel('Momentum Norm')
#plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.title('Momentum Norm for 2nd Layer Weights - ' + str(num_accepted) + ' Accepted')

plt.subplot(5, 2, 4)
for j in range(num_samples):
    if j not in remove:
        if acceptance_list[j] == 0:  # Only plot if sample was accepted
            for i in range(num_leapfrog_steps):
                if j != first_rejected:  # Only label the first point of each accepted sample
                    plt.plot(i, momentum_norm_total[j][i][1], 'o', color=colors[j])
                plt.plot(i, momentum_norm_total[first_rejected][i][1], 'o', color='red', markersize=15, label=samples_array[j])                    
plt.xlabel('Leapfrog Steps')
plt.ylabel('Momentum Norm')
#plt.legend(loc='upper right', ncol=2, fontsize='small')
plt.title('Momentum Norm for 2nd Layer Weights - ' + str(num_rejected) + ' Rejected')

# Plot Grad Norm for 2nd Layer Weights (only accepted samples)
plt.subplot(5, 2, 5)
handles = []
for j in range(num_samples):
    if j not in remove:
        if acceptance_list[j] == 1:  # Only plot if sample was accepted
            x_values = list(range(num_leapfrog_steps))
            y_values = [grad_list_total[j][i][1] for i in range(num_leapfrog_steps)]
            
            # Plot with lines
            if j != first_accepted:
                line, = plt.plot(x_values, y_values, '-', color=colors[j])
            else:
                line, = plt.plot(x_values, y_values, '-', color='red', linewidth=15, label=samples_array[j])
            handles.append(line)
# Create the legend for accepted samples only
accepted_labels = [samples_array[j] for j in range(num_samples) if acceptance_list[j] == 1]
#plt.legend(handles, accepted_labels, loc='upper right', ncol=2, fontsize='small')
plt.xlabel('Leapfrog Steps')
plt.ylabel('Grad Norm')
plt.title('Grad Norm for 2nd Layer Weights - ' + str(num_accepted) + ' Accepted')

# Plot Grad Norm for 2nd Layer Weights (only accepted samples)
plt.subplot(5, 2, 6)
handles = []
for j in range(num_samples):
    if j not in remove:
        if acceptance_list[j] == 0:  # Only plot if sample was accepted
            x_values = list(range(num_leapfrog_steps))
            y_values = [grad_list_total[j][i][1] for i in range(num_leapfrog_steps)]
            
            # Plot with lines
            if j != first_rejected:
                line, = plt.plot(x_values, y_values, '-', color=colors[j])
            else:
                line, = plt.plot(x_values, y_values, '-', color='red', linewidth=15,label=samples_array[j])
            handles.append(line)
accepted_labels = [samples_array[j] for j in range(num_samples) if acceptance_list[j] == 0]
#plt.legend(handles, accepted_labels, loc='upper right', ncol=2, fontsize='small')
plt.xlabel('Leapfrog Steps')
plt.ylabel('Grad Norm')
plt.title('Grad Norm for 2nd Layer Weights - ' + str(num_rejected) + ' Rejected')

plt.subplot(5, 2, 7)
for j in range(num_samples):
    if j not in remove:
        if acceptance_list[j] == 1:  # Accepted samples
            if j != first_accepted:
                plt.plot(negative_log_posterior_list_total[j], color=colors[j], label=f'Sample {j+1}')
            else:
                plt.plot(negative_log_posterior_list_total[j], color='red', linewidth=15, label=f'Sample {j+1}')
plt.xlabel('Leapfrog Steps')
plt.ylabel('Negative Log Posterior')
plt.title('Negative Log Posterior - ' + str(num_accepted) + ' Accepted')
#plt.legend(loc='upper right', ncol=2, fontsize='small')

# Plot Negative Log Posterior for Rejected Samples
plt.subplot(5, 2, 8)
for j in range(num_samples):
    if j not in remove:
        if acceptance_list[j] == 0:  # Rejected samples
            if j != first_rejected:
                plt.plot(negative_log_posterior_list_total[j], color=colors[j], label=f'Sample {j+1}', alpha=0.5)
            else:
                plt.plot(negative_log_posterior_list_total[j], color='red', linewidth=15, label=f'Sample {j+1}')
plt.xlabel('Leapfrog Steps')
plt.ylabel('Negative Log Posterior')
plt.title('Negative Log Posterior - ' + str(num_rejected) + ' Rejected')
#plt.legend(loc='upper right', ncol=2, fontsize='small')

plt.subplot(5, 2, 9)
plt.plot(grad_list_total[first_accepted])
plt.xlabel('Leapfrog Steps')
plt.ylabel('Grad Norm')
plt.title('Gradient Norm')

plt.subplot(5, 2, 10)
plt.plot(momentum_norm_total[first_accepted])
plt.xlabel('Leapfrog Steps')
plt.ylabel('Momentum Norm')
plt.title('Momentum Norm')

plt.tight_layout()
plt.savefig('../../Results/HMC/BPINN_leapfrog_' + str(num_leapfrog_steps) +'_cleaned.png')
plt.show()