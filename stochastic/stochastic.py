import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os

parser = argparse.ArgumentParser(description='Stochastic SIR Model Simulation')
parser.add_argument('--config', required=True, help='Path to configuration JSON file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

# Extract params
N = config.get('N', 1000)                                  # Total population
S0 = config.get('S0', N - 10)                              # Initial susceptible
I0 = config.get('I0', 10)                                  # Initial infected
R0 = config.get('R0', 0)                                   # Initial recovered
dt = config.get('dt', 0.1)                                 # Time step
days = config.get('days', 100)                             # Total days
lambda_s = config.get('lambda_s', 0.3)                     # Transmission rate
lambda_i = config.get('lambda_i', 0.1)                     # Recovery rate
output_dir = config.get('output_dir', './results')         # Output directory
result_filename = config.get('result_filename', 'stochastic_sir_plot.png')  # Output file name

time = np.arange(0, days, dt)                              # Time array

# Initialize arrays
S = np.zeros(len(time))                                    # Susceptible count over time
I = np.zeros(len(time))                                    # Infected count over time
R = np.zeros(len(time))                                    # Recovered count over time

# Set initial counts
S_count = S0
I_count = I0
R_count = R0

# initial population states
def assign_population(N, S0, I0, R0):
    population = np.array([0] * S0 + [1] * I0 + [2] * R0)  # 0: S, 1: I, 2: R
    np.random.shuffle(population)                          # Shuffle population
    return population

population = assign_population(N, S0, I0, R0)

# Simulation loop
for t in range(len(time)):
    for i in range(N):
        if population[i] == 0:                             # Susceptible individual
            if np.random.random() < (lambda_s * I_count / N * dt):
                population[i] = 1                          # Becomes infected
                S_count -= 1
                I_count += 1
        elif population[i] == 1:                           # Infected individual
            if np.random.random() < (lambda_i * dt):
                population[i] = 2                          # Becomes recovered
                I_count -= 1
                R_count += 1

    # Update counts
    S[t] = S_count
    I[t] = I_count
    R[t] = R_count

    if R_count == N: # Early exit if everyone is recovered
        S = S[:t+1]                                        
        I = I[:t+1]
        R = R[:t+1]
        time = time[:t+1]
        break

# Plots
plt.figure(figsize=(11, 6.5))
plt.plot(time, S, label='Susceptible')
plt.plot(time, I, label='Infected')
plt.plot(time, R, label='Recovered')
plt.xlabel('Time (Days)')
plt.ylabel('Population')
plt.legend()
plt.title('Stochastic SIR Model Simulation')

os.makedirs(output_dir, exist_ok=True)

# Save the plot
plot_path = os.path.join(output_dir, result_filename)
plt.savefig(plot_path)