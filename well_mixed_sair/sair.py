# well_mixed_sair/sair.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse
import json
import os

def run_deterministic_sair(config):
    # Model parameters
    lambda_s = config.get('lambda_s', 0.3)
    lambda_i = config.get('lambda_i', 0.1)
    lambda_a = config.get('lambda_a', 0.2)
    gamma = config.get('gamma', 0.3)
    initial_infected = config.get('initial_infected', 10)
    initial_asymptomatic = config.get('initial_asymptomatic', 5)
    N = config.get('N', 1000)

    t_max = config.get('t_max', 200)
    dt = config.get('dt', 0.1)
    t_values = np.arange(0, t_max + dt, dt)

    # Initializing arrays to store SAIR values
    S_det = np.zeros(len(t_values))
    A_det = np.zeros(len(t_values))
    I_det = np.zeros(len(t_values))
    R_det = np.zeros(len(t_values))

    # Initial conditions
    S_det[0] = N - (initial_infected + initial_asymptomatic)
    I_det[0] = initial_infected
    A_det[0] = initial_asymptomatic
    R_det[0] = 0

    # Euler's method to solve the differential equations
    for i in range(1, len(t_values)):
        dSdt = -((lambda_s / N)) * S_det[i-1] * (I_det[i-1] + A_det[i-1])
        dAdt = (gamma * (lambda_s / N) * S_det[i-1] * (I_det[i-1] + A_det[i-1])) - (lambda_a * A_det[i-1])
        dIdt = ((1 - gamma) * (lambda_s / N) * S_det[i-1] * (I_det[i-1] + A_det[i-1])) - lambda_i * I_det[i-1]
        dRdt = (lambda_i * I_det[i-1]) + (lambda_a * A_det[i-1])

        S_det[i] = S_det[i-1] + dSdt * dt
        A_det[i] = A_det[i-1] + dAdt * dt
        I_det[i] = I_det[i-1] + dAdt * dt
        R_det[i] = R_det[i-1] + dRdt * dt

    return t_values, S_det, A_det, I_det, R_det

def main():
    parser = argparse.ArgumentParser(description='Deterministic SAIR Model Simulation')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    t_values, S_det, A_det, I_det, R_det = run_deterministic_sair(config)

    # Create output directory if it doesn't exist
    os.makedirs(config.get('output_dir', 'results'), exist_ok=True)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_values, S_det, label='Susceptible')
    plt.plot(t_values, I_det, label='Infected')
    plt.plot(t_values, A_det, label='Asymptomatic')
    plt.plot(t_values, R_det, label='Recovered')
    plt.title('Deterministic SAIR Model')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_path = os.path.join(config.get('output_dir', 'results'), config.get('result_filename', 'sair_model_plot.png'))
    plt.savefig(plot_path)
    plt.close()

    print(f"Simulation complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    main()