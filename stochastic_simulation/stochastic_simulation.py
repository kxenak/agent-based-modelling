# stochastic_simulation/stochastic_simulation.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse
import json
import os

def run_stochastic_sir(config):
    N = config.get('N', 1000)
    S0 = config.get('S0', N - 10)
    I0 = config.get('I0', 10)
    R0 = config.get('R0', 0)
    dt = config.get('dt', 0.1)
    days = config.get('days', 100)
    lambda_s = config.get('lambda_s', 0.3)
    lambda_i = config.get('lambda_i', 0.1)
    num_simulations = config.get('num_simulations', 10)
    output_dir = config.get('output_dir', './results')
    result_filename = config.get('result_filename', 'stochastic_sir_average_plot.png')

    time = np.arange(0, days + dt, dt)
    num_time_steps = len(time)
    S_sum = np.zeros(num_time_steps)
    I_sum = np.zeros(num_time_steps)
    R_sum = np.zeros(num_time_steps)

    for sim in range(num_simulations):
        S = np.zeros(num_time_steps)
        I = np.zeros(num_time_steps)
        R = np.zeros(num_time_steps)

        S_count = S0
        I_count = I0
        R_count = R0

        population = np.array([0] * int(S0) + [1] * int(I0) + [2] * int(R0))
        np.random.shuffle(population)

        for t in range(num_time_steps):
            for i in range(N):
                if population[i] == 0:
                    if np.random.random() < (lambda_s * I_count / N * dt):
                        population[i] = 1
                        S_count -= 1
                        I_count += 1
                elif population[i] == 1:
                    if np.random.random() < (lambda_i * dt):
                        population[i] = 2
                        I_count -= 1
                        R_count += 1

            S[t] = S_count
            I[t] = I_count
            R[t] = R_count

            if R_count == N:
                S[t+1:] = S_count
                I[t+1:] = I_count
                R[t+1:] = R_count
                break

        S_sum += S
        I_sum += I
        R_sum += R

    S_avg = S_sum / num_simulations
    I_avg = I_sum / num_simulations
    R_avg = R_sum / num_simulations

    return time, S_avg, I_avg, R_avg

if __name__ == "__main__":
    # Existing code to parse arguments and call run_stochastic_sir
    parser = argparse.ArgumentParser(description='Stochastic SIR Model Simulation with Averaging')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    time, S_avg, I_avg, R_avg = run_stochastic_sir(config)

    # Plotting
    plt.figure(figsize=(11, 6.5))
    plt.plot(time, S_avg, label='Average Susceptible', linewidth=2.5, color='blue')
    plt.plot(time, I_avg, label='Average Infected', linewidth=2.5, color='red')
    plt.plot(time, R_avg, label='Average Recovered', linewidth=2.5, color='green')

    plt.xlabel('Time (Days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('Stochastic SIR Simulations (Average)')

    # Ensure output directory exists
    os.makedirs(config.get('output_dir', './results'), exist_ok=True)

    # Save the plot
    plot_path = os.path.join(config.get('output_dir', './results'), config.get('result_filename', 'stochastic_sir_average_plot.png'))
    plt.savefig(plot_path)
    plt.close()