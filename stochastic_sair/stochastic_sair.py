# stochastic_sair/stochastic_sair.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse
import json
import os

def run_stochastic_sair(config):
    N = config.get('N', 1000)
    initial_infected = config.get('initial_infected', 10)
    initial_asymptomatic = config.get('initial_asymptomatic', 5)
    S0 = N - (initial_infected + initial_asymptomatic)
    I0 = initial_infected
    A0 = initial_asymptomatic
    R0 = 0
    dt = config.get('dt', 0.1)
    lambda_s = config.get('lambda_s', 0.3)
    lambda_i = config.get('lambda_i', 0.1)
    lambda_a = config.get('lambda_a', 0.2)
    gamma = config.get('gamma', 0.3)
    t_max = config.get('t_max', 200)
    n_sim = config.get('n_sim', 100)

    time = np.arange(0, t_max + dt, dt)
    num_time_steps = len(time)

    S_sum = np.zeros(num_time_steps)
    I_sum = np.zeros(num_time_steps)
    A_sum = np.zeros(num_time_steps)
    R_sum = np.zeros(num_time_steps)

    for simulation in range(n_sim):
        S = np.zeros(num_time_steps)
        I = np.zeros(num_time_steps)
        A = np.zeros(num_time_steps)
        R = np.zeros(num_time_steps)

        S[0] = S0
        I[0] = I0
        A[0] = A0
        R[0] = R0

        population = np.array([0] * int(S0) + [1] * int(I0) + [2] * int(A0) + [3] * int(R0))
        np.random.shuffle(population)

        S_count = S0
        I_count = I0
        A_count = A0
        R_count = R0

        for t_idx in range(1, num_time_steps):
            for i in range(N):
                if population[i] == 0:
                    prob_infection = (lambda_s / N) * (I_count + A_count) * dt
                    if np.random.random() < prob_infection:
                        if np.random.random() <= gamma:
                            population[i] = 2
                            S_count -= 1
                            A_count += 1
                        else:
                            population[i] = 1
                            S_count -= 1
                            I_count += 1
                elif population[i] == 1:
                    if np.random.random() < (lambda_i * dt):
                        population[i] = 3
                        I_count -= 1
                        R_count += 1
                elif population[i] == 2:
                    if np.random.random() < (lambda_a * dt):
                        population[i] = 3
                        A_count -= 1
                        R_count += 1

            S[t_idx] = S_count
            I[t_idx] = I_count
            A[t_idx] = A_count
            R[t_idx] = R_count

            if R_count >= N:
                S[t_idx+1:] = S_count
                I[t_idx+1:] = I_count
                A[t_idx+1:] = A_count
                R[t_idx+1:] = R_count
                break

        S_sum += S
        I_sum += I
        A_sum += A
        R_sum += R

    S_avg = S_sum / n_sim
    I_avg = I_sum / n_sim
    A_avg = A_sum / n_sim
    R_avg = R_sum / n_sim

    return time, S_avg, A_avg, I_avg, R_avg

def main():
    parser = argparse.ArgumentParser(description='Stochastic SAIR Model Simulation')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    time, S_avg, A_avg, I_avg, R_avg = run_stochastic_sair(config)

    # Prepare for plotting
    plt.figure(figsize=(11, 6.5))
    plt.plot(time, S_avg, label='Average Susceptible', color='blue')
    plt.plot(time, I_avg, label='Average Infected', color='red')
    plt.plot(time, A_avg, label='Average Asymptomatic', color='purple')
    plt.plot(time, R_avg, label='Average Recovered', color='green')

    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title(f'Stochastic SAIR Simulations ({config.get("n_sim", 100)} simulations)')
    plt.legend()
    plt.grid(True)

    # Save the plot
    os.makedirs(config.get('output_dir', 'results'), exist_ok=True)
    plot_path = os.path.join(config.get('output_dir', 'results'), config.get('result_filename', 'stochastic_sair_plot.png'))
    plt.savefig(plot_path)
    plt.close()

    print(f"Simulation complete. Plot saved to {plot_path}")

if __name__ == "__main__":
    main()