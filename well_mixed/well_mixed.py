# well_mixed/well_mixed.py

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import argparse
import json
import os

def run_deterministic_sir(config):
    # extract params
    lambda_s = config.get('lambda_s', 0.3)
    lambda_i = config.get('lambda_i', 0.2)
    N = config.get('N', 1000000)
    I0 = config.get('I0', 1000)
    R0 = config.get('R0', 0)
    T = config.get('T', 200)
    dt = config.get('dt', 0.1)
    output_dir = config.get('output_dir', '.')
    result_filename = config.get('result_filename', 'sir_model_plot.png')

    S0 = N - I0 - R0

    time = np.arange(0, T + dt, dt)
    S = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for t in range(1, len(time)):
        S[t] = S[t-1] + dt * (-lambda_s * S[t-1] * I[t-1] / N)
        I[t] = I[t-1] + dt * (lambda_s * S[t-1] * I[t-1] / N - lambda_i * I[t-1])
        R[t] = R[t-1] + dt * (lambda_i * I[t-1])

    return time, S, I, R

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SIR Model Simulation')
    parser.add_argument('--config', required=True, help='Path to configuration JSON file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    time, S, I, R = run_deterministic_sir(config)

    # plot
    plt.plot(time, S, label='Susceptible')
    plt.plot(time, I, label='Infected')
    plt.plot(time, R, label='Recovered')
    plt.xlabel('Time (Days)')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Well-Mixed Solution')

    os.makedirs(config.get('output_dir', '.'), exist_ok=True)

    # save plot
    plot_path = os.path.join(config.get('output_dir', '.'), config.get('result_filename', 'sir_model_plot.png'))
    plt.savefig(plot_path)
    plt.close()