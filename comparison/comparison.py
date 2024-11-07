import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Common parameters
N = 1000                  # Total population
T = 100                   # Total time
dt = 0.1                  # Time step
num_simulations = 100     # Number of simulations for stochastic models

time = np.arange(0, T + dt, dt)

### SIR Model ###

# Parameters for SIR model
I0_sir = 10               # Initial infected
R0_sir = 0                # Initial recovered
lambda_s_sir = 0.3        # Transmission rate
lambda_i_sir = 0.1        # Recovery rate

# Deterministic SIR Model
def run_deterministic_sir():
    S0 = N - I0_sir - R0_sir
    S = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))
    S[0] = S0
    I[0] = I0_sir
    R[0] = R0_sir

    for t in range(1, len(time)):
        S_prev = S[t-1]
        I_prev = I[t-1]
        R_prev = R[t-1]

        dS = -lambda_s_sir * S_prev * I_prev / N
        dI = lambda_s_sir * S_prev * I_prev / N - lambda_i_sir * I_prev
        dR = lambda_i_sir * I_prev

        S[t] = S_prev + dS * dt
        I[t] = I_prev + dI * dt
        R[t] = R_prev + dR * dt

    return S, I, R

# Stochastic SIR Model
def run_stochastic_sir():
    S_samples = np.zeros((num_simulations, len(time)))
    I_samples = np.zeros((num_simulations, len(time)))
    R_samples = np.zeros((num_simulations, len(time)))

    for sim in range(num_simulations):
        S = np.zeros(len(time))
        I = np.zeros(len(time))
        R = np.zeros(len(time))
        S[0] = N - I0_sir - R0_sir
        I[0] = I0_sir
        R[0] = R0_sir

        for t in range(1, len(time)):
            S_prev = S[t-1]
            I_prev = I[t-1]
            R_prev = R[t-1]

            # Infection probability
            p_infection = 1 - np.exp(-lambda_s_sir * I_prev / N * dt)
            new_infections = np.random.binomial(int(S_prev), p_infection)

            # Recovery probability
            p_recovery = 1 - np.exp(-lambda_i_sir * dt)
            new_recoveries = np.random.binomial(int(I_prev), p_recovery)

            # Update compartments
            S[t] = S_prev - new_infections
            I[t] = I_prev + new_infections - new_recoveries
            R[t] = R_prev + new_recoveries

        S_samples[sim] = S
        I_samples[sim] = I
        R_samples[sim] = R

    # Average over simulations
    S_avg = np.mean(S_samples, axis=0)
    I_avg = np.mean(I_samples, axis=0)
    R_avg = np.mean(R_samples, axis=0)

    return S_avg, I_avg, R_avg

# Run SIR simulations
S_det_sir, I_det_sir, R_det_sir = run_deterministic_sir()
S_stoch_sir, I_stoch_sir, R_stoch_sir = run_stochastic_sir()

# Plotting SIR results
plt.figure(figsize=(10, 6))
plt.plot(time, S_det_sir, label='Deterministic Susceptible', color='blue')
plt.plot(time, I_det_sir, label='Deterministic Infected', color='red')
plt.plot(time, R_det_sir, label='Deterministic Recovered', color='green')
plt.plot(time, S_stoch_sir, '--', label='Stochastic Avg Susceptible', color='blue')
plt.plot(time, I_stoch_sir, '--', label='Stochastic Avg Infected', color='red')
plt.plot(time, R_stoch_sir, '--', label='Stochastic Avg Recovered', color='green')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Deterministic vs Stochastic SIR Model')
plt.show()

### SAIR Model ###

# Parameters for SAIR model
I0_sair = 10              # Initial infected
A0_sair = 5               # Initial asymptomatic
R0_sair = 0               # Initial recovered
lambda_s_sair = 0.3       # Transmission rate
lambda_i_sair = 0.1       # Recovery rate for infected
lambda_a_sair = 0.2       # Recovery rate for asymptomatic
gamma_sair = 0.3          # Proportion asymptomatic

# Deterministic SAIR Model
def run_deterministic_sair():
    S0 = N - I0_sair - A0_sair - R0_sair
    S = np.zeros(len(time))
    A = np.zeros(len(time))
    I = np.zeros(len(time))
    R = np.zeros(len(time))
    S[0] = S0
    A[0] = A0_sair
    I[0] = I0_sair
    R[0] = R0_sair

    for t in range(1, len(time)):
        S_prev = S[t-1]
        A_prev = A[t-1]
        I_prev = I[t-1]
        R_prev = R[t-1]

        infection_rate = lambda_s_sair * S_prev * (I_prev + A_prev) / N
        dS = -infection_rate
        dA = gamma_sair * infection_rate - lambda_a_sair * A_prev
        dI = (1 - gamma_sair) * infection_rate - lambda_i_sair * I_prev
        dR = lambda_a_sair * A_prev + lambda_i_sair * I_prev

        S[t] = S_prev + dS * dt
        A[t] = A_prev + dA * dt
        I[t] = I_prev + dI * dt
        R[t] = R_prev + dR * dt

    return S, A, I, R

# Stochastic SAIR Model
def run_stochastic_sair():
    S_samples = np.zeros((num_simulations, len(time)))
    A_samples = np.zeros((num_simulations, len(time)))
    I_samples = np.zeros((num_simulations, len(time)))
    R_samples = np.zeros((num_simulations, len(time)))

    for sim in range(num_simulations):
        S = np.zeros(len(time))
        A = np.zeros(len(time))
        I = np.zeros(len(time))
        R = np.zeros(len(time))
        S[0] = N - I0_sair - A0_sair - R0_sair
        A[0] = A0_sair
        I[0] = I0_sair
        R[0] = R0_sair

        for t in range(1, len(time)):
            S_prev = S[t-1]
            A_prev = A[t-1]
            I_prev = I[t-1]
            R_prev = R[t-1]

            # Infection probability
            p_infection = 1 - np.exp(-lambda_s_sair * (I_prev + A_prev) / N * dt)
            new_infections = np.random.binomial(int(S_prev), p_infection)

            # Asymptomatic and symptomatic infections
            new_asymptomatic = np.random.binomial(new_infections, gamma_sair)
            new_infected = new_infections - new_asymptomatic

            # Recovery probabilities
            p_recovery_A = 1 - np.exp(-lambda_a_sair * dt)
            p_recovery_I = 1 - np.exp(-lambda_i_sair * dt)
            new_recoveries_A = np.random.binomial(int(A_prev), p_recovery_A)
            new_recoveries_I = np.random.binomial(int(I_prev), p_recovery_I)

            # Update compartments
            S[t] = S_prev - new_infections
            A[t] = A_prev + new_asymptomatic - new_recoveries_A
            I[t] = I_prev + new_infected - new_recoveries_I
            R[t] = R_prev + new_recoveries_A + new_recoveries_I

        S_samples[sim] = S
        A_samples[sim] = A
        I_samples[sim] = I
        R_samples[sim] = R

    # Average over simulations
    S_avg = np.mean(S_samples, axis=0)
    A_avg = np.mean(A_samples, axis=0)
    I_avg = np.mean(I_samples, axis=0)
    R_avg = np.mean(R_samples, axis=0)

    return S_avg, A_avg, I_avg, R_avg

# Run SAIR simulations
S_det_sair, A_det_sair, I_det_sair, R_det_sair = run_deterministic_sair()
S_stoch_sair, A_stoch_sair, I_stoch_sair, R_stoch_sair = run_stochastic_sair()

# Plotting SAIR results
plt.figure(figsize=(10, 6))
plt.plot(time, S_det_sair, label='Deterministic Susceptible', color='blue')
plt.plot(time, A_det_sair, label='Deterministic Asymptomatic', color='purple')
plt.plot(time, I_det_sair, label='Deterministic Infected', color='red')
plt.plot(time, R_det_sair, label='Deterministic Recovered', color='green')
plt.plot(time, S_stoch_sair, '--', label='Stochastic Avg Susceptible', color='blue')
plt.plot(time, A_stoch_sair, '--', label='Stochastic Avg Asymptomatic', color='purple')
plt.plot(time, I_stoch_sair, '--', label='Stochastic Avg Infected', color='red')
plt.plot(time, R_stoch_sair, '--', label='Stochastic Avg Recovered', color='green')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('Deterministic vs Stochastic SAIR Model')
plt.show()