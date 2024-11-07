using Random
using Plots
using Statistics
using JSON
using StatsBase      # For countmap function
using Printf
using Dates          # For timing functions

# Load configuration
config = JSON.parsefile("config.json")

# Extract parameters from config
lambda_s = config["lambda_s"]                               # Transmission rate
lambda_i = config["lambda_i"]                               # Recovery rate
N = config["N"]                                             # Total population
I0 = config["I0"]                                           # Initial infected
S0 = config["S0"]                                           # Initial susceptible
R0 = config["R0"]                                           # Initial recovered
days = config["days"]                                       # Total days
dt = config["dt"]                                           # Time step
quarantine_enabled = config["quarantine_enabled"]           # Quarantine flag
n_home = config["n_home"]                                   # Number of homes
n_work = config["n_work"]                                   # Number of workplaces
output_dir = config["output_dir"]                           # Output directory
result_filename = config["result_filename"]                 # Simulation plot filename
hist_home_filename = config["hist_home_filename"]           # Home occupancy histogram filename
hist_work_filename = config["hist_work_filename"]           # Work occupancy histogram filename
n_sim = config["n_sim"]                                     # Number of simulations

# Ensure output directory exists
isdir(output_dir) || mkdir(output_dir)                      # Create output directory if needed

Random.seed!(123)                                           # Set random seed

n_location = n_home + n_work                                # Total locations

@enum Status::UInt8 S=0 I=1 R=2                             # Health status
@enum Loc::UInt8 Home=0 Work=1                              # Location types

mutable struct Person
    status::Status
    schedule::NTuple{4, Loc}
    locations::Dict{Loc, Int}
    current_location_id::Int
    infectedby::Int
    quarantine::Bool
    home_location_id::Int
end

mutable struct Place
    infected::Int
    total::Int
    I_prob::Float64
    R_prob::Float64
    infectors::Vector{Int}
end

function assign_locations(n_home, n_work, n_location)
    home = rand(1:n_home)                                   # Assign home ID
    work = rand((n_home + 1):n_location)                    # Assign work ID
    Dict(Home => home, Work => work)                        # Return locations
end

function init_population(S0, I0, N)
    initial_population = Vector{Person}(undef, N)           # Initialize population array
    statuses = vcat(fill(S, S0), fill(I, I0))               # Initial statuses
    shuffle!(statuses)                                      # Shuffle statuses
    home_ids = zeros(Int, N)                                # Collect home IDs
    work_ids = zeros(Int, N)                                # Collect work IDs
    for i in 1:N
        locations = assign_locations(n_home, n_work, n_location)
        person = Person(
            statuses[i],
            (Home, Work, Work, Home),
            locations,
            0,
            0,
            false,
            locations[Home]
        )
        initial_population[i] = person
        home_ids[i] = locations[Home]                       # Store home ID
        work_ids[i] = locations[Work]                       # Store work ID
    end
    return initial_population, home_ids, work_ids
end

function init_city(n_location::Int)
    [Place(0, 0, 0.0, 0.0, Int[]) for _ in 1:n_location]    # Initialize city places
end

function calculate_prob(city, dt, lambda_s, lambda_i)
    for location in city
        I = location.infected
        N = location.total
        if N > 0
            location.I_prob = lambda_s * I / N * dt         # Infection probability
            location.R_prob = lambda_i * dt                 # Recovery probability
        else
            location.I_prob = 0.0
            location.R_prob = 0.0
        end
    end
    return city
end

function update_location(population, empty_city, to_index, dt, lambda_s, lambda_i)
    city = deepcopy(empty_city)                             # Copy empty city
    for (i, person) in enumerate(population)
        if quarantine_enabled && person.quarantine
            new_location_id = person.home_location_id       # Stay at home
        else
            to_where = person.schedule[to_index]            # Get next location
            new_location_id = person.locations[to_where]    # Get location ID
        end
        person.current_location_id = new_location_id
        city[new_location_id].total += 1                    # Increment total count
        if person.status == I
            city[new_location_id].infected += 1             # Increment infected count
            push!(city[new_location_id].infectors, i)       # Add infector index
        end
    end
    city = calculate_prob(city, dt, lambda_s, lambda_i)     # Calculate probabilities
    return city
end

function simulation(initial_population, empty_city, n_sim, dt, days, lambda_s, lambda_i)
    num_ticks = Int(days / dt) + 1                          # Total time steps
    S_sim = [zeros(Int, num_ticks) for _ in 1:n_sim]        # Susceptible counts
    I_sim = [zeros(Int, num_ticks) for _ in 1:n_sim]        # Infected counts
    R_sim = [zeros(Int, num_ticks) for _ in 1:n_sim]        # Recovered counts
    all_populations = Vector{Vector{Person}}(undef, n_sim)  # Store populations

    for simulation in 1:n_sim
        population = deepcopy(initial_population)           # Copy initial population

        for tick in 0:(num_ticks - 1)
            tuple_index = mod(tick, 4) + 1                  # Schedule index
            city = update_location(population, empty_city, tuple_index, dt, lambda_s, lambda_i)

            S_count, I_count, R_count = 0, 0, 0             # Reset counts
            for person in population
                if person.status == S
                    S_count += 1
                    if rand() < city[person.current_location_id].I_prob
                        person.status = I
                        person.infectedby = rand(city[person.current_location_id].infectors)
                        if quarantine_enabled
                            person.quarantine = true
                        end
                    end
                elseif person.status == I
                    I_count += 1
                    if rand() < city[person.current_location_id].R_prob
                        person.status = R
                        if quarantine_enabled
                            person.quarantine = false
                        end
                    end
                else
                    R_count += 1
                end
            end

            S_sim[simulation][tick + 1] = S_count           # Store S count
            I_sim[simulation][tick + 1] = I_count           # Store I count
            R_sim[simulation][tick + 1] = R_count           # Store R count
        end

        all_populations[simulation] = population            # Store population
    end

    return S_sim, I_sim, R_sim, all_populations
end

# Measure start time
start_time = now()                                          # Record start time

# Initialize population and city
initial_population, home_ids, work_ids = init_population(S0, I0, N)
empty_city = init_city(n_location)

# Run simulations
S_sim, I_sim, R_sim, all_populations = simulation(initial_population, empty_city, n_sim, dt, days, lambda_s, lambda_i)

# Measure end time
end_time = now()                                            # Record end time
run_time = end_time - start_time                            # Calculate run time

# Time points for plotting
time_points = 0:dt:days

# Calculate means
S_mean = vec(mean(hcat(S_sim...), dims=2))                  # Mean S
I_mean = vec(mean(hcat(I_sim...), dims=2))                  # Mean I
R_mean = vec(mean(hcat(R_sim...), dims=2))                  # Mean R

# Plotting SIR curves
plot(xlabel="Days", ylabel="Population", title="SIR Model Simulations (Network Structure)", legend=:outerright)

# Plot individual simulations
for i in 1:n_sim
    plot!(time_points, S_sim[i], alpha = 0.1, color = :blue, label = "")
    plot!(time_points, I_sim[i], alpha = 0.1, color = :red, label = "")
    plot!(time_points, R_sim[i], alpha = 0.1, color = :green, label = "")
end

# Plot mean curves
plot!(time_points, S_mean, linewidth=2, color=:blue, label="Susceptible (Mean)")
plot!(time_points, I_mean, linewidth=2, color=:red, label="Infected (Mean)")
plot!(time_points, R_mean, linewidth=2, color=:green, label="Recovered (Mean)")

# Save SIR plot
savefig(joinpath(output_dir, result_filename))              # Save simulation plot

# Histogram of home occupancy
home_occupancy = countmap(home_ids)                         # Count home occupants
occupancy_counts = collect(values(home_occupancy))          # Occupant counts
histogram_home = histogram(occupancy_counts, bins=maximum(occupancy_counts),
                           xlabel="Number of Occupants", ylabel="Number of Homes",
                           title="Home Occupancy Distribution")
savefig(histogram_home, joinpath(output_dir, hist_home_filename))  # Save home histogram

# Histogram of work occupancy
work_occupancy = countmap(work_ids)                         # Count work occupants
work_counts = collect(values(work_occupancy))               # Occupant counts
histogram_work = histogram(work_counts, bins=maximum(work_counts),
                           xlabel="Number of Occupants", ylabel="Number of Workplaces",
                           title="Work Occupancy Distribution")
savefig(histogram_work, joinpath(output_dir, hist_work_filename))  # Save work histogram

# Save run time to a text file
run_time_file = joinpath(output_dir, "run_times.txt")       # Run time file path
open(run_time_file, "a") do io
    println(io, "Run time: $(run_time) seconds")            # Append run time
end