#=

This is the general script for the PINN.
This will be agnostic to architecture and size of the neural network (right now it is only for feedforeward)

Instead of the neural network approximating the solution u(x) directly, it learns
the optimal coefficients of a truncated power series that solves the ODE.

The process involves:
1. Defining the ODE and its boundary conditions.
2. Setting up a neural network that outputs a vector of coefficients.
3. Creating a loss function that measures how poorly the power series (built from the NN's coefficients)
   satisfies the ODE and boundary conditions.
4. Using an optimization algorithm (Adam) to train the network's parameters to minimize this loss.
5. Plotting the results to see how well our solution approximates the true, analytic solution.

=#

# ---------------------------------------------------------------------------
# Step 1: Import necessary libraries
# ---------------------------------------------------------------------------

module PINN

using Lux, ModelingToolkit
using Optimization, OptimizationOptimJL, OptimizationOptimisers
using Zygote
using ComponentArrays
import IntervalSets: Interval
using Plots, ProgressMeter
import Random
using TaylorSeries
using CSV
using DataFrames

include("../utils/ProgressBar.jl")
using .ProgressBar

include("../utils/loss_functions.jl")
using .loss_functions

include("../utils/helper_funcs.jl")
using .helper_funcs

# Ensure a "data" directory exists for saving plots.
isdir("data") || mkpath("data")

# Define the floating point type to use throughout the script (e.g., Float32).
# Using Float32 is standard for neural networks as it's computationally faster.
F = Float32

# ---------------------------------------------------------------------------
# Step 2: Define the PINN Settings Structure
# ---------------------------------------------------------------------------

struct PINNSettings
  neuron_num::Int
  seed::Int
  ode_matrices::Dict{Any,Any} # from the specific training run that is specified by the run number
  maxiters_adam::Int
  maxiters_lbfgs::Int
  n_terms_for_power_series::Int # The degree of the highest power term in the series.
  num_supervised::Int # The number of coefficients we will supervise during training.
  num_points::Int # number of points evaluated
  x_left::Float32 # left boundary 
  x_right::Float32 # right boundary 
  supervised_weight::Float32
  bc_weight::Float32 # for now we are going to test the two of these to zero
  pde_weight::Float32
  xs::Any
end

# We need to have a parameter for the PINN to allow us to swap architectures easily
# ---------------------------------------------------------------------------
# Step 3: Define the Mathematical Problem (The ODE)
# ---------------------------------------------------------------------------

# Using ModelingToolkit, we define the independent variable `x` and the dependent variable `u(x)`.
@parameters x
@variables u(..)

# Differential operator
Dx = Differential(x)

# Define the ordinary differential equation: u'(x) + u(x) = 0
# equation = Dx(u(x)) + u(x) ~ 0

# Define the domain over which the ODE is valid.
# domains = [x ∈ Interval(x_left, x_right)]

# ---------------------------------------------------------------------------
# Step 4: Setup the Power Series and Neural Network
# ---------------------------------------------------------------------------

# We will approximate the solution u(x) with a truncated power series of degree N.
N = 21 # The degree of the highest power term in the series.

# Pre-calculate factorials (0!, 1!, ..., N!) for use in the series.

num_supervised = 21 # The number of coefficients we will supervise during training.

# Create a set of points inside the domain to enforce the ODE. These are called "collocation points".
num_points = 10


# Domain boundaries
x_left = F(0.0)  # Left boundary of the domain
x_right = F(1.0) # Right boundary of the domain

# Define a weight for the boundary condition, surpivesed coefficients, and the pde
# supervised_weight = F(1.0)  # Weight for the supervised loss term in the total loss function.
# bc_weight = F(1.0) # for now we are going to test the two of these to zero
# pde_weight = F(1.0)

# ---------------------------------------------------------------------------
# Step 5: Initialize Neural Network with Settings
# ---------------------------------------------------------------------------

function initialize_network(settings::PINNSettings)
  # Find the maximum matrix dimensions for input layer size
  # alpha_matrix = eval(Meta.parse(alpha_matrix_key)) # convert to string

  max_input_size = maximum(prod(size(alpha_matrix_key)) for (alpha_matrix_key, series_coeffs) in settings.ode_matrices) # AHHHHH! what a messs

  coeff_net = Lux.Chain(
    Lux.Dense(max_input_size, settings.neuron_num, σ),      # First hidden layer or the input layer? for now this is the input layer
    Lux.Dense(settings.neuron_num, settings.neuron_num, σ), # Second hidden layer
    Lux.Dense(settings.neuron_num, settings.neuron_num, σ), # Third hidden layer ? 
    Lux.Dense(settings.neuron_num, settings.n_terms_for_power_series + 1)              # N+1? # Output layer with N+1 coefficients
  )

  # Initialize the network's parameters with the specified seed
  rng = Random.default_rng()
  Random.seed!(rng, settings.seed)
  p_init, st = Lux.setup(rng, coeff_net) # this sets the parameters of the Neural Network

  # Wrap the initial parameters in a ComponentArray
  p_init_ca = ComponentArray(p_init)

  return coeff_net, p_init_ca, st
end

# ---------------------------------------------------------------------------
# Step 6: Define the Loss Function
# ---------------------------------------------------------------------------

function loss_fn(p_net, data, coeff_net, st, ode_matrix_flat, boundary_condition, settings::PINNSettings, data_dir)
  # Run the network to get the current vector of power series coefficients
  a_vec = first(coeff_net(ode_matrix_flat, p_net, st))[:, 1]

  # Define the approximate solution and its derivatives using the coefficients
  # u_approx(x) = sum(a_vec[i] * x^(i - 1) for i in 1:N+1)
  # Du_approx(x) = sum(a_vec[i] * x^(i - 2) for i in 2:N+1) # First derivative

  # For the ODE: a*y' + b*y = 0
  # This can be written as: b*y + a*y' = 0
  # So ode_matrix_flat should be [b, a]

  loss_func_settings = LossFunctionSettings(
    a_vec,
    settings.n_terms_for_power_series,
    ode_matrix_flat,
    settings.x_left,
    boundary_condition,
    settings.xs,
    settings.num_points,
    settings.num_supervised,
    data,
  )

  # Define u_approx (the 0th derivative, which is y) with coefficient b
  # u_approx(x) = sum(a_vec[i] * x^(i - 1) for i in 1:settings.n_terms_for_power_series+1)

  # Calculate the PDE loss (residual of the ODE)
  # loss_pde = sum(abs2, ode_residual(xi, ode_matrix_flat, a_vec, settings.n_terms_for_power_series) for xi in settings.xs) / settings.num_points 
  loss_pde = generate_loss_pde_value(loss_func_settings)

  # Calculate the loss from the boundary conditions
  # loss_bc = abs2(u_approx(settings.x_left) - F(boundary_condition))
  loss_bc = generate_loss_bc_value(loss_func_settings)

  # Calculate supervised loss using the plugboard coefficients
  # loss_supervised = sum(abs2, a_vec[1:settings.num_supervised] - data) / settings.num_supervised
  loss_supervised = generate_loss_supervised_value(loss_func_settings)

  # The total loss is a weighted sum of the components
  # We want to return the global loss but we want to return the individual loss as well
  return loss_pde * settings.pde_weight + settings.bc_weight * loss_bc + settings.supervised_weight * loss_supervised, loss_bc, loss_pde, loss_supervised
end

# ---------------------------------------------------------------------------
# Step 7: Global Loss Function
# ---------------------------------------------------------------------------

function global_loss(p_net, settings::PINNSettings, coeff_net, st, csv_file)
  total_loss = F(0.0)
  total_local_loss_bc = F(0.0)
  total_local_loss_pde = F(0.0)
  total_local_loss_supervised = F(0.0)
  num_of_training_examples = length(settings.ode_matrices)

  # println(settings.ode_matrices) # print out the ode_matrices dictionary
  # println("The global loss is globally lossing...")
  for (alpha_matrix_key, series_coeffs) in settings.ode_matrices
    # println("The current  ODE I am calculating the loss for right now: ", alpha_matrix_key)
    # println("The local loss is locally lossing...")
    # alpha_matrix = eval(Meta.parse(alpha_matrix_key)) # convert from string to matrix 
    matrix_flat = vec(alpha_matrix_key)  # Flatten to a column vector
    boundary_condition = series_coeffs[1]  # copy this
    local_loss, local_loss_bc, local_loss_pde, local_loss_supervised = loss_fn(p_net, series_coeffs, coeff_net, st, matrix_flat, boundary_condition, settings::PINNSettings, csv_file) # calculate the local loss
    # println(local_loss)
    total_loss += local_loss # add up the local loss to find the global loss
    total_local_loss_bc += local_loss_bc
    total_local_loss_pde += local_loss_pde
    total_local_loss_supervised += local_loss_supervised
  end

  normalized_loss = total_loss / num_of_training_examples
  normalized_loss_bc = total_local_loss_bc / num_of_training_examples
  normalized_loss_pde = total_local_loss_pde / num_of_training_examples
  normalized_loss_supervised = total_local_loss_supervised / num_of_training_examples

  Zygote.ignore() do
    buffer_loss_values(  # ← Changed from create_csv_file_for_loss
      total_loss=normalized_loss,
      total_loss_bc=normalized_loss_bc,
      total_loss_pde=normalized_loss_pde,
      total_loss_supervised=normalized_loss_supervised
    )
  end

  # println(total_loss)
  return normalized_loss
end

# ---------------------------------------------------------------------------
# Step 8: Training Function
# ---------------------------------------------------------------------------

#=
We train the PINN on the training dataset and return the network
=#

function train_pinn(settings::PINNSettings, csv_file)
  # Initialize network
  coeff_net, p_init_ca, st = initialize_network(settings)

  initialize_loss_buffer()
  # global_loss_tuple = Tuple{Int64, Float64, Float64, Float64, Float64}[] # this will store the global loss per iteration milestone
  # Create wrapper function for optimization

  function loss_wrapper(p_net, _)
    return global_loss(p_net, settings, coeff_net, st, csv_file)
  end

  # ---------------- Stage 1: ADAM ----------------
  println("Starting Adam training...")
  p_one = ProgressBar.ProgressBarSettings(settings.maxiters_adam, "Adam Training...") # the progress bar has not been called...
  callback_one = ProgressBar.Bar(p_one)

  # Define the optimization problem
  adtype = Optimization.AutoZygote()
  optfun = OptimizationFunction(loss_wrapper, adtype)
  prob = OptimizationProblem(optfun, p_init_ca)

  res = solve(prob,
    OptimizationOptimisers.Adam(F(1e-3));
    callback=callback_one, # this is for the progress bar 
    maxiters=settings.maxiters_adam)

  # ---------------- Stage 2: LBFGS ----------------
  println("Starting LBFGS fine-tuning...")
  p_two = ProgressBar.ProgressBarSettings(settings.maxiters_lbfgs, "LBFGS fine-tune...")
  callback_two = ProgressBar.Bar(p_two)

  prob2 = remake(prob; u0=res.u)
  res = solve(prob2,
    OptimizationOptimJL.LBFGS();
    callback=callback_two,
    maxiters=settings.maxiters_lbfgs)

  # Extract final trained parameters
  p_trained = res.u

  println("\nTraining complete.")

  write_buffer_to_csv(csv_file) # write from buffer to csv

  return p_trained, coeff_net, st
end

# ---------------------------------------------------------------------------
# Step 9: Evaluation and Analysis Functions
# ---------------------------------------------------------------------------

# This code is the true solution for the ODE
# analytic_sol_func(x) = (pi * x * (-x + (pi^2) * (2x - 3) + 1) - sin(pi * x)) / (pi^3) # We replace with our training examples
# This is then represented as a TaylorSeries 

function evaluate_solution(settings::PINNSettings, p_trained, coeff_net, st, benchmark_dataset, data_directories)
  println(benchmark_dataset)
  converted_benchmark_dataset = convert_plugboard_keys(benchmark_dataset)
  fact = factorial.(big.(0:settings.n_terms_for_power_series)) # I am not considering this in the series. The PINN will guess the coefficients

  # We will update the error. For now we are only going to do ONE test set.
  loss = F(0.0)

  for (alpha_matrix_key, benchmark_series_coeffs) in converted_benchmark_dataset
    # we need to compute the loss from the PINNs guess and the real function
    # We will then use this for our contour maps
    matrix_flat = vec(alpha_matrix_key)  # Flatten to a column vector
    boundary_condition = benchmark_series_coeffs[1]
    benchmark_loss, _, _, _ = loss_fn(p_trained, benchmark_series_coeffs, coeff_net, st, matrix_flat, boundary_condition, settings::PINNSettings, data_directories[5])
    loss += benchmark_loss

    a_learned = first(coeff_net(matrix_flat, p_trained, st))[:, 1] # extract learned coefficients

    u_real_func(x) = sum(benchmark_series_coeffs[i] * x^(i - 1) / fact[i] for i in 1:settings.n_terms_for_power_series)
    # this is the taylor series that is predicted by the PINN

    u_predict_func(x) = sum(a_learned[i] * x^(i - 1) / fact[i] for i in 1:settings.n_terms_for_power_series)

    # Generate plotting points
    x_plot = settings.x_left:F(0.01):settings.x_right
    # It makes sense that this has to be replaced because this is used for plotting the error as well
    u_real = u_real_func.(x_plot)
    u_predict = u_predict_func.(x_plot)
    # ============================================================================
    # FIGURE 1: Function Analysis (u(x) comparison and error)
    # ============================================================================

    # Plot 1a: Compare analytic solution vs PINN prediction
    function_comparison = plot(x_plot, u_real,
      label="Analytic Solution",
      linestyle=:dash,
      linewidth=3,
      title="ODE Solution Comparison",
      xlabel="x",
      ylabel="u(x)",
      legend=:best)

    plot!(function_comparison, x_plot, u_predict,
      label="PINN Power Series",
      linewidth=2)

    # Plot 1b: Function error
    function_error_data = max.(abs.(u_real .- u_predict), F(1e-20))
    function_error_plot = plot(x_plot, function_error_data,
      title="Absolute Error of Solution",
      label="|Analytic - Predicted|",
      yscale=:log10,
      xlabel="x",
      ylabel="Error",
      linewidth=2)

    # Combine into Figure 1
    figure_one = plot(function_comparison, function_error_plot,
      layout=(2, 1),
      size=(800, 800))

    savefig(figure_one, data_directories[1])

    # ============================================================================
    # FIGURE 2: Coefficient Analysis (comparison and error)
    # ============================================================================

    # Prepare data
    n_length_benchmark = length(benchmark_series_coeffs)
    indices = 1:n_length_benchmark

    # Plot 2a: Compare benchmark coefficients vs learned coefficients
    coefficient_comparison = plot(indices, benchmark_series_coeffs,
      title="Coefficient Comparison",
      label="Benchmark",
      xlabel="Coefficient Index",
      ylabel="Coefficient Value",
      legend=:best)

    plot!(coefficient_comparison, indices, a_learned[1:n_length_benchmark],
      label="PINN")

    # Plot 2b: Coefficient error
    coefficient_error_data = max.(abs.(benchmark_series_coeffs .- a_learned[1:n_length_benchmark]), 1e-20)
    coefficient_error_plot = plot(indices, coefficient_error_data,
      title="Absolute Error of Coefficients",
      label="|Benchmark - PINN|",
      yscale=:log10,
      xlabel="Coefficient Index",
      ylabel="Absolute Error",
      linewidth=2)

    # Combine into Figure 2
    figure_two = plot(coefficient_comparison, coefficient_error_plot,
      layout=(2, 1),
      size=(800, 800))
    savefig(figure_two, data_directories[2])


    # Read the CSV file
    df = CSV.read(data_directories[6], DataFrame)

    # Helper function to extract values for a specific loss type
    function get_loss_values(df, loss_type_name)
      row = df[df.loss_type.==loss_type_name, :]
      if nrow(row) == 0
        return Float32[]
      end
      return Vector{Float32}(collect(skipmissing(row[1, 2:end])))
    end

    # Extract all loss values
    total_loss = get_loss_values(df, "total_loss")
    total_loss_bc = get_loss_values(df, "total_loss_bc")
    total_loss_pde = get_loss_values(df, "total_loss_pde")
    total_loss_supervised = get_loss_values(df, "total_loss_supervised")

    #= THIS CODE IS COMMENTED OUT BECAUSE THE WAY WE HAVE IT IS THE CSV FILE IS UPDATED AT EACH CALL OF THE GLOBAL LOSS FUNCTION
    # Split into Adam and LBFGS phases
    split_point = min(settings.maxiters_adam, length(total_loss))

    total_loss_adam = total_loss[1:split_point]
    total_loss_lbfgs = total_loss[(split_point + 1):end]

    total_loss_bc_adam = total_loss_bc[1:split_point]
    total_loss_bc_lbfgs = total_loss_bc[(split_point + 1):end]

    total_loss_pde_adam = total_loss_pde[1:split_point]
    total_loss_pde_lbfgs = total_loss_pde[(split_point + 1):end]

    total_loss_supervised_adam = total_loss_supervised[1:split_point]
    total_loss_supervised_lbfgs = total_loss_supervised[(split_point + 1):end]
    =#

    # Adam plots (using row indices 1:1000)
    total_loss_plot = plot(
      1:length(total_loss),
      total_loss,
      title="Global Loss per Global Loss Call",
      xlabel="Loss Call",
      ylabel="Global Loss",
      yscale=:log10
    )

    total_bc_loss_plot = plot(
      1:length(total_loss_bc),
      total_loss_bc,
      title="Global Boundary Condition Loss per Global Loss Call",
      xlabel="Loss Call",
      ylabel="BC Loss", yscale=:log10)

    total_pde_loss_plot = plot(
      1:length(total_loss_pde),
      total_loss_pde,
      title="Global PDE Loss per Global Loss Call",
      xlabel="Loss Call",
      ylabel="PDE Loss",
      yscale=:log10
    )

    total_supervised_loss_plot = plot(
      1:length(total_loss_supervised),
      total_loss_supervised,
      title="Global Supervised Loss per Global Loss Call",
      xlabel="Loss Call",
      ylabel="Supervised Loss", yscale=:log10
    )

    iteration_plot = plot(
      total_loss_plot,
      total_bc_loss_plot,
      total_pde_loss_plot,
      total_supervised_loss_plot,
      layout=(4, 1),
      size=(1000, 1000),
      yscale=:log10
    )

    # Save plots
    savefig(iteration_plot, data_directories[5])

    println("\nPlots saved to 'data' directory.")

    println("PINN's guess for coefficients: ", a_learned)
    println("The REAL coefficients: ", benchmark_series_coeffs)
  end
  return loss
end

# ---------------------------------------------------------------------------
# Step 10: Export Functions
# ---------------------------------------------------------------------------

export PINNSettings, train_pinn, global_loss, evaluate_solution, initialize_network

end
