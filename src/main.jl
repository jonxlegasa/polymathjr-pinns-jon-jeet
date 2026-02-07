using Dates
using JSON
using CSV
using DataFrames

# Include util functions
include("../utils/plugboard.jl")
using .Plugboard

include("../utils/ProgressBar.jl")
using .ProgressBar

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../utils/two_d_grid_search_hyperparameters.jl")
using .TwoDGridSearchOnWeights

include("../modelcode/PINN.jl")
using .PINN

include("../utils/training_schemes.jl")
using .training_schemes

#=
This function does the following:
Create training run directories
=#
function create_training_run_dirs(run_number::Int64, batch_size::Any)
  """
  Creates a training run directory and output file with specified naming convention.
  Args:
      run_number: The training run number (will be zero-padded to 2 digits)
      training_examples: Array of natural numbers representing training examples for each model
  """
  # Create data directory if it doesn't exist
  data_dir = "data"
  if !isdir(data_dir)
    mkdir(data_dir)
    @info "Created data directory: $data_dir"
  end

  # Format run number with zero padding (01, 02, 03, etc.)
  run_number_formatted = lpad(run_number, 2, '0')
  # Create training run directory
  training_run_dir = joinpath(data_dir, "training-run-$run_number_formatted")
  if !isdir(training_run_dir)
    mkdir(training_run_dir)
    @info "Created training run directory: $training_run_dir"
  end

  # Generate output file with training run information
  output_file = joinpath(training_run_dir, "training_info.txt")
  # Get current date and time
  current_datetime = now()

  # Write training run information to file
  open(output_file, "w") do file
    println(file, "Training Run Information")
    println(file, "="^30)
    println(file, "Training Run Number: $run_number_formatted")
    println(file, "Training Examples per Model: $batch_size")
    println(file, "Training Run Commenced: $current_datetime")
    println(file, "="^30)
  end

  @info "Training run $run_number_formatted setup complete" output_file

  return training_run_dir, output_file
end

# ---- Configuration ----
GENERATE_DATASET = true  # Set to false to skip dataset generation and use existing data
MODE = "SPECIFIC"         # "SPECIFIC" = benchmark on hardcoded test_matrix, "RANDOM" = random benchmark

# These are the training and benchmark directories
training_data_dir = "./data/training_dataset.json"
benchmark_data_dir = "./data/benchmark_dataset.json"

#=
This function initializes training run batches
and creates training and benchmark dataset
=#

function init_batches(batch_sizes::Array{Int})
  """
  Initializes batches by generating datasets for different batch sizes.
  Args:
      batch_sizes: Array of integers representing different batch sizes
  """

  benchmark_dataset_setting::Settings = Plugboard.Settings(2, 0, 1, benchmark_data_dir, 10)

  # generate training datasets and benchmarks 
  for (batch_index, k) in enumerate(batch_sizes)
    training_dataset_setting::Settings = Plugboard.Settings(2, 0, k, training_data_dir, 10)
    # set up plugboard for solutions to ay' + by = 0 where a,b != 0
    run_number_formatted = lpad(batch_index, 2, '0')

    @info "Generating datasets for batch $run_number_formatted" num_examples=k mode=MODE

    # Training data depends on MODE
    specific_matrix = [1; 6; 2;;]
    if MODE == "SPECIFIC"
      @warn "In $MODE mode. Generating specific training dataset for $specific_matrix"
      Plugboard.generate_specific_ode_dataset(training_dataset_setting, batch_index, specific_matrix)
    elseif MODE == "RANDOM"
      Plugboard.generate_random_ode_dataset(training_dataset_setting, batch_index)
    end

    # Benchmark always uses the specific test matrix for consistent evaluation
    Plugboard.generate_specific_ode_dataset(benchmark_dataset_setting, 1, specific_matrix)
  end
end

#= 
This function takes the batches and their sizes and runs
the PINN on the training dataset
=#

function run_training_sequence(batch_sizes::Array{Int})
  """
  Runs a sequence of training runs with different training example configurations.
  Args:
      batch_sizes: Array of integers representing different batch sizes
  """
  # Initialize all batches first (generate datasets via plugboard)
  if GENERATE_DATASET
    init_batches(batch_sizes)
  end

  # we only load the training data dir here
  training_dataset = JSON.parsefile(training_data_dir)
  benchmark_dataset = JSON.parsefile(benchmark_data_dir)

  F = Float32
  # We will approximate the solution u(x) with a truncated power series of degree N.
  # BS on pde_weight with supervised and bc fixed at 1.0

  #=
  binary_search_weights(
    training_dataset,
    :pde,
    (0, 100),
    20,
    fixed_weights = (supervised=supervised_weight, bc=bc_weight),
    num_supervised = num_supervised,
    N = N,
    x_left = x_left,
    x_right = x_right,
    xs = xs,
    base_data_dir = "data"
  )
  =#

  N = 10 # The degree of the highest power term in the series.

  num_supervised = 10 # The number of coefficients we will supervise during training.
  # Create a set of points inside the domain to enforce the ODE. These are called "collocation points".
  num_points = 10

  # Domain boundaries
  x_left = F(0.0)  # Left boundary of the domain
  x_right = F(1.0) # Right boundary of the domain

  # Define a weight for the boundary condition, surpivesed coefficients, and the pde
  supervised_weight = F(1.0)  # Weight for the supervised loss term in the total loss function.
  bc_weight = F(1.0)# for now we are going to test the two of these to zero
  pde_weight = F(1.0)

  xs = range(x_left, x_right, length=num_points)

  # Output directory for results
  output_dir = "results"
  mkpath(output_dir)

  #=
  # Single run with one dataset and fixed iteration count
  for (run_idx, inner_dict) in training_dataset
    converted_dict = convert_plugboard_keys(inner_dict)

    float_converted_dict = Dict{Matrix{Float32}, Any}()
    for (mat, series) in converted_dict
      float_converted_dict[Float32.(mat)] = series
    end

    settings = PINNSettings(10, 1234, float_converted_dict, 100, num_supervised, N, 10, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs, "adam")

    # Train the network
    p_trained, coeff_net, st, run_id = train_pinn(settings, output_dir)
    function_error = evaluate_solution(settings, p_trained, coeff_net, st, benchmark_dataset["01"], output_dir, run_id)
    @info "Function error: $function_error"
  end
  =#

  #=
  # Scaling runs (disabled)
  #=
    result = grid_search_2d(
      training_dataset,
      benchmark_dataset,
      :pde, (0.1, 1.0),  # supervised weight range
      :supervised, (0.1, 1.0),           # bc weight range
      10,                          # 10x10 grid = 100 evaluations
      fixed_weights=(bc=1.0,),
      num_supervised=21,
      N=21,
      x_left=0.0f0,
      x_right=1.0f0,
      xs=xs
    )
  =#

  #=
  scaling_neurons_settings = TrainingSchemesSettings(training_dataset, benchmark_dataset, N, num_supervised, num_points, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs)
  neurons_counts = Dict(
    "ten_neurons" => 10,
    "fifty_neurons" => 50,
    "hundred_neurons" => 100
  )

  # grid_search_at_scale(scaling_neurons_settings, neurons_counts)
  # println(result)

  # this increase the neuron count in an iterative process
  scaling_neurons(scaling_neurons_settings, neurons_counts)
  =#
  =#

  maxiters = 100000
  milestone_interval = 100

  scaling_adam_settings = TrainingSchemesSettings(training_dataset, benchmark_dataset, N, num_supervised, num_points, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs)
  scaling_adam(scaling_adam_settings, maxiters, milestone_interval)
end

batch = [10]

run_training_sequence(batch)
