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
    println("Created data directory: $data_dir")
  end

  # Format run number with zero padding (01, 02, 03, etc.)
  run_number_formatted = lpad(run_number, 2, '0')
  # Create training run directory
  training_run_dir = joinpath(data_dir, "training-run-$run_number_formatted")
  if !isdir(training_run_dir)
    mkdir(training_run_dir)
    println("Created training run directory: $training_run_dir")
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

  println("Training run $run_number_formatted setup complete!")
  println("Output file created: $output_file")

  return training_run_dir, output_file
end

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

    println("\n" * "="^50)
    println("Generating datasets for training and benchmarks $run_number_formatted")
    println("="^50)
    println("Number of examples: ", k)

    #=
    #  linear combination of coefficients of the ODEs
    Plugboard.generate_random_ode_dataset(training_dataset_setting, batch_index) # create training data
    # create_training_run_dirs(batch_index, k) # Create the training dirs

    training_dataset = JSON.parsefile(training_data_dir)
    # add the ode matrices together
    matrices_to_be_added = Matrix{Int}[
      alpha_matrix_key
      for (run_idx, inner_dict) in training_dataset
      for (alpha_matrix_key, series_coeffs) in convert_plugboard_keys(inner_dict)
    ]

    linear_combination_of_matrices = reduce(+, matrices_to_be_added)
    println("Linear combos: ", linear_combination_of_matrices)

    Plugboard.generate_specific_ode_dataset(benchmark_dataset_setting, 1, linear_combination_of_matrices)
    =#

    # code for scalar multiples of the coefficients of one ODE
    #= 
    array_of_matrices = Matrix{Int64}[]
    beginning_alpha_matrix = reshape([3, 4], 2, 1)  # 2x1 Matrix{Int64}
    push!(array_of_matrices, beginning_alpha_matrix)

    for n in 1:10
      push!(array_of_matrices, beginning_alpha_matrix * (n))
    end
    =#

    # test_matrix = [1; 1;;]
    # Plugboard.generate_specific_ode_dataset(benchmark_dataset_setting, 1, test_matrix)

    # n = 10 # this will the number of matrices we create
    # array_of_matrices = create_matrix_array(n)

    # test_matrix = [1; 6; 2;;]
    test_matrix = [1; 6; 2;;]
    #=
    Plugboard.generate_random_ode_dataset(training_dataset_setting, batch_index)
    Plugboard.generate_random_ode_dataset(benchmark_dataset_setting, batch_index)
    =#

    # Plugboard.generate_random_ode_dataset(training_dataset_setting, batch_index)
    # Plugboard.generate_specific_ode_dataset(benchmark_dataset_setting, 1, test_matrix)
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
  # Initialize all batches first
  init_batches(batch_sizes)

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
#=
  # This code is for the classic training scheme for no change in neuron count or whatever
  for (run_idx, inner_dict) in training_dataset
    # Convert the alpha matrix keys from strings to matrices
    # Because zygote is being mean
    base_data_dir = "data"
    iteration_dir = joinpath(base_data_dir, "test")
    mkpath(iteration_dir)

    data_directories = [
      joinpath(iteration_dir, "function_comparison.png"),
      joinpath(iteration_dir, "coefficient_comparison.png"),
      joinpath(iteration_dir, "adam_iteration_and_loss_comparison.png"),
      joinpath(iteration_dir, "lbfgs_iteration_and_loss_comparison.png"),
      joinpath(iteration_dir, "iteration_plot.png"),

      joinpath(iteration_dir, "iteration_output.csv"),
    ]
    converted_dict = convert_plugboard_keys(inner_dict)

    float_converted_dict = Dict{Matrix{Float32}, Any}()
    for (mat, series) in converted_dict
      float_converted_dict[Float32.(mat)] = series
    end

    settings = PINNSettings(100, 1234, float_converted_dict, 5, num_supervised, N, 10, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs)

    # Train the network
    p_trained, coeff_net, st = train_pinn(settings, data_directories[6]) # this is where we call the training process
    function_error = evaluate_solution(settings, p_trained, coeff_net, st, benchmark_dataset["01"], data_directories)
    println(function_error)
  end

  =#



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

  scaling_iterations_settings = TrainingSchemesSettings(training_dataset, benchmark_dataset, N, num_supervised, num_points, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs)
  iteration_counts = Dict(
    "lbfgs_1000" => 1000,
    "lbfgs_10000" => 10000,
    "lbfgs_100000" => 100000,
  )

  scaling_adam_settings = TrainingSchemesSettings(training_dataset, benchmark_dataset, N, num_supervised, num_points, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs)
  scaling_adam(scaling_adam_settings, iteration_counts)
end

batch = [10]

run_training_sequence(batch)
