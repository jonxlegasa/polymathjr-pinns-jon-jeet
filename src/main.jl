using Dates
using JSON

# Include util functions
include("../utils/plugboard.jl")
using .Plugboard

include("../utils/ProgressBar.jl")
using .ProgressBar

include("../utils/ConvertStringToMatrix.jl")
using .ConvertStringToMatrix

include("../scripts/PINN.jl")
using .PINN

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

  # We only generate one benchmark dataset
  # benchmark_dataset_setting::Settings = Plugboard.Settings(1, 0, 1, benchmark_data_dir)
  # Plugboard.generate_random_ode_dataset(benchmark_dataset_setting, 1) 

  # generate training datasets and benchmarks 
  for (batch_index, k) in enumerate(batch_sizes)
    # set up plugboard for solutions to ay' + by = 0 where a,b != 0
    run_number_formatted = lpad(batch_index, 2, '0')
    training_dataset_setting::Settings = Plugboard.Settings(1, 0, k, training_data_dir)

    println("\n" * "="^50)
    println("Generating datasets for training and benchmarks $run_number_formatted")
    println("="^50)
    println("Number of examples: ", k)
    
    # Plugboard.generate_random_ode_dataset(training_dataset_setting, batch_index) # training data
   # create_training_run_dirs(batch_index, k) # Create the training dirs
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

  # loop through each training run and pass in the series coefficients 
  # and its corresponding ODE embedding
  for (run_idx, inner_dict) in training_dataset
    # Convert the alpha matrix keys from strings to matrices
    # Because zygote is being mean
    ConvertSettings = StringToMatrixSettings(inner_dict)
    converted_dict = ConvertStringToMatrix.convert(ConvertSettings)

    #=
    This establishes settings for PINN. 64 neurons, a random seed to init
    parameters, and iterations for training.
    =#
    settings = PINNSettings(5, 1234, converted_dict, 500, 100)

    # Train the network
    p_trained, coeff_net, st = train_pinn(settings) # this is where we call the training process
 
    #=
    this is the matrix we test the solution for. 
    The solution is y = Ae^x
    CALL GLOBAL LOSS HERE WITH 
    =#

    evaluate_solution(p_trained, coeff_net, st, benchmark_dataset["01"])
  end
end

# making the array large will increase number of training runs.
# each entry of the array is an interger that determines the # of examples generated in 
# each training run

batch = [1]

# Uncomment to run the example
run_training_sequence(batch) # we first start here with the "foreign call" error
