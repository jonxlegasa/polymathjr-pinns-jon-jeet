using Dates
using JSON

# Include util functions
include("../utils/plugboard.jl")
using .Plugboard

include("../utils/ProgressBar.jl")
using .ProgressBar

include("../scripts/PINN.jl")
using .PINN

function setup_training_run(run_number::Int64, batch_size::Any)
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

#=
# Notes: 
#  We want each training run to follow these batch sizes [1, 10, 50, 100]
#   lets make this an array babyyy
=#

function init_batches(batch_sizes::Array{Int})
  """
  Initializes batches by generating datasets for different batch sizes.
  Args:
      batch_sizes: Array of integers representing different batch sizes
  """
  for (batch_index, k) in enumerate(batch_sizes)
    # set up plugboard for solutions to ay' + by = 0 where a,b != 0

    run_number_formatted = lpad(batch_index, 2, '0')
    s::Settings = Plugboard.Settings(2, 2, k)

    println("\n" * "="^50)
    println("Generating datasets for training run $run_number_formatted")
    println("="^50)
    println("Number of examples: ", k)

    Plugboard.generate_random_ode_dataset(s, batch_index) # training data
    # Plugboard.generate_random_ode_dataset(s, batch_index) # maybe create the validation JSON data
    # Create the training dirs
    println("\n" * "="^50)
    println("Starting Training Run $run_number_formatted")
    println("="^50)
    setup_training_run(batch_index, k)
  end
end

function run_training_sequence(batch_sizes::Array{Int})
  """
  Runs a sequence of training runs with different training example configurations.
  Args:
      batch_sizes: Array of integers representing different batch sizes
  """
  # Initialize all batches first
  init_batches(batch_sizes)

  # Load the generated dataset
  dataset = JSON.parsefile("./data/dataset.json")

  # Loop through each entry in the JSON object
  # This is the code that is not working. Alpha matrix is seen as a number.
  for (run_idx, inner_dict) in dataset
    println(inner_dict)
    for (alpha_matrix_key, series_coeffs) in inner_dict
      # TODO: We need to setup the pinn training here
      println("Series coefficients that will be trained soon...: $series_coeffs") # lil error checking
      println("This is the alpha matrix: $alpha_matrix_key")
      println("Current training run: $run_idx")

      # Convert string key back to matrix
      alpha_matrix = eval(Meta.parse(alpha_matrix_key))
      settings = PINNSettings(64, 1234, alpha_matrix, 500, 100)

      #=
      # Train the network
      p_trained, coeff_net, st = train_pinn(settings, series_coeffs)
      sample_matrix = [1;
        1]

      # Evaluate results
      a_learned, u_func = evaluate_solution(p_trained, coeff_net, st, sample_matrix)
      println(a_learned)
      println(u_func)

      =#



      # TODO: Add the training implementation for the PINN Here
      # PINN training here using:
      # - alpha_matrix (converted from key)
      # - series_coeffs as target
      # - ASK VICTOR HOW TO IMPLEMENT THE PINN WITH THIS
    end
  end
end

# making the array large will increase number of training runs.
# each entry of the array is an interger that determines the # of examples generated in 
# each training run

batch = [1, 1, 1]

# Uncomment to run the example
run_training_sequence(batch)
