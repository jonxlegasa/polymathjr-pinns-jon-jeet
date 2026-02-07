module training_schemes
using DataFrames
using CSV
using Plots

include("../modelcode/PINN.jl")
using .PINN

include("../utils/helper_funcs.jl")
using .helper_funcs

include("../utils/two_d_grid_search_hyperparameters.jl")
using .TwoDGridSearchOnWeights

struct TrainingSchemesSettings
  training_dataset::Dict{String,Dict{String,Any}}
  benchmark_dataset::Dict{String,Dict{String,Any}}
  N::Int
  num_supervised::Int
  num_points::Int
  x_left::Float32
  x_right::Float32
  supervised_weight::Float32
  bc_weight::Float32
  pde_weight::Float32
  xs::Vector{Float32}
end

# This will traing each NN on different neuron counts
function scaling_neurons(settings::TrainingSchemesSettings, neurons_counts::Dict{String,Int})
  for (filename, neuron_count) in neurons_counts
    @info "Starting training for $filename for $neuron_count neurons"
    for (run_idx, inner_dict) in settings.training_dataset
      converted_dict = convert_plugboard_keys(inner_dict)

      pinn_settings = PINNSettings(neuron_count, 1234, converted_dict, 500, settings.num_supervised, settings.N, settings.num_points, settings.x_left, settings.x_right, settings.supervised_weight, settings.bc_weight, settings.pde_weight, settings.xs, "adam")

      output_dir = joinpath("results", filename)
      mkpath(output_dir)

      converted_dict = convert_plugboard_keys(inner_dict)

      # Train the network
      p_trained, coeff_net, st, run_id = train_pinn(pinn_settings, output_dir)
      function_error = evaluate_solution(pinn_settings, p_trained, coeff_net, st, settings.benchmark_dataset["01"], output_dir, run_id)
      @info "Function error: $function_error"
    end
  end
end

# Envokes the grid_search with increasing neuron count
function grid_search_at_scale(settings::TrainingSchemesSettings, neurons_counts::Dict{String,Int})
  for (filename, neuron_count) in neurons_counts
    println("Starting grid search with $neuron_count neurons")
    result = grid_search_2d(
      neuron_count,
      settings.training_dataset,
      settings.benchmark_dataset,
      :pde, (0.1, 1.0),  # supervised weight range
      :supervised, (0.1, 1.0),           # bc weight range
      100,                          # nxn grid search
      fixed_weights=(bc=1.0,),
      num_supervised=10, # num_supervised N output of coefficients
      N=10,
      x_left=0.0f0,
      x_right=1.0f0,
      xs=settings.xs,
      base_data_dir="./data/$filename/grid_search"
    )
  end

  println("Good luck ;)")
  println(result)
end


## Enough neurons, lets do iterations — train once, evaluate at milestones.
function scaling_adam(settings::TrainingSchemesSettings, maxiters::Int, milestone_interval::Int)
  output_dir = "results"
  mkpath(output_dir)

  for (run_idx, inner_dict) in settings.training_dataset
    converted_dict = convert_plugboard_keys(inner_dict)

    pinn_settings = PINNSettings(100, 1234, converted_dict, maxiters, settings.num_supervised, settings.N, settings.num_points, settings.x_left, settings.x_right, settings.supervised_weight, settings.bc_weight, settings.pde_weight, settings.xs, "adam")

    # Milestone callback — called mid-training every milestone_interval iterations
    function on_milestone(p_current, iteration, coeff_net, st, run_id)
      error = evaluate_solution(pinn_settings, p_current, coeff_net, st, settings.benchmark_dataset["01"], output_dir, run_id; iteration=iteration)
      @info "Milestone $iteration — error: $error"
    end

    # Train once, evaluating at each milestone interval
    p_trained, coeff_net, st, run_id = train_pinn(pinn_settings, output_dir; milestone_interval=milestone_interval, on_milestone=on_milestone)
    @info "Training complete for example $run_idx"
  end
end


# Essentially the code is the same but it is just for LBFGs now. I do not think we will keep this... may have to delete it
# because I am just commenting out the adam training part.
function scaling_lbfgs(settings::TrainingSchemesSettings, iteration_counts::Dict{String,Int})
  array_of_benchmark_loss = Float64[]
  for (filename, iteration_count) in iteration_counts
    @info "Starting training for $filename for $iteration_count"
    for (run_idx, inner_dict) in settings.training_dataset
      converted_dict = convert_plugboard_keys(inner_dict)

      pinn_settings = PINNSettings(100, 1234, converted_dict, iteration_count, settings.num_supervised, settings.N, settings.num_points, settings.x_left, settings.x_right, settings.supervised_weight, settings.bc_weight, settings.pde_weight, settings.xs, "lbfgs")

      output_dir = joinpath("results", filename)
      mkpath(output_dir)

      converted_dict = convert_plugboard_keys(inner_dict)

      # Train the network
      p_trained, coeff_net, st, run_id = train_pinn(pinn_settings, output_dir)
      function_error = evaluate_solution(pinn_settings, p_trained, coeff_net, st, settings.benchmark_dataset["01"], output_dir, run_id)

      push!(array_of_benchmark_loss, function_error)
      @info "Function error: $function_error"
    end
  end
  # Save summary to CSV
  df = DataFrame(
    index = 1:length(array_of_benchmark_loss),
    function_error = array_of_benchmark_loss
  )
  CSV.write("./results/benchmark_losses.csv", df)
end



# Envokes the grid_search with increasing neuron count
function grid_search_at_scale(settings::TrainingSchemesSettings, neurons_counts::Dict{String,Int})
  for (filename, neuron_count) in neurons_counts
    println("Starting grid search with $neuron_count neurons")
    result = grid_search_2d(
      neuron_count,
      settings.training_dataset,
      settings.benchmark_dataset,
      :pde, (0.1, 1.0),  # supervised weight range
      :supervised, (0.1, 1.0),           # bc weight range
      100,                          # nxn grid search
      fixed_weights=(bc=1.0,),
      num_supervised=10, # num_supervised N output of coefficients
      N=10,
      x_left=0.0f0,
      x_right=1.0f0,
      xs=settings.xs,
      base_data_dir="./data/$filename/grid_search"
    )
  end

  println("Good luck ;)")
  println(result)
end



export TrainingSchemesSettings, scaling_neurons, grid_search_at_scale, scaling_adam

end
