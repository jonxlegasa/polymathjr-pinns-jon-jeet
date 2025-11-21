module training_schemes

include("../scripts/PINN.jl")
using .PINN

include("../utils/helper_funcs.jl")
using .helper_funcs


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
    println("Starting training for $filename for $neuron_count neurons")
    for (run_idx, inner_dict) in settings.training_dataset
      converted_dict = convert_plugboard_keys(inner_dict)

      pinn_settings = PINNSettings(neuron_count, 1234, converted_dict, 50, 1, settings.num_supervised, settings.N, settings.num_points, settings.x_left, settings.x_right, settings.supervised_weight, settings.bc_weight, settings.pde_weight, settings.xs)
      # Convert the alpha matrix keys from strings to matrices
      # Because zygote is being mean
      base_data_dir = "data"
      iteration_dir = joinpath(base_data_dir, filename)
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

      # Train the network
      p_trained, coeff_net, st = train_pinn(pinn_settings, data_directories[6]) # this is where we call the training process
      function_error = evaluate_solution(pinn_settings, p_trained, coeff_net, st, settings.benchmark_dataset["01"], data_directories)
      println(function_error)
    end
  end
end

export TrainingSchemesSettings, scaling_neurons

end
