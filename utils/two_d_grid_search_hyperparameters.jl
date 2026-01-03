module TwoDGridSearchOnWeights

using Plots
using Statistics

include("../modelcode/PINN.jl")
using .PINN

include("../utils/helper_funcs.jl")
using .helper_funcs

"""
  GridSearchResult
  
Stores the results of a hyperparameter search.
"""

struct GridSearchResult
  weight_values::Dict{Symbol,Vector{Float64}}  # e.g., :supervised => [1.0, 2.0, ...]
  objective_values::Array{Float64}  # N-dimensional array of objective values
  best_weights::NamedTuple  # Best weight configuration found
  best_objective::Float64  # Best objective value
end

"""
  evaluate_weight_configuration(training_dataset, weights::NamedTuple, 
                                 num_supervised, N, x_left, x_right, xs)

Train and evaluate a single weight configuration.
Returns an objective value (lower is better).
"""
function evaluate_weight_configuration(neuron_count, training_dataset, benchmark_dataset, weights::NamedTuple,
num_supervised, N, x_left, x_right, xs, base_data_dir)
  # Extract weights
  supervised_weight = weights.supervised
  bc_weight = weights.bc
  pde_weight = weights.pde

  # iteration_dir = joinpath(base_data_dir, "test")
  # mkpath(iteration_dir)

  # Create directory for this configuration
  config_dir = joinpath(base_data_dir,
    "s$(round(supervised_weight, digits=2))-" *
    "b$(round(bc_weight, digits=2))-" *
    "p$(round(pde_weight, digits=2))")
  mkpath(config_dir)

  # Initialize objective value accumulator
  total_error = 0.0

  # Train with current weight configuration
  for (run_idx, inner_dict) in training_dataset
    data_directories = [
      joinpath(config_dir, "function_comparison.png"),
      joinpath(config_dir, "coefficient_comparison.png"),
      joinpath(config_dir, "adam_iteration_and_loss_comparison.png"),
      joinpath(config_dir, "lbfgs_iteration_and_loss_comparison.png"),
      joinpath(config_dir, "iteration_plot.png"),

      joinpath(config_dir, "iteration_output.csv"),
    ]
    converted_dict = convert_plugboard_keys(inner_dict)

    float_converted_dict = Dict{Matrix{Float32}, Any}()
    for (mat, series) in converted_dict
      float_converted_dict[Float32.(mat)] = series
    end

    settings = PINNSettings(neuron_count, 1234, float_converted_dict, 10000, 1, num_supervised, N, 1000, x_left, x_right, supervised_weight, bc_weight, pde_weight, xs)

    # Train the network
    p_trained, coeff_net, st = train_pinn(settings, data_directories[6]) # this is where we call the training process
    function_error = evaluate_solution(settings, p_trained, coeff_net, st, benchmark_dataset["01"], data_directories)
    println(function_error)

    total_error += function_error
  end

  return total_error
end

"""
  grid_search_2d(training_dataset, weight1::Symbol, weight1_range::Tuple,
                 weight2::Symbol, weight2_range::Tuple, num_points::Int;
                 fixed_weights::NamedTuple, kwargs...)

Perform 2D grid search over two hyperparameters.
"""
function grid_search_2d(neuron_count, training_dataset, benchmark_dataset,
  weight1::Symbol, weight1_range::Tuple{Float64,Float64},
  weight2::Symbol, weight2_range::Tuple{Float64,Float64},
  num_points::Int;
  fixed_weights::NamedTuple,
  num_supervised, N, x_left, x_right,
  xs,
  base_data_dir)

  println("Starting 2D grid search")
  println("Weight 1: $(weight1), Range: $(weight1_range)")
  println("Weight 2: $(weight2), Range: $(weight2_range)")
  println("Grid points per dimension: $(num_points)")
  println("Fixed weights: $(fixed_weights)")
  println("="^50)

  # Create grid of weight values
  weight1_values = range(weight1_range[1], weight1_range[2], length=num_points)
  weight2_values = range(weight2_range[1], weight2_range[2], length=num_points)

  # Initialize results matrix
  objective_matrix = zeros(Float64, num_points, num_points)

  # Track best configuration
  best_objective = Inf
  best_weights = nothing

  # Iterate through grid
  for (i, w1) in enumerate(weight1_values)
    for (j, w2) in enumerate(weight2_values)
      println("\nGrid point ($(i), $(j)) / ($(num_points), $(num_points))")
      println("  Testing: $(weight1)=$(w1), $(weight2)=$(w2)")

      # Create weight configuration
      weights = create_weight_tuple(weight1, w1, weight2, w2, fixed_weights)

      # Evaluate this configuration
      objective_value = evaluate_weight_configuration(neuron_count,
        training_dataset, benchmark_dataset, weights, num_supervised, N,
        x_left, x_right, xs, base_data_dir
      )

      objective_matrix[j, i] = objective_value

      println("  Objective value: $(objective_value)")

      # Update best configuration
      if objective_value < best_objective
        best_objective = objective_value
        best_weights = weights
        println("  *** New best configuration found! ***")
      end
    end
  end

  # Store results
  weight_values = Dict(weight1 => collect(weight1_values),
    weight2 => collect(weight2_values))

  result = GridSearchResult(weight_values, objective_matrix,
    best_weights, best_objective)

  # Create visualizations
  visualize_search_results(result, weight1, weight2, base_data_dir)

  # Save summary
  save_search_summary(result, weight1, weight2, base_data_dir)

  return result
end

"""
  random_search_2d(training_dataset, weight1::Symbol, weight1_range::Tuple,
                   weight2::Symbol, weight2_range::Tuple, num_samples::Int;
                   fixed_weights::NamedTuple, kwargs...)

Perform random search over two hyperparameters.
"""
function random_search_2d(training_dataset,
  weight1::Symbol, weight1_range::Tuple{Float64,Float64},
  weight2::Symbol, weight2_range::Tuple{Float64,Float64},
  num_samples::Int;
  fixed_weights::NamedTuple,
  num_supervised, N, x_left, x_right,
  xs,
  base_data_dir="data/random_search")
  println("Starting 2D random search")
  println("Weight 1: $(weight1), Range: $(weight1_range)")
  println("Weight 2: $(weight2), Range: $(weight2_range)")
  println("Number of samples: $(num_samples)")
  println("Fixed weights: $(fixed_weights)")
  println("="^50)

  # Store sampled points and their objectives
  weight1_samples = Float64[]
  weight2_samples = Float64[]
  objective_values = Float64[]

  best_objective = Inf
  best_weights = nothing

  for sample in 1:num_samples
    println("\nSample $(sample) / $(num_samples)")

    # Sample random weight values
    w1 = rand() * (weight1_range[2] - weight1_range[1]) + weight1_range[1]
    w2 = rand() * (weight2_range[2] - weight2_range[1]) + weight2_range[1]

    push!(weight1_samples, w1)
    push!(weight2_samples, w2)

    # Create weight configuration
    weights = create_weight_tuple(weight1, w1, weight2, w2, fixed_weights)

    # Evaluate this configuration
    objective_value = evaluate_weight_configuration(
      training_dataset, weights, num_supervised, N,
      x_left, x_right, xs; base_data_dir=base_data_dir
    )

    push!(objective_values, objective_value)

    println("  Weights: $(weight1)=$(w1), $(weight2)=$(w2)")
    println("  Objective value: $(objective_value)")

    # Update best configuration
    if objective_value < best_objective
      best_objective = objective_value
      best_weights = weights
      println("  *** New best configuration found! ***")
    end
  end

  # Create visualization with scattered points
  visualize_random_search(weight1_samples, weight2_samples, objective_values,
    weight1, weight2, weight1_range, weight2_range,
    base_data_dir)

  # Save summary
  save_random_search_summary(weight1_samples, weight2_samples, objective_values,
    best_weights, best_objective, weight1, weight2,
    base_data_dir)

  return (weight1_samples, weight2_samples, objective_values, best_weights, best_objective)
end

"""
  create_weight_tuple(weight1::Symbol, w1::Float64, 
                     weight2::Symbol, w2::Float64, 
                     fixed_weights::NamedTuple)

Helper function to create a NamedTuple with all three weights.
"""
function create_weight_tuple(weight1::Symbol, w1::Float64,
  weight2::Symbol, w2::Float64,
  fixed_weights::NamedTuple)

  weights_dict = Dict{Symbol,Float64}()

  # Set the two variables being searched
  weights_dict[weight1] = w1
  weights_dict[weight2] = w2

  # Add the fixed weight
  for (key, val) in pairs(fixed_weights)
    if key != weight1 && key != weight2
      weights_dict[key] = val
    end
  end

  # Ensure all three weights are present
  if !haskey(weights_dict, :supervised) ||
     !haskey(weights_dict, :bc) ||
     !haskey(weights_dict, :pde)
    error("Weight configuration incomplete. Need supervised, bc, and pde weights.")
  end

  return (supervised=weights_dict[:supervised],
    bc=weights_dict[:bc],
    pde=weights_dict[:pde])
end

"""
  visualize_search_results(result::GridSearchResult, 
                          weight1::Symbol, weight2::Symbol, 
                          base_data_dir::String)

Create contour plot visualization of grid search results.
"""
function visualize_search_results(result::GridSearchResult,
  weight1::Symbol, weight2::Symbol,
  base_data_dir::String)

  w1_values = result.weight_values[weight1]
  w2_values = result.weight_values[weight2]
  obj_matrix = result.objective_values


  # Calculate appropriate number of levels based on data range
  min_obj = minimum(obj_matrix)
  max_obj = maximum(obj_matrix)
  obj_range = max_obj - min_obj

  # num_levels = max(5, min(30, Int(round(obj_range * 10))))
  # num_levels = 10.0 .^ range(-2, 2, length=20) 
  # num_levels = 12
  # num_levels = 20
  # println("Objective value range: $(min_obj) to $(max_obj)")
  # println("Using $(num_levels) contour levels")

  # Create contour plot
  #=
  p = contour(w1_values, w2_values, obj_matrix,
    xlabel=String(weight1),
    ylabel=String(weight2),
    yscale=:log10,
    title="Hyperparameter Search: $(weight1) vs $(weight2)",
    levels=num_levels,
    linewidth=2,
    color=:magma,
    fill=false)
  =# # NO CONTOUR MAP, LOSS IS TOO GREAT

  # Add grid points as scatter plot
  # Create all combinations of w1 and w2 values
  grid_w1 = repeat(w1_values, outer=length(w2_values))
  grid_w2 = repeat(w2_values, inner=length(w1_values))

  scatter!(grid_w1, grid_w2,
    marker=:x,
    markersize=10,
    color=:green,
    label="Grid points",
    alpha=1)

  # Mark the best point
  best_w1 = result.best_weights[weight1]
  best_w2 = result.best_weights[weight2]
  #=
  scatter!([best_w1], [best_w2],
    marker=:star,
    markersize=10,
    color=:pink,
    label="Best (obj=$(round(result.best_objective, digits=4)))")
  =#
  # Save plot
  # savefig(p, joinpath(base_data_dir, "hyperparameter_contour.png"))

  # Create 3D surface plot
  p3d = surface(w1_values, w2_values, obj_matrix,
    xlabel=String(weight1),
    ylabel=String(weight2),
    zlabel="Objective Value",
    title="Objective Landscape",
    camera=(45, 30),
    color=:magma)
  # Add grid points to 3D plot
  grid_obj = vec(obj_matrix')  # Flatten the objective matrix to match grid points
  scatter3d!(grid_w1, grid_w2, grid_obj,
    marker=:circle,
    markersize=3,
    color=:green,
    label="Grid points",
    alpha=0.6)

  scatter3d!([best_w1], [best_w2], [result.best_objective],
    marker=:star,
    markersize=8,
    color=:red,
    label="Best")

  savefig(p3d, joinpath(base_data_dir, "hyperparameter_surface.png"))

  println("\nVisualizations saved to:")
  # println("  - $(joinpath(base_data_dir, "hyperparameter_contour.png"))")
  println("  - $(joinpath(base_data_dir, "hyperparameter_surface.png"))")
end

"""
  visualize_random_search(w1_samples, w2_samples, objectives,
                         weight1::Symbol, weight2::Symbol,
                         w1_range, w2_range, base_data_dir::String)

Create visualization of random search results with scattered points.
"""
function visualize_random_search(w1_samples, w2_samples, objectives,
  weight1::Symbol, weight2::Symbol,
  w1_range, w2_range,
  base_data_dir::String)

  # Create scatter plot with color indicating objective value
  p = scatter(w1_samples, w2_samples,
    marker_z=objectives,
    xlabel=String(weight1),
    ylabel=String(weight2),
    title="Random Search: $(weight1) vs $(weight2)",
    color=:viridis,
    colorbar=true,
    markersize=6,
    label="",
    xlims=w1_range,
    ylims=w2_range)

  # Mark the best point
  best_idx = argmin(objectives)
  scatter!([w1_samples[best_idx]], [w2_samples[best_idx]],
    marker=:star,
    markersize=12,
    color=:red,
    label="Best (obj=$(round(objectives[best_idx], digits=4)))")

  savefig(p, joinpath(base_data_dir, "random_search_scatter.png"))

  # Optionally, interpolate to create a contour plot
  # This requires a package like Interpolations.jl

  println("\nVisualization saved to:")
  println("  - $(joinpath(base_data_dir, "random_search_scatter.png"))")
end

"""
  save_search_summary(result::GridSearchResult, weight1::Symbol, 
                     weight2::Symbol, base_data_dir::String)

Save summary of grid search results.
"""
function save_search_summary(result::GridSearchResult,
  weight1::Symbol, weight2::Symbol,
  base_data_dir::String)

  summary_file = joinpath(base_data_dir, "search_summary.txt")
  open(summary_file, "w") do f
    write(f, "Grid Search Summary\n")
    write(f, "="^50 * "\n\n")
    write(f, "Search Parameters:\n")
    write(f, "  Weight 1: $(weight1)\n")
    write(f, "  Weight 2: $(weight2)\n")
    write(f, "  Grid points: $(length(result.weight_values[weight1])) x $(length(result.weight_values[weight2]))\n\n")
    write(f, "Best Configuration Found:\n")
    write(f, "  supervised: $(result.best_weights.supervised)\n")
    write(f, "  bc: $(result.best_weights.bc)\n")
    write(f, "  pde: $(result.best_weights.pde)\n")
    write(f, "  Objective value: $(result.best_objective)\n\n")
    write(f, "Statistics:\n")
    write(f, "  Min objective: $(minimum(result.objective_values))\n")
    write(f, "  Max objective: $(maximum(result.objective_values))\n")
    write(f, "  Mean objective: $(mean(result.objective_values))\n")
    write(f, "  Std objective: $(std(result.objective_values))\n")
  end

  println("Summary saved to: $(summary_file)")
end

"""
  save_random_search_summary(w1_samples, w2_samples, objectives,
                            best_weights, best_objective,
                            weight1::Symbol, weight2::Symbol,
                            base_data_dir::String)

Save summary of random search results.
"""
function save_random_search_summary(w1_samples, w2_samples, objectives,
  best_weights, best_objective,
  weight1::Symbol, weight2::Symbol,
  base_data_dir::String)

  summary_file = joinpath(base_data_dir, "random_search_summary.txt")
  open(summary_file, "w") do f
    write(f, "Random Search Summary\n")
    write(f, "="^50 * "\n\n")
    write(f, "Search Parameters:\n")
    write(f, "  Weight 1: $(weight1)\n")
    write(f, "  Weight 2: $(weight2)\n")
    write(f, "  Number of samples: $(length(objectives))\n\n")
    write(f, "Best Configuration Found:\n")
    write(f, "  supervised: $(best_weights.supervised)\n")
    write(f, "  bc: $(best_weights.bc)\n")
    write(f, "  pde: $(best_weights.pde)\n")
    write(f, "  Objective value: $(best_objective)\n\n")
    write(f, "Statistics:\n")
    write(f, "  Min objective: $(minimum(objectives))\n")
    write(f, "  Max objective: $(maximum(objectives))\n")
    write(f, "  Mean objective: $(mean(objectives))\n")
    write(f, "  Std objective: $(std(objectives))\n")
  end

  println("Summary saved to: $(summary_file)")
end

export GridSearchResult, grid_search_2d, random_search_2d

end
