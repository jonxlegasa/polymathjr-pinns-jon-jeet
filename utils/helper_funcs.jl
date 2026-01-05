module helper_funcs

using CSV
using DataFrames
using Plots

function convert_plugboard_keys(inner_dict)
  converted_dict = Dict{Matrix{Int},Any}()
  for (alpha_matrix_key, series_coeffs) in inner_dict
    # println("The current  ODE I am calculating the loss for right now: ", alpha_matrix_key)
    # println("The local loss is locally lossing...")
    alpha_matrix = eval(Meta.parse(alpha_matrix_key)) # convert from string to matrix 
    converted_dict[alpha_matrix] = series_coeffs
  end

  return converted_dict
end

const loss_buffer = Ref{Vector{Dict{Symbol,Float64}}}(Dict{Symbol,Float64}[])

"""
Initialize the loss buffer at the start of training
"""
function initialize_loss_buffer()
  loss_buffer[] = Dict{Symbol,Float64}[]
  println("Loss buffer initialized")
end

"""
Add loss values to the in-memory buffer (very fast)
"""
function buffer_loss_values(; kwargs...)
  loss_dict = Dict{Symbol,Float64}()
  for (key, value) in kwargs
    loss_dict[key] = Float64(value)
  end
  push!(loss_buffer[], loss_dict)
end

"""
Write all buffered loss values to CSV file at once
"""
function write_buffer_to_csv(csv_file)
  # Create directory if needed
  dir = dirname(csv_file)
  if !isdir(dir)
    mkpath(dir)
  end

  # Initialize DataFrame with loss_type column
  df = DataFrame(loss_type=String[])

  # Process each buffered entry
  for (i, loss_dict) in enumerate(loss_buffer[])
    col_name = "iter_$(i)"

    # Add new column for this iteration
    df[!, col_name] = Vector{Union{Missing,Float64}}(missing, nrow(df))

    # Fill in loss values for this iteration
    for (key, value) in loss_dict
      loss_name = String(key)

      # Find if this loss type already exists as a row
      row_idx = findfirst(df.loss_type .== loss_name)

      if isnothing(row_idx)
        # New loss type - create new row
        new_row = DataFrame(loss_type=[loss_name])

        # Fill previous iterations with missing
        for existing_col in names(df)[2:end]  # Skip loss_type column
          if existing_col != col_name
            new_row[!, existing_col] = [missing]
          end
        end

        # Add current value
        new_row[!, col_name] = [Float64(value)]
        append!(df, new_row, promote=true)
      else
        # Existing loss type - update the cell
        df[row_idx, col_name] = Float64(value)
      end
    end
  end

  # Write to CSV once
  CSV.write(csv_file, df)
  println("Wrote $(length(loss_buffer[])) evaluations to $(csv_file)")
end

"""
Get the number of buffered evaluations
"""
function get_buffer_size()
  return length(loss_buffer[])
end

function show_benchmark_loss(
  array_of_benchmark_loss,
  filename,
  title,
  color,
  bar_width
)

  # Create bar graph from array
  bar(
    1:length(array_of_benchmark_loss),
    array_of_benchmark_loss,
    xlabel = "Training Run",
    ylabel = "Function Error",
    title = title,
    legend = false,
    grid = true,
    bar_width = bar_width,
    color = color
  )

  # Save the plot
  savefig(filename)
end

# this will be a reference for our new analytic_sol_fun
# analytic_sol_func(x) = (pi * x * (-x + (pi^2) * (2x - 3) + 1) - sin(pi * x)) / (pi^3) # We replace with our training examples

"""
This function will give us all the types of solutions for the homogenous ODE y'' + ay' + by = 0
depending on the discriminats value
"""

#=

function exponential_solution(x, c1, c2, α, β, D)
  if D > 0
    return c1 * exp(α * x) + c2 * exp(β * x)
  elseif D = 0
    return (c1 + c2 * x) * exp((-α * x)/2)
  elseif D < 0
    return exp(α * x) * (c1 * cos(β * x) + c2 * sin(β * x))
  end
end

=#

"""
Quadratic formula, this will be used for our analytic function we have. This will determine the roots of the function
"""
function quadratic_formula(a, b, c)
  discr = b^2 - 4*a*c
  # Handle complex roots if discriminant is negative
  sq = (discr > 0) ? sqrt(discr) : sqrt(discr + 0im)
  roots = [(-b - sq) / (2a), (-b + sq) / (2a)]
  return sort(roots, by=real, rev=true)  # Sort by real part, descending
end


# Generate random values for a and b that satisfy a² - 4b > 0
# Strategy: choose a first, then choose b such that b < a²/4
function generate_valid_matrix()
  a = rand(-10:10)  # Random integer for a
  max_b_float = (a^2) / 4 - 1  # Subtract 1 to ensure strict inequality with integers
  max_b = floor(Int, max_b_float)  # Convert to integer

  # Only generate b if max_b is positive
  if max_b > 0
    b = rand(0:max_b)
  else
    b = 0
  end

  return reshape([1, a, b], 3, 1)  # 3x1 column matrix
end

function create_matrix_array(n::Int)
  array_of_matrices = Matrix{Int64}[]

  for i in 1:n
    valid_matrix = generate_valid_matrix()

    # Verify the constraint (optional, for debugging)
    a = valid_matrix[2, 1]
    b = valid_matrix[3, 1]
    @assert a^2 - 4*b > 0 "Constraint violated: a² - 4b = $(a^2 - 4*b)"

    push!(array_of_matrices, valid_matrix)
  end

  return array_of_matrices
end


export convert_plugboard_keys, initialize_loss_buffer, buffer_loss_values, write_buffer_to_csv, get_buffer_size, exponential_solution, quadratic_formula, generate_valid_matrix, create_matrix_array
end
