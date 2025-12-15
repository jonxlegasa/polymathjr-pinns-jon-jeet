module Plugboard

using LinearAlgebra
using TaylorSeries
using Random
using JSON

struct Settings
  ode_order::Int
  poly_degree::Int
  dataset_size::Int
  data_dir::String
  num_of_terms::Int
end

function get_user_inputs()
  println("The Plugboard: Randomized ODE Generator")
  println("=================================")
  print("What order do you want your ODE to be? (e.g., 1 for first order, 2 for second order): ")
  ode_order = parse(Int, readline())
  print("What is the highest degree polynomial you want? (e.g., 2 for degree 2): ")
  poly_degree = parse(Int, readline())
  print("How many ODEs do you want solved? (e.g., 50 for 50 training examples): ")
  dataset_size = parse(Int, readline())
  return ode_order, poly_degree, dataset_size
end

# Generate random alpha matrix - unchanged
function generate_random_alpha_matrix(ode_order, poly_degree)
  rows = ode_order + 1
  cols = poly_degree + 1
  α_matrix = Matrix{Int}(undef, rows, cols)
  for i in 1:rows
    for j in 1:cols
      α_matrix[i, j] = rand(Bool) ? rand(-10:-1) : rand(1:10)
    end
  end
  return α_matrix
end

# Factorial product - keep the same
function factorial_product_numeric(n_val, k, i)
  if k == 0
    return 1.0
  end
  product = 1.0
  for j in 1:k
    product *= (n_val + j - i)
  end
  return product
end

# New closed-form implementation
# this just 
function solve_ode_series_closed_form(α_matrix, initial_conditions, num_terms)
  rows, cols = size(α_matrix)
  m = rows - 1  # ODE order

  # Initialize series with initial conditions
  series_coeffs = Float64.(initial_conditions)

  println("Starting with initial conditions: ", series_coeffs)
  println("ODE order: ", m)

  # Compute coefficients using closed form
  for n in 0:(num_terms-length(initial_conditions)-1)
    # Check if c_{m,0} is zero (would make equation singular)
    c_m_0 = α_matrix[m+1, 1]  # c_{m,0} is at position [m+1, 1]
    if c_m_0 == 0
      println("Error: c_{m,0} = 0, cannot solve")
      break
    end

    # Compute the sum term
    sum_term = 0.0

    for k in 0:m
      for j in 0:(cols-1)
        c_kj = α_matrix[k+1, j+1]  # c_{k,j} at position [k+1, j+1]

        if c_kj != 0
          coeff_index = n - j + k

          # Check if we have this coefficient available
          if coeff_index >= 0 && coeff_index < length(series_coeffs)
            factorial_term = factorial_product_numeric(n - j, k, 0)
            term_value = c_kj * factorial_term * series_coeffs[coeff_index+1] # is this what they mean by SHUT OFF THE FACTORIAL !!!!
            # term_value = c_kj * series_coeffs[coeff_index+1] # this is when we want factorial_tern off.
            sum_term += term_value
          end
        end
      end
    end

    # Apply the closed form formula: a_{n+m} = -(1/(c_{m,0} * (n+m)!)) * sum
    factorial_nm = factorial(big(n + m))  # Use big integer for large factorials
    # denominator = c_m_0 * factorial_nm # no this is where it is
    denominator = c_m_0

    new_coeff = -sum_term / denominator
    push!(series_coeffs, new_coeff)
  end

  return series_coeffs
end

# just generates the json file that you see in ./data
function generate_random_ode_dataset(s::Settings, batch_index::Int)
  ode_order = s.ode_order
  poly_degree = s.poly_degree
  println("\ngenerating random α matrices for:")
  println("- ode order: $ode_order")
  println("- polynomial degree: $poly_degree")

  # Generate dataset_size examples
  for example_k in 1:s.dataset_size
    α_matrix = generate_random_alpha_matrix(s.ode_order, s.poly_degree) # generate ODE matrix
    println("\n--- Example #$example_k ---")
    println("α matrix:")
    display(α_matrix)
    # generate exactly ode_order initial conditions
    initial_conditions = Float64[]
    for i in 0:(ode_order-1)
      if i == 0
        push!(initial_conditions, rand(1:5))  # y(0) = a_0
        println("y(0) = ", initial_conditions[end])
      elseif i == 1
        push!(initial_conditions, rand(1:5))  # y'(0) = a_1
        println("y'(0) = ", initial_conditions[end])
      end
    end
    try
      # output taylor series and its coefficients
      # this actually computes the taylor series
      series_coeffs = solve_ode_series_closed_form(α_matrix, initial_conditions, s.num_of_terms)
      println("truncated series coefficients: ", series_coeffs)
      # read existing data
      existing_data = if isfile(s.data_dir)
        JSON.parsefile(s.data_dir)
      else
        Dict()
      end

      # Determine which training run this is based on existing data
      dataset_key = lpad(batch_index, 2, '0')

      # Initialize dataset key if it doesn't exist
      if !haskey(existing_data, dataset_key)
        existing_data[dataset_key] = Dict()
      end

      # use alpha matrix as key, series coefficients as value within the dataset batch
      existing_data[dataset_key][string(α_matrix)] = series_coeffs # this is the source of our problems 

      isdir("data") || mkpath("data") # ensure a data folder exists
      json_string = JSON.json(existing_data)
      write(s.data_dir, json_string)
    catch e
      println("failed to solve this ode: ", e)
      continue
    end
  end
end

# Generate a specific benchmark from one alpha matrix
function generate_specific_ode_dataset(s::Settings, batch_index::Int, α_matrix::Matrix{Int64})
  ode_order = s.ode_order
  poly_degree = s.poly_degree
  # α_matrix = generate_random_alpha_matrix(s.ode_order, s.poly_degree) # generate ODE matrix
  println("α matrix:")
  display(α_matrix)
  # generate exactly ode_order initial conditions
  initial_conditions = Float64[]
  for i in 0:(ode_order-1)
    if i == 0
      push!(initial_conditions, 1)  # y(0) = a_0, we will set thie init condition to be 1
      # push!(initial_conditions, rand(1:5))  # y(0) = a_0
      println("y(0) = ", initial_conditions[end])
    elseif i == 1
      push!(initial_conditions, rand(1:5))  # y'(0) = a_1
      println("y'(0) = ", initial_conditions[end])
    end
  end
  try
    # output taylor series and its coefficients
    taylor_series, series_coeffs = solve_ode_series_closed_form(α_matrix, initial_conditions, s.num_of_terms) # haha this was the issue 
    println("truncated taylor series: ", taylor_series)
    println("truncated series coefficients: ", series_coeffs)
    # read existing data
    existing_data = if isfile(s.data_dir)
      JSON.parsefile(s.data_dir)
    else
      Dict()
    end
    # Determine which training run this is based on existing data
    dataset_key = lpad(batch_index, 2, '0')
    # Initialize dataset key if it does not exist
    if !haskey(existing_data, dataset_key)
      existing_data[dataset_key] = Dict()
    end
    # use alpha matrix as key, series coefficients as value within the dataset batch
    #  α_matrix_key = join(["[" * join(row, ", ") * "]" for row in eachrow(α_matrix)], "; ")

    existing_data[dataset_key][string(α_matrix)] = series_coeffs # this is the source of our problems 
    isdir("data") || mkpath("data") # ensure a data folder exists
    json_string = JSON.json(existing_data)
    write(s.data_dir, json_string)
  catch e
    println("failed to solve this ode: ", e)
    return nothing
  end
end

# This function will be used for our experiment of taking scalar multiples of the coefficients of one ODE
function generate_ode_dataset_from_array_of_alpha_matrices(s::Settings, batch_index::Int, α_matrices::Array{Matrix{Int64}})
  ode_order = s.ode_order
  poly_degree = s.poly_degree
  # α_matrix = generate_random_alpha_matrix(s.ode_order, s.poly_degree) # generate ODE matrix
  # generate exactly ode_order initial conditions
  initial_conditions = Float64[]
  for i in 0:(ode_order-1)
    if i == 0
      # push!(initial_conditions, rand(1:5))  # y(0) = a_0
      push!(initial_conditions, 1)  # y(0) = a_0, #NOTE:We want to make it so that a_0 = 0 to ensure that all solution vectors are linearly dependent
      println("y(0) = ", initial_conditions[end])
    elseif i == 1
      push!(initial_conditions, rand(1:5))  # y'(0) = a_1
      println("y'(0) = ", initial_conditions[end])
    end
  end
  try
    for matrix in α_matrices
      taylor_series, series_coeffs = solve_ode_series_closed_form(matrix, initial_conditions, s.num_of_terms) # haha this was the issue 
      println("truncated taylor series: ", taylor_series)
      println("truncated series coefficients: ", series_coeffs)
      # read existing data
      existing_data = if isfile(s.data_dir)
        JSON.parsefile(s.data_dir)
      else
        Dict()
      end

      # Determine which training run this is based on existing data
      dataset_key = lpad(batch_index, 2, '0')
      # Initialize dataset key if it does not exist
      if !haskey(existing_data, dataset_key)
        existing_data[dataset_key] = Dict()
      end
      # use alpha matrix as key, series coefficients as value within the dataset batch
      #  α_matrix_key = join(["[" * join(row, ", ") * "]" for row in eachrow(α_matrix)], "; ")

      existing_data[dataset_key][string(matrix)] = series_coeffs # this is the source of our problems 
      isdir("data") || mkpath("data") # ensure a data folder exists
      json_string = JSON.json(existing_data)
      write(s.data_dir, json_string)
    end
  catch e
    println("failed to solve this ode: ", e)
    return nothing
  end
end

export Settings, generate_random_ode_dataset, generate_specific_ode_dataset, solve_ode_series_closed_form, generate_ode_dataset_from_array_of_alpha_matrices
end
