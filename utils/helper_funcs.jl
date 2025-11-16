module helper_funcs

using CSV
using DataFrames

function convert_plugboard_keys(inner_dict)
  converted_dict = Dict{Matrix{Int}, Any}()
  for (alpha_matrix_key, series_coeffs) in inner_dict
    # println("The current  ODE I am calculating the loss for right now: ", alpha_matrix_key)
    # println("The local loss is locally lossing...")
    alpha_matrix = eval(Meta.parse(alpha_matrix_key)) # convert from string to matrix 
    converted_dict[alpha_matrix] = series_coeffs
  end

  return converted_dict
end

function create_csv_file_for_loss(csv_file; kwargs...)
  # Create directory if needed
  dir = dirname(csv_file)
  if !isdir(dir)
    println("CSV File created")
    mkpath(dir)
  end

  # Read or initialize DataFrame
  if isfile(csv_file) && filesize(csv_file) > 0
    df = CSV.read(csv_file, DataFrame)
  else
    # Initialize with loss_type column
    df = DataFrame(loss_type = String[])
  end

  # Determine next iteration column name
  existing_cols = names(df)
  iter_cols = filter(name -> startswith(name, "iter_"), existing_cols)
  next_iter = length(iter_cols) + 1
  new_col_name = "iter_$(next_iter)"

  # Add new column with missing values for existing rows
  # df[!, new_col_name] = fill(missing, nrow(df))

  df[!, new_col_name] = Vector{Union{Missing, Float64}}(missing, nrow(df))  # ← FIXED
  # Add or update each loss type
  for (key, value) in kwargs
    loss_name = String(key)
    # Find if this loss type already exists
    row_idx = findfirst(df.loss_type .== loss_name)
    if isnothing(row_idx)
      # New loss type - add new row
      new_row = DataFrame(loss_type = [loss_name])
      # Fill previous iterations with missing
      for col in iter_cols
        new_row[!, col] = Union{Missing, Float64}[missing]
      end
      new_row[!, new_col_name] = Union{Missing, Float64}[Float64(value)]

      append!(df, new_row, promote=true)
    else
      # Existing loss type - update the cell
      df[row_idx, new_col_name] = Float64(value)
    end
  end
  # Write to file
  CSV.write(csv_file, df)
end

export create_csv_file_for_loss, convert_plugboard_keys

end
