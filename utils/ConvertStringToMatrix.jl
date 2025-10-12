module ConvertStringToMatrix

struct StringToMatrixSettings
  inner_dict::Any
end

function convert(s::StringToMatrixSettings)
  converted_dict = Dict{Matrix{Int}, Any}()
  for (alpha_matrix_key, series_coeffs) in s.inner_dict
    # println("The current  ODE I am calculating the loss for right now: ", alpha_matrix_key)
    # println("The local loss is locally lossing...")
    alpha_matrix = eval(Meta.parse(alpha_matrix_key)) # convert from string to matrix 
    converted_dict[alpha_matrix] = series_coeffs
  end

  return converted_dict
end

export StringToMatrixSettings, convert

end
