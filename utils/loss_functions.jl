module loss_functions
using CSV
using DataFrames
using Zygote

include("./helper_funcs.jl")
using .helper_funcs

struct LossFunctionSettings
  a_vec::Vector{Float32}
  n_terms_for_power_series::Int
  ode_matrix_flat::Vector{Float32}
  x_left::Float32
  boundary_condition::Float32
  xs::Vector{Float32}
  num_points::Int
  num_supervised::Int
  data::Vector{Float32}
end

function ode_residual(settings::LossFunctionSettings, x)
  return sum(
    settings.ode_matrix_flat[order + 1] * (
      order == 0 ? 
      sum(settings.a_vec[i] * x^(i - 1) for i in 1:settings.n_terms_for_power_series+1) :
      sum(
          prod((i - 1 - j) for j in 0:(order - 1)) * settings.a_vec[i] * x^(i - 1 - order)
          for i in (order + 1):(settings.n_terms_for_power_series + 1)
      )
    )
    for order in 0:(length(settings.ode_matrix_flat) - 1)
  )
end

# Updated functions using the settings struct
function generate_u_approx(settings::LossFunctionSettings)
  u_approx(x) = sum(settings.a_vec[i] * x^(i - 1) for i in 1:(settings.n_terms_for_power_series + 1))
  return u_approx
end

function generate_loss_pde_value(settings::LossFunctionSettings)
  loss_pde = sum(
    abs2,
    ode_residual(settings, xi)
    for xi in settings.xs
  ) / settings.num_points

  return loss_pde
end

function generate_loss_bc_value(settings::LossFunctionSettings)
  u_approx = generate_u_approx(settings)
  loss_bc = abs2(u_approx(settings.x_left) - settings.boundary_condition)

  return loss_bc
end

function generate_loss_supervised_value(settings::LossFunctionSettings)
  loss_supervised = sum(abs2, settings.a_vec[1:settings.num_supervised] - settings.data) / settings.num_supervised

  return loss_supervised
end

export LossFunctionSettings, generate_loss_pde_value, generate_loss_bc_value, generate_loss_supervised_value, ode_residual, generate_u_approx

end
