module loss_functions
using CSV
using DataFrames
using Zygote: Zygote

include("./helper_funcs.jl")
using .helper_funcs

# Parametric struct: V can be Vector{Float32} (CPU) or CuVector{Float32} (GPU)
struct LossFunctionSettings{V<:AbstractVector{Float32}}
  a_vec::V
  n_terms_for_power_series::Int
  ode_matrix_flat::V
  x_left::Float32
  boundary_condition::V
  xs::V
  num_points::Int
  num_supervised::Int
  data::V
end

# Pre-compute inverse factorials in Float32 for GPU-compatible loss computation
# 21! fits in Float64; 1/21! ≈ 1.95e-20 is above Float32 minimum
const INV_FACT = Float32.(1.0 ./ factorial.(big.(0:40)))

# Keep BigFloat factorials for evaluation/plotting code (CPU only)
fact = factorial.(big.(0:21))

"""
    generate_loss_pde_value(settings)

GPU-compatible PDE loss using vectorized matrix operations.
Precomputes composite operator W (constant w.r.t. a_vec) so that
residual = W * a_vec is a single differentiable matrix-vector multiply.
"""
function generate_loss_pde_value(settings::LossFunctionSettings)
  N1 = settings.n_terms_for_power_series + 1

  # Build W matrix inside Zygote.ignore — it doesn't depend on a_vec
  # so Zygote correctly treats it as a constant
  W = Zygote.ignore() do
    xs_cpu = Float32.(collect(settings.xs))
    ode_cpu = Float32.(collect(settings.ode_matrix_flat))
    P = length(xs_cpu)
    M = length(ode_cpu)

    # W[j, i] = Σ_{k=0}^{min(i-1, M-1)} ode[k+1] * xs[j]^(i-1-k) * INV_FACT[i-k]
    W_cpu = zeros(Float32, P, N1)
    for j in 1:P
      for i in 1:N1
        for k in 0:min(i - 1, M - 1)
          W_cpu[j, i] += ode_cpu[k + 1] * xs_cpu[j]^(i - 1 - k) * INV_FACT[i-k]
        end
      end
    end

    # Transfer to same device as a_vec
    W_dev = similar(settings.a_vec, P, N1)
    copyto!(W_dev, W_cpu)
    W_dev
  end

  # Single differentiable operation — Zygote handles this on both CPU and GPU
  residual = W * settings.a_vec
  return sum(abs2, residual) / settings.num_points
end

"""
    generate_loss_bc_value(settings)

GPU-compatible boundary condition loss.
Uses full-length dot products with precomputed power vectors
to avoid scalar indexing of a_vec.
"""
function generate_loss_bc_value(settings::LossFunctionSettings)
  N1 = settings.n_terms_for_power_series + 1
  x0 = settings.x_left

  # Precompute power vectors (constants w.r.t. a_vec)
  pow_u, pow_du_full, bc1, bc2 = Zygote.ignore() do
    # u(x0) weights: x0^(i-1) * INV_FACT[i]
    pow_u_cpu = Float32[x0^(i - 1) * INV_FACT[i] for i in 1:N1]

    # Du(x0) weights, padded to full length (index 1 = 0) to avoid slicing a_vec
    pow_du_cpu = Float32[i == 1 ? 0.0f0 : x0^(i - 2) * INV_FACT[i-1] for i in 1:N1]

    # Transfer to same device as a_vec
    pu = similar(settings.a_vec)
    copyto!(pu, pow_u_cpu)
    pdu = similar(settings.a_vec)
    copyto!(pdu, pow_du_cpu)

    # Extract boundary condition scalars to CPU
    bc_cpu = Float32.(collect(settings.boundary_condition))
    (pu, pdu, bc_cpu[1], bc_cpu[2])
  end

  a = settings.a_vec
  u_val = sum(a .* pow_u)
  du_val = sum(a .* pow_du_full)

  loss_bc = abs(u_val - bc1) + abs(du_val - bc2)
  return loss_bc
end

"""
    generate_loss_supervised_value(settings)

GPU-compatible supervised loss.
Uses padded data + mask to avoid a_vec[1:K] slicing
which triggers Zygote's broken ∇getindex! on GPU.
"""
function generate_loss_supervised_value(settings::LossFunctionSettings)
  K = settings.num_supervised
  N1 = length(settings.a_vec)

  padded_data, mask = Zygote.ignore() do
    d_cpu = Float32.(collect(settings.data))
    pd_cpu = zeros(Float32, N1)
    m_cpu = zeros(Float32, N1)
    pd_cpu[1:min(K, length(d_cpu))] .= d_cpu[1:min(K, length(d_cpu))]
    m_cpu[1:K] .= 1.0f0

    pd = similar(settings.a_vec)
    copyto!(pd, pd_cpu)
    m = similar(settings.a_vec)
    copyto!(m, m_cpu)
    (pd, m)
  end

  diff = (settings.a_vec - padded_data) .* mask
  return sum(abs2, diff) / K
end

# Keep scalar versions for evaluation/plotting (CPU only, not used in training gradient path)
function ode_residual(settings::LossFunctionSettings, x)
  return sum(
    settings.ode_matrix_flat[order+1] * (
      order == 0 ?
      sum(settings.a_vec[i] * x^(i - 1) / fact[i] for i in 1:settings.n_terms_for_power_series+1) :
      sum(
        settings.a_vec[i] * x^(i - 1 - order) / fact[i-order]
        for i in (order+1):(settings.n_terms_for_power_series+1)
      )
    )
    for order in 0:(length(settings.ode_matrix_flat)-1)
  )
end

function generate_u_approx(settings::LossFunctionSettings)
  u_approx(x) = sum(settings.a_vec[i] * x^(i - 1) / fact[i] for i in 1:(settings.n_terms_for_power_series+1))
  return u_approx
end

export LossFunctionSettings, generate_loss_pde_value, generate_loss_bc_value, generate_loss_supervised_value, ode_residual, generate_u_approx

end
