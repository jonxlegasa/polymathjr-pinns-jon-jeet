# Training Workflow

Complete pipeline from ODE definition to trained model.

---

## Pipeline Overview

```
1. Dataset Generation
   └─→ plugboard.jl generates training_dataset.json

2. Initialization
   └─→ Load datasets, create PINNSettings

3. Device Detection
   └─→ Auto-detect GPU (CUDA) or fall back to CPU

4. Network Setup
   └─→ initialize_network() creates Lux model, transfers to GPU if available

5. Adam Training (LBFGS under investigation)
   └─→ Configurable iterations, optimization

6. Evaluation
   └─→ Parameters transferred back to CPU, benchmark testing, plot generation

7. Output
   └─→ CSV loss history, PNG plots
```

---

## Detailed Steps

### Step 1: Dataset Generation

```julia
# In plugboard.jl
settings = Settings(
    ode_order = 2,
    poly_degree = 0,
    dataset_size = 10,
    num_of_terms = 15
)
generate_random_ode_dataset(settings, batch_index)
```

**Output:** `training_dataset.json` with ODEs and true coefficients

---

### Step 2: Create PINNSettings

```julia
settings = PINNSettings(
    neuron_num = 50,
    seed = 42,
    ode_matrices = load_training_data(),
    maxiters_adam = 10000,
    n_terms_for_power_series = 10,
    # ... other parameters
)
```

---

### Step 3: Device Detection & Network Setup

```julia
# Automatic GPU detection in train_pinn()
use_gpu = GPUUtils.is_gpu_available()  # true if CUDA functional

coeff_net, p_net, st = initialize_network(settings; use_gpu=use_gpu)
```

**Architecture:**
- Input: flattened ODE matrix + initial conditions
- Hidden: 4 layers × `neuron_num` neurons (σ activation)
- Output: N+1 coefficients
- Parameters transferred to GPU via `CUDA.cu()` when available

---

### Step 4: Adam Training

```julia
# In train_pinn()
opt_func = OptimizationFunction(loss_wrapper, AutoZygote())
prob = OptimizationProblem(opt_func, p_net)
result = solve(prob, Adam(0.001f0), maxiters=settings.maxiters_lbfgs)
```

**Purpose:** Optimization with configurable iteration count. Loss functions use vectorized matrix operations for GPU compatibility (no scalar indexing).

> **Note:** LBFGS code is present in `train_pinn` but currently commented out. Further work is needed to understand its convergence behavior before re-enabling.

---

### Step 6: Evaluation

```julia
evaluate_solution(
    settings,
    p_trained,
    coeff_net,
    st,
    benchmark_dataset,
    output_directory
)
```

**Generates:**
- Solution comparison plot
- Coefficient comparison plot
- Error analysis plot

---

### Step 7: Output Files

```
data/training-run-N/batch-01/
├── iteration_output.csv
├── solution_plot.png
├── coefficient_plot.png
└── error_plot.png
```

---

## Loss During Training

Each iteration logs:

| Component | Description |
|-----------|-------------|
| `total_loss` | Weighted sum of all losses |
| `total_loss_pde` | ODE residual loss |
| `total_loss_bc` | Boundary condition loss |
| `total_loss_supervised` | Coefficient MSE |

---

## Checkpointing

Currently, training runs to completion without intermediate checkpoints. For long runs, consider:
- Reducing `maxiters_adam` for testing
- Monitoring loss CSV for convergence

---

*See also: [PINN.jl](../julia-modules/pinn.md)*
