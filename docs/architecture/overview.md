# System Overview

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                     main.jl (Orchestrator)                    │
└───────────────┬─────────────────────┬────────────────────────┘
                │                     │
    ┌───────────▼───────────┐   ┌─────▼─────────────────┐
    │   plugboard.jl        │   │      PINN.jl          │
    │   (Dataset Gen)       │   │   (Core Training)     │
    └───────────┬───────────┘   └─────────┬─────────────┘
                │                         │
                ▼                         ▼
    ┌───────────────────────┐   ┌─────────────────────────┐
    │  training_dataset.json│   │  data/training-run-N/   │
    └───────────────────────┘   └──────────┬──────────────┘
                                           │
                                ┌──────────▼──────────┐
                                │   visualizer.py     │
                                └─────────────────────┘
```

---

## Neural Network Architecture

```
Input: [ODE coefficients, initial conditions]
    ↓
Dense(input → neurons, activation) x 6 hidden layers
    ↓
Dense(neurons → output_size)
    ↓
Output: power series coefficients
```

---

## Training Pipeline

1. **Detect device** — auto-selects GPU (CUDA) or CPU
2. **Initialize** network with random parameters (transferred to GPU if available)
3. **Adam** optimization (configurable iterations; LBFGS under investigation)
4. **Evaluate** on benchmark dataset (always on CPU)
5. **Save** plots and loss history

---

*See also: [Data Flow](data-flow.md), [PINN.jl](../julia-modules/pinn.md)*
