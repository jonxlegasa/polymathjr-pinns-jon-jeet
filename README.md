# PINN Power Series ODE Solver

- A Julia implementation of a Physics-Informed Neural Network (PINN) that approximates the solution of a given ODE via a truncated power series.
- Trains the PINN on different datasets and saves them into separated training runs.


| Folder | What’s inside |
|--------|---------------|
| `src/` | **`main.jl`** - the main script|
| `scripts/` | Contains architectures for NN (RNN, Feed forward, etc) |
| `data/` | output goes here (`.jld2`, `.png`, `.gif`, `.txt`, `.json`) |

## Quick start
```bash
# 1. grab the code
git clone git@github.com:jonxlegasa/polymathjr-pinns-jon-jeet.git
cd polymathjr-pinns-jon-jeet

# 2. download the dependencies once
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# 3. run the demo simulation (saves data & figs to ./data/training-run-#)
julia --project=. src/main.jl
```
