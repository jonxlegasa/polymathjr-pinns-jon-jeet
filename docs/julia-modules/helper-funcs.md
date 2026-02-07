# helper_funcs.jl

Utility functions for data conversion and results output.

**Location:** `utils/helper_funcs.jl`

---

## Data Conversion

### `convert_plugboard_keys(inner_dict)`

Converts JSON string keys back to Julia matrices.

```julia
convert_plugboard_keys(dict) → Dict
```

---

## Math Utilities

### `quadratic_formula(a, b, c)`

Solves quadratic equations. Used for computing analytical ODE solutions.

```julia
quadratic_formula(a, b, c) → (root1, root2)
```

---

### `generate_valid_matrix()`

Generates random 3x1 matrix satisfying the real roots constraint.

### `create_matrix_array(n)`

Creates array of n random valid matrices.

```julia
create_matrix_array(n::Int) → Vector{Matrix}
```

---

*See also: [plugboard.jl](plugboard.md), [Output CSV](../data-formats/output-csv.md)*
