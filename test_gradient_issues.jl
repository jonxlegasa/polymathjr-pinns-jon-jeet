using Zygote

F = Float32

# Test: BC loss non-smoothness
println("=" ^ 70)
println("TEST 1: BC loss uses abs() - non-smooth")
println("=" ^ 70)

function bc_loss_only(a_vec)
    # BC loss from loss_functions.jl lines 98-103
    pow_u = Float32[1.0, 0.5, 0.2]
    pow_du = Float32[0.5, 0.2, 0.1]
    u_val = sum(a_vec .* pow_u)
    du_val = sum(a_vec .* pow_du)
    loss_bc = abs(u_val - F(2.0)) + abs(du_val - F(1.0))
    return loss_bc
end

println("Testing BC loss gradient at various points:")
test_points = [
    (Float32[1.0, 0.5, 0.2], "generic point"),
    (Float32[2.0, 0.0, 0.0], "near BC boundary"),
]

for (a_test, desc) in test_points
    println("\n  $desc: a = $a_test")
    loss_val = bc_loss_only(a_test)
    println("    Loss: $loss_val")
    
    try
        g = gradient(bc_loss_only, a_test)[1]
        if g === nothing
            println("    ERROR: gradient is nothing!")
        else
            println("    Gradient: $g")
        end
    catch e
        println("    ERROR: $e")
    end
end

println("\n  Critical issue: At points near BC boundary, abs() has:")
println("    - Zero gradient exactly at boundary (u_val == bc1)")
println("    - Discontinuous 2nd derivative (Hessian undefined)")
println("    - This breaks LBFGS Hessian approximation")

# Test: Loss accumulation in global_loss
println("\n" ^ 2)
println("=" ^ 70)
println("TEST 2: Loss accumulation and Zygote differentiation")
println("=" ^ 70)

function simple_accumulated_loss(local_loss_1, local_loss_2, local_loss_3)
    # Mimics lines 219-222 of PINN.jl
    total_loss = F(0.0)
    total_loss += local_loss_1
    total_loss += local_loss_2
    total_loss += local_loss_3
    return total_loss
end

l1 = F(5.5e15)
l2 = F(3.2e15)
l3 = F(1.1e15)

println("Testing accumulated loss gradient:")
println("  Input losses: $l1, $l2, $l3")

try
    g = gradient((a, b, c) -> simple_accumulated_loss(a, b, c), l1, l2, l3)
    println("  Gradients: $g")
    println("  Expected: (1.0, 1.0, 1.0)")
except e
    println("  ERROR: $e")
end

# Test: Combining PDE (smooth) + BC (non-smooth)
println("\n" ^ 2)
println("=" ^ 70)
println("TEST 3: PDE loss (smooth) vs BC loss (non-smooth) interaction")
println("=" ^ 70)

# PDE loss is smooth
function pde_loss(a_vec)
    residual = a_vec  # Simplified: assumes W*a_vec ≈ a_vec
    return sum(abs2, residual) / length(a_vec)
end

# BC loss is non-smooth
function bc_loss(a_vec)
    pow_u = Float32[1.0, 0.5, 0.2]
    pow_du = Float32[0.5, 0.2, 0.1]
    u_val = sum(a_vec .* pow_u)
    du_val = sum(a_vec .* pow_du)
    return abs(u_val - F(2.0)) + abs(du_val - F(1.0))
end

function combined_loss(a_vec)
    return 1.0f0 * pde_loss(a_vec) + 1.0f0 * bc_loss(a_vec)
end

a_test = Float32[1.0, 0.5, 0.2]
println("Testing combined loss (PDE + BC) at a = $a_test:")

try
    g_pde = gradient(pde_loss, a_test)[1]
    println("  PDE gradient: $g_pde (smooth)")
    
    g_bc = gradient(bc_loss, a_test)[1]
    println("  BC gradient:  $g_bc (non-smooth from abs())")
    
    g_combined = gradient(combined_loss, a_test)[1]
    println("  Combined gradient: $g_combined")
    println("\n  Problem: Combined gradient inherits non-smoothness from BC!")
    println("  LBFGS line search fails when following non-smooth gradient directions")
except e
    println("  ERROR: $e")
end

# Summary of findings
println("\n" ^ 2)
println("=" ^ 70)
println("ANALYSIS SUMMARY: Why LBFGS fails but Adam works")
println("=" ^ 70)

summary = """
1. BC Loss Non-Smoothness (Lines 102 in loss_functions.jl):
   Current:  loss_bc = abs(u_val - bc1) + abs(du_val - bc2)
   Problem: abs() has zero 2nd derivative (Hessian is undefined/infinite)
   Impact:  LBFGS uses Hessian approximation - breaks at non-smooth points
   
2. Gradient Discontinuity:
   - At u_val == bc1: gradient sign jumps from -1 to +1
   - LBFGS line search assumes smooth quadratic landscape
   - Adam uses adaptive learning - ignores these issues
   
3. Global Loss Accumulation (Lines 219-222 in PINN.jl):
   Problem: All three losses combined in a loop
   When BC loss dominates and is non-smooth, entire combined gradient
   inherits the non-smoothness, breaking LBFGS's assumptions
   
4. Evidence: Your results
   - LBFGS: stuck at 5.5e15 after ~15 iterations (loss scale 1e15)
   - Adam: converges normally
   - This is typical of optimizer hitting non-smooth landscape
   
5. Recommended Fix:
   Replace abs() with smooth approximation:
   - Use (u_val - bc1)^2 + (du_val - bc2)^2 for smooth 2nd derivatives
   - Or use Huber loss: smooth near zero, linear at distance
   - Or manually set BC using constraints (not soft losses)
"""

println(summary)

