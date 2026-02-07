using Zygote
using CUDA

# Test 1: Check if abs() in loss_bc causes gradient discontinuity
println("=" ^ 60)
println("TEST 1: Gradient discontinuity in loss_bc with abs()")
println("=" ^ 60)

# This mimics the BC loss computation
function loss_bc_with_abs(a_vec, pow_u, pow_du, bc1, bc2)
    u_val = sum(a_vec .* pow_u)
    du_val = sum(a_vec .* pow_du)
    loss_bc = abs(u_val - bc1) + abs(du_val - bc2)
    return loss_bc
end

# Test with gradient
a_test = Float32[1.0, 2.0, 3.0]
pow_u = Float32[1.0, 0.5, 0.2]
pow_du = Float32[0.5, 0.2, 0.1]
bc1 = Float32(2.0)
bc2 = Float32(1.0)

try
    grad = gradient(a -> loss_bc_with_abs(a, pow_u, pow_du, bc1, bc2), a_test)[1]
    println("Gradient at a=$a_test:")
    println("  $grad")
    println("  Issue: abs() has zero derivative at boundary (u_val == bc1 or du_val == bc2)")
    println("  LBFGS uses second derivatives; zero gradients are problematic")
catch e
    println("Error computing gradient: $e")
end

# Test 2: Check scalar accumulation type issues in global_loss
println("\n" ^ 2)
println("=" ^ 60)
println("TEST 2: Type mismatch in loss accumulation (Float32 vs Float64)")
println("=" ^ 60)

F = Float32
total_loss = F(0.0)
local_loss_1 = Float32(5.5e15)
local_loss_2 = Float32(3.2e15)

total_loss += local_loss_1
println("After adding loss_1 ($local_loss_1): $total_loss")
total_loss += local_loss_2
println("After adding loss_2 ($local_loss_2): $total_loss")

# Check if this causes issues with gradients
function accumulate_losses(losses)
    total = F(0.0)
    for l in losses
        total += l
    end
    return total
end

test_losses = Float32[1e15, 2e15, 3e15]
accumulated = accumulate_losses(test_losses)
println("\nAccumulated losses: $accumulated")

try
    g = gradient(losses -> accumulate_losses(losses), test_losses)[1]
    println("Gradient of accumulated losses: $g")
    println("Expected: ones (since d/d(loss[i]) accumulated_loss = 1 for all i)")
catch e
    println("Error: $e")
end

# Test 3: Check Zygote.ignore() impact on gradient flow
println("\n" ^ 2)
println("=" ^ 60)
println("TEST 3: Zygote.ignore() handling in loss computation")
println("=" ^ 60)

function loss_with_ignore(a_vec)
    # Simulate matrix construction in Zygote.ignore
    W = Zygote.ignore() do
        zeros(Float32, 2, 3)  # Constant w.r.t. a_vec
    end
    
    # This should differentiate w.r.t. a_vec
    residual = W * a_vec  # W is constant, a_vec is variable
    return sum(abs2, residual)
end

try
    a_test = Float32[1.0, 2.0, 3.0]
    g = gradient(a -> loss_with_ignore(a), a_test)[1]
    println("Gradient computed successfully: $g")
catch e
    println("Error: $e")
end

# Test 4: Check if matrix-vector multiply has proper gradient
println("\n" ^ 2)
println("=" ^ 60)
println("TEST 4: Gradient through matrix-vector multiply (W * a)")
println("=" ^ 60)

function test_matmul_gradient(a_vec)
    W = Float32[1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3x2 matrix
    residual = W * a_vec  # 3x1 result
    loss = sum(abs2, residual)
    return loss
end

a_test = Float32[1.0, 2.0]
try
    g = gradient(test_matmul_gradient, a_test)[1]
    println("Gradient through W*a: $g")
    println("This is differentiable and should work fine with LBFGS")
catch e
    println("Error: $e")
end

# Test 5: Check conditional behavior (comparing with abs)
println("\n" ^ 2)
println("=" ^ 60)
println("TEST 5: Non-smooth regions - abs() vs squared difference")
println("=" ^ 60)

# Using abs (current implementation)
function loss_abs(a)
    return abs(a - 2.0)
end

# Using squared difference (smooth)
function loss_squared(a)
    return (a - 2.0)^2
end

a_vals = Float32[-1.0, 0.0, 1.0, 1.99, 1.999, 2.0, 2.001, 2.1, 3.0]
println("a\t\tloss_abs(a)\t\t∇loss_abs(a)\t\tloss_sq(a)\t\t∇loss_sq(a)")
println("-" ^ 80)

for a in a_vals
    try
        grad_abs = gradient(loss_abs, a)[1]
        grad_sq = gradient(loss_squared, a)[1]
        println("$a\t\t$(loss_abs(a))\t\t$grad_abs\t\t$(loss_squared(a))\t\t$grad_sq")
    catch e
        println("$a\t\tError computing gradient")
    end
end

println("\nKEY OBSERVATION: abs() has undefined gradient at a=2.0 (derivative jumps from -1 to +1)")
println("LBFGS relies on smooth gradients. Adam is robust to noisy/undefined gradients.")

