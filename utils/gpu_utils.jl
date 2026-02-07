module GPUUtils

using CUDA

export is_gpu_available, get_gpu_device, to_gpu, to_device, get_device

const GPU_AVAILABLE = Ref{Bool}(false)
const DEVICE = Ref{Union{CuDevice, Nothing}}(nothing)

function __init__()
    GPU_AVAILABLE[] = CUDA.functional()
    if GPU_AVAILABLE[]
        DEVICE[] = CUDA.device()
        # Disable TF32 on Ampere+ GPUs — TF32 reduces mantissa from 23 to 10 bits,
        # which destabilizes LBFGS line search and can cause early termination
        CUDA.math_mode!(CUDA.PEDANTIC_MATH)
        @info "GPU detected: $(CUDA.name(DEVICE[])) (TF32 disabled for numerical stability)"
    else
        @warn "GPU not available, falling back to CPU"
    end
end

is_gpu_available() = GPU_AVAILABLE[]

function get_gpu_device()
    return DEVICE[]
end

function to_gpu(x::AbstractArray)
    @assert is_gpu_available() "GPU not available"
    return CUDA.cu(Float32.(x))
end

"""
    to_device(x; gpu=false)

Transfer array to GPU or keep on CPU based on flag.
Converts to Float32 for neural network compatibility.
"""
function to_device(x::AbstractArray; gpu::Bool=false)
    if gpu && is_gpu_available()
        return CUDA.cu(Float32.(x))
    else
        return Float32.(collect(x))
    end
end

function get_device()
    if is_gpu_available()
        return CUDA.name(CUDA.device())
    else
        return "CPU"
    end
end

end
