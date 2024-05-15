using CUDA 
using BenchmarkTools 


N = 2^20
# First kernels
x_d = CUDA.fill(1.0f0, N)  # a vector stored on the GPU filled with 1.0 (Float32)
y_d = CUDA.fill(2.0f0, N)  # a vector stored on the GPU filled with 2.0

# bad kernel
function gpu_add1!(y, x)
    for i in 1:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

function bench_gpu1!(y, x)
    CUDA.@sync begin
        @cuda gpu_add1!(y, x)
    end
    return nothing
end

println("Bad kernel")
@btime bench_gpu1!($y_d, $x_d)

# good kernel
function gpu_add3!(y, x)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(y)
        @inbounds y[i] += x[i]
    end
    return nothing
end

numblocks = ceil(Int, N/256)

function bench_gpu3!(y, x)
    numblocks = ceil(Int, length(y)/256)
    CUDA.@sync begin
        @cuda threads=256 blocks=numblocks gpu_add3!(y, x)
    end
    return nothing
end

println("Good kernel")
@btime bench_gpu3!($y_d, $x_d)

## Steps in kernel programming
# Create CuArarrays on GPU
# a = CUDA.zeros(1024)

# # Write kernels (functions that apllies to each thread)
# function kernel(a)
#     i = threadIdx().x
#     a[i] += 1
#     return
# end

# # Launch kernels in parallel using @cuda
# @cuda threads=length(a) kernel(a)