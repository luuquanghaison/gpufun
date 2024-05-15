using KernelAbstractions
using CUDA

# write kernel
@kernel function mul2_kernel(A)
    I = @index(Global)
    A[I] = 2 * A[I]
end

A = CuArray(ones(1024, 1024))

# Lauch kernel with CUDA backend
backend = get_backend(A)
mul2_kernel(backend, 64)(A, ndrange=size(A))
KernelAbstractions.synchronize(backend)
all(A .== 2.0)