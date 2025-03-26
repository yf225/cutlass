// nvcc cuda_example_2.cu -o main_2

#include <cstdio>
#include <cuda_runtime.h>

// Recursive template structure to force multiple instantiations
template <int Depth, int Branch>
struct Torpedo {
    static __device__ __host__ void explode() {
        Torpedo<Depth - 1, Branch>::explode();
        Torpedo<Depth - 1, Branch + 1>::explode();
    }
};

// Template base case
template <int Branch>
struct Torpedo<0, Branch> {
    static __device__ __host__ void explode() {
        printf("Boom %d!\n", Branch);
    }
};

// Template CUDA kernel with multiple parameters
template <int Depth, int Branch>
__global__ void NuclearKernel() {
    Torpedo<Depth, Branch>::explode();
};

// Compile-time unrolled loop for kernel launches
template <int N>
struct Launcher {
    template <int D>
    static void Deploy() {
        Launcher<N - 1>::template Deploy<D>();
        NuclearKernel<D, N><<<1, 1>>>();
        cudaDeviceSynchronize();
    }
};

template <>
struct Launcher<0> {
    template <int D>
    static void Deploy() {
        NuclearKernel<D, 0><<<1, 1>>>();
        cudaDeviceSynchronize();
    }
};

int main() {
    // Each increment here increases template instantiations exponentially
    // Try changing 5 to higher numbers (warning: compilation time grows rapidly!)
    Launcher<10>::Deploy<20>();
    return 0;
}
