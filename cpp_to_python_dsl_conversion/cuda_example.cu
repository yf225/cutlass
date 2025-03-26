// nvcc cuda_example.cu -o main -ftemplate-depth=1000

#include <cuda_runtime.h>
#include <iostream>

// This code is intentionally designed to compile extremely slowly 
// by heavily using C++ template meta-programming.
// You may need to adjust your compiler's template recursion limit 
// (e.g., using -ftemplate-depth=6000 with nvcc) when compiling.

// A recursive template that instantiates a chain of 5000 recursive calls.
template <int N>
struct HeavyRecursive {
    static constexpr int value = HeavyRecursive<N - 1>::value + 1;
};

template <>
struct HeavyRecursive<0> {
    static constexpr int value = 0;
};

// A template that uses the heavy recursive template to compute a compile-time multiplier.
template <int N>
struct CompileTimeMultiplier {
    // Multiply the result of the heavy recursion by N
    static constexpr int value = HeavyRecursive<N>::value * N;
};

// Variadic template to compute the sum of a list of integers.
template <typename T, T... Values>
struct VariadicSum;

template <typename T, T First, T... Rest>
struct VariadicSum<T, First, Rest...> {
    static constexpr T value = First + VariadicSum<T, Rest...>::value;
};

template <typename T>
struct VariadicSum<T> {
    static constexpr T value = 0;
};

// A simple CUDA kernel that uses the compile-time computed values.
__global__ void kernel(int *data) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Use the heavy compile-time computed values:
    //   CompileTimeMultiplier<5000>::value instantiates the heavy recursion,
    //   and VariadicSum<int, 1,2,3,4,5>::value adds a small constant.
    data[idx] = CompileTimeMultiplier<100>::value + VariadicSum<int, 1, 2, 3, 4, 5>::value;
}

int main() {
    const int N = 256;
    int *d_data;
    int h_data[N];
    
    cudaMalloc(&d_data, N * sizeof(int));
    kernel<<<1, N>>>(d_data);
    cudaDeviceSynchronize();
    cudaMemcpy(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    
    std::cout << "Result: " << h_data[0] << std::endl;
    return 0;
}