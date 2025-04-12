#include "cutlass/core_io.h"
#include <cuda_runtime_api.h>
#include <cuda/atomic>

#include "cutlass/profiler/cublas_helpers.h"
#include "cutlass/profiler/gemm_operation_profiler.h"
#include "cutlass/profiler/gpu_timer.h"
#include "cutlass/library/singleton.h"
#include "cutlass/library/library.h"
#include "cutlass/library/handle.h"

// LibTorch includes
#include <torch/torch.h>

namespace cutlass {
namespace profiler {

// Helper function to map CUTLASS NumericTypeID to Torch ScalarType
torch::ScalarType CutlassTypeToTorchType(cutlass::library::NumericTypeID cutlass_type) {
    using cutlass::library::NumericTypeID;
    switch (cutlass_type) {
        case NumericTypeID::kF64: return torch::kFloat64;
        case NumericTypeID::kF32: return torch::kFloat32;
        case NumericTypeID::kF16: return torch::kFloat16;
        case NumericTypeID::kBF16: return torch::kBFloat16;
        case NumericTypeID::kFE4M3: return torch::kFloat8_e4m3fn;
        case NumericTypeID::kFE5M2: return torch::kFloat8_e5m2;
        default:
            throw std::runtime_error("Unknown or unsupported CUTLASS NumericTypeID: " + std::to_string(int(cutlass_type)));
    }
}

// Helper function to get element size in bytes
size_t GetElementSizeInBytes(cutlass::library::NumericTypeID cutlass_type) {
    size_t bits = cutlass::library::sizeof_bits(cutlass_type);
    if (bits == 0) {
        throw std::runtime_error("Could not determine size for CUTLASS NumericTypeID");
    }
    if (bits % 8 != 0) {
         throw std::runtime_error("Cannot handle types with size not divisible by 8 bits");
    }
    return bits / 8;
}


// Function to perform the conversion and save
void convert_and_save_as_torch_tensor(
    cutlass::profiler::DeviceAllocation* device_alloc,
    const std::string& output_filename)
{
    // --- Step 1: Extract information from DeviceAllocation ---
    if (!device_alloc->good()) {
        throw std::runtime_error("Input DeviceAllocation is not valid.");
    }

    void* data_ptr = device_alloc->data(); // Pointer to device memory
    cutlass::library::NumericTypeID cutlass_type = device_alloc->type();
    cutlass::library::LayoutTypeID cutlass_layout = device_alloc->layout(); // Get layout
    std::vector<int> cutlass_extent = device_alloc->extent(); // Dimensions/Shape
    std::vector<int64_t> cutlass_strides = device_alloc->stride(); // Strides in elements

    // --- Step 2: Convert metadata for LibTorch ---
    torch::ScalarType torch_type = CutlassTypeToTorchType(cutlass_type);
    size_t element_size_bytes = GetElementSizeInBytes(cutlass_type);

    // Convert extent (shape) to int64_t vector
    std::vector<int64_t> torch_sizes;
    torch_sizes.reserve(cutlass_extent.size());
    for (int dim : cutlass_extent) {
        torch_sizes.push_back(static_cast<int64_t>(dim));
    }

    // Convert strides from elements to bytes
    std::vector<int64_t> torch_strides;
    torch_strides.reserve(cutlass_strides.size());
    for (int64_t stride : cutlass_strides) {
        torch_strides.push_back(stride);
    }

    // If extent has 2 dims but strides only has 1, calculate missing stride based on layout
    if (torch_sizes.size() == 2 && torch_strides.size() == 1) {
        int64_t provided_stride = torch_strides[0];
        if (cutlass_layout == cutlass::library::LayoutTypeID::kRowMajor) {
            std::cout << "Info: Detected 2D RowMajor tensor with 1 stride provided. Assuming inner stride is 1." << std::endl;
            // For RowMajor [M, N], strides are [N, 1]. We have N, need to add 1.
             if (provided_stride != torch_sizes[1]) {
                 std::cerr << "Warning: Provided stride (" << provided_stride
                           << ") does not match expected outer stride for RowMajor (" << torch_sizes[1] << ")" << std::endl;
             }
            torch_strides.push_back(1); // Append stride for inner dimension
        } else if (cutlass_layout == cutlass::library::LayoutTypeID::kColumnMajor) {
            std::cout << "Info: Detected 2D ColumnMajor tensor with 1 stride provided. Assuming inner stride is 1." << std::endl;
            // For ColumnMajor [M, N], strides are [1, M]. We have M, need to add 1.
             if (provided_stride != torch_sizes[0]) {
                 std::cerr << "Warning: Provided stride (" << provided_stride
                           << ") does not match expected outer stride for ColumnMajor (" << torch_sizes[0] << ")" << std::endl;
             }
            torch_strides.insert(torch_strides.begin(), 1); // Prepend stride for inner dimension
        } else {
            // Unknown or non-standard layout, default to previous behavior with a warning
            throw std::runtime_error("Unknown or non-standard layout detected. Cannot determine missing stride.");
        }
    } else if (torch_sizes.size() != torch_strides.size() && !torch_sizes.empty()) {
        // Throw error if dimensions mismatch in other non-scalar cases
        throw std::runtime_error(
            "Mismatch between dimensionality of extent (" + std::to_string(torch_sizes.size()) +
            ") and stride (" + std::to_string(torch_strides.size()) + ") is not handled.");
    }

    // --- Step 3: Create LibTorch tensor using from_blob ---
    // Assumes the allocation is on the currently active CUDA device
    // torch::from_blob doesn't own the memory. Ensure device_alloc lifetime > torch_tensor lifetime.
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(torch_type)
                                       .device(torch::kCUDA) // Assumes CUDA device
                                       .requires_grad(false);

    torch::Tensor torch_tensor = torch::from_blob(
        data_ptr,
        torch_sizes,          // Use int64_t sizes
        torch_strides,
        options);

    // --- Step 4: Save the LibTorch tensor to a file ---
    try {
        torch::save(torch_tensor.clone(), output_filename);
        std::cout << "Tensor successfully saved to " << output_filename << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error saving tensor: " << e.msg() << std::endl;
        throw; // Re-throw exception
    }
}
}}
