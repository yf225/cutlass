import torch

"""
- aten.addmm                                     4496.327T     77.32%
                                                  1747.966T     30.06%  ['(21504,)(1,):torch.bfloat16', '(4352, 3072)(3072, 1):torch.bfloat16', '(3072, 21504)(1, 3072):torch.bfloat16']
                                                  1248.547T     21.47%  ['(3072,)(1,):torch.bfloat16', '(4352, 15360)(15360, 1):torch.bfloat16', '(15360, 3072)(1, 15360):torch.bfloat16']
                                                   470.041T      8.08%  ['(12288,)(1,):torch.bfloat16', '(4096, 3072)(3072, 1):torch.bfloat16', '(3072, 12288)(1, 3072):torch.bfloat16']
                                                   470.041T      8.08%  ['(3072,)(1,):torch.bfloat16', '(4096, 12288)(12288, 1):torch.bfloat16', '(12288, 3072)(1, 12288):torch.bfloat16']
                                                   352.531T      6.06%  ['(9216,)(1,):torch.bfloat16', '(4096, 3072)(3072, 1):torch.bfloat16', '(3072, 9216)(1, 3072):torch.bfloat16']
                                                   117.510T      2.02%  ['(3072,)(1,):torch.bfloat16', '(4096, 3072)(3072, 1):torch.bfloat16', '(3072, 3072)(1, 3072):torch.bfloat16']
"""

def create_tensor_with_strides(shape, strides, dtype, device):
    # Create a contiguous tensor first
    t = torch.randn(*shape, dtype=dtype, device=device)
    # Permute and reshape as needed to get the desired strides
    return t.as_strided(shape, strides)

def run_addmm_operations():
    # Set device and dtype
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Case 1: Largest computation (30.06%)
    bias1 = create_tensor_with_strides((21504,), (1,), dtype, device)
    mat1_1 = create_tensor_with_strides((4352, 3072), (3072, 1), dtype, device)
    mat2_1 = create_tensor_with_strides((3072, 21504), (1, 3072), dtype, device)
    result1 = torch.addmm(bias1, mat1_1, mat2_1)
    print(f"Shape 1 output: {result1.shape}, strides: {result1.stride()}")

    # Case 2: Second largest (21.47%)
    bias2 = create_tensor_with_strides((3072,), (1,), dtype, device)
    mat1_2 = create_tensor_with_strides((4352, 15360), (15360, 1), dtype, device)
    mat2_2 = create_tensor_with_strides((15360, 3072), (1, 15360), dtype, device)
    result2 = torch.addmm(bias2, mat1_2, mat2_2)
    print(f"Shape 2 output: {result2.shape}, strides: {result2.stride()}")

    # Case 3: (8.08%)
    bias3 = create_tensor_with_strides((12288,), (1,), dtype, device)
    mat1_3 = create_tensor_with_strides((4096, 3072), (3072, 1), dtype, device)
    mat2_3 = create_tensor_with_strides((3072, 12288), (1, 3072), dtype, device)
    result3 = torch.addmm(bias3, mat1_3, mat2_3)
    print(f"Shape 3 output: {result3.shape}, strides: {result3.stride()}")

    # Case 4: (8.08%)
    bias4 = create_tensor_with_strides((3072,), (1,), dtype, device)
    mat1_4 = create_tensor_with_strides((4096, 12288), (12288, 1), dtype, device)
    mat2_4 = create_tensor_with_strides((12288, 3072), (1, 12288), dtype, device)
    result4 = torch.addmm(bias4, mat1_4, mat2_4)
    print(f"Shape 4 output: {result4.shape}, strides: {result4.stride()}")

    # Case 5: (6.06%)
    bias5 = create_tensor_with_strides((9216,), (1,), dtype, device)
    mat1_5 = create_tensor_with_strides((4096, 3072), (3072, 1), dtype, device)
    mat2_5 = create_tensor_with_strides((3072, 9216), (1, 3072), dtype, device)
    result5 = torch.addmm(bias5, mat1_5, mat2_5)
    print(f"Shape 5 output: {result5.shape}, strides: {result5.stride()}")

if __name__ == "__main__":
    run_addmm_operations() 