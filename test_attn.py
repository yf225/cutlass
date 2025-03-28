import torch

"""
- aten._scaled_dot_product_flash_attention      1061.265T     18.25%
                                                 707.510T     12.17%  ['(1, 24, 4352, 128)(3072, 128, 3072, 1):torch.bfloat16', 
                                                                      '(1, 24, 4352, 128)(3072, 128, 3072, 1):torch.bfloat16', 
                                                                      '(1, 24, 4352, 128)(93585408, 128, 21504, 1):torch.bfloat16']
                                                 353.755T      6.08%  ['(1, 24, 4352, 128)(13369344, 557056, 128, 1):torch.bfloat16',
                                                                      '(1, 24, 4352, 128)(13369344, 557056, 128, 1):torch.bfloat16',
                                                                      '(1, 24, 4352, 128)(13369344, 557056, 128, 1):torch.bfloat16']
"""

def create_tensor_with_strides(shape, strides, dtype, device):
    # Calculate required storage size based on maximum offset
    max_offset = 0
    for dim, (size, stride) in enumerate(zip(shape, strides)):
        max_offset = max(max_offset, (size - 1) * stride)
    storage_size = max_offset + 1  # Add 1 to account for the last element
    
    # Create storage with sufficient size
    storage = torch.empty(storage_size, dtype=dtype, device=device)
    storage.normal_()  # Fill with random normal values
    
    # Create tensor with custom strides
    return torch.as_strided(storage, shape, strides)

def run_flash_attention_operations():
    # Set device and dtype
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # Case 1: Largest computation (12.17%)
    query1 = create_tensor_with_strides(
        (1, 24, 4352, 128), (3072, 128, 3072, 1), dtype, device
    )
    key1 = create_tensor_with_strides(
        (1, 24, 4352, 128), (3072, 128, 3072, 1), dtype, device
    )
    value1 = create_tensor_with_strides(
        (1, 24, 4352, 128), (93585408, 128, 21504, 1), dtype, device
    )
    
    # Using scaled_dot_product_attention as it's the public API for flash attention
    result1 = torch.nn.functional.scaled_dot_product_attention(
        query1, key1, value1,
        dropout_p=0.0,
        is_causal=True
    )
    print(f"Case 1 output shape: {result1.shape}, strides: {result1.stride()}")

    # Case 2: Second computation (6.08%)
    stride2 = (13369344, 557056, 128, 1)
    query2 = create_tensor_with_strides(
        (1, 24, 4352, 128), stride2, dtype, device
    )
    key2 = create_tensor_with_strides(
        (1, 24, 4352, 128), stride2, dtype, device
    )
    value2 = create_tensor_with_strides(
        (1, 24, 4352, 128), stride2, dtype, device
    )
    
    result2 = torch.nn.functional.scaled_dot_product_attention(
        query2, key2, value2,
        dropout_p=0.0,
        is_causal=True
    )
    print(f"Case 2 output shape: {result2.shape}, strides: {result2.stride()}")

if __name__ == "__main__":
    run_flash_attention_operations() 