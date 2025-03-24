"""
1. First, make sure you have llama.cpp installed and running locally on a different terminal:
```
export llamacpp_model_path=QwQ-32B-GGUF/QwQ-32B.Q8_0.gguf
export CUDA_VISIBLE_DEVICES=0  # customize this (can set to 1 / 2 / etc. to run on a different rank)
export gpu_id=${CUDA_VISIBLE_DEVICES:-0}
export base_port=12345
export port=$((base_port + gpu_id))
./build/bin/llama-server \
--host 0.0.0.0 \
--port ${port} \
--model models/${llamacpp_model_path} \
--n-gpu-layers 999 \
--temp 0.6 \
--ctx-size 61440 \
--batch-size 1 \
--ubatch-size 1 \
--threads 96 \
--mlock \
--no-mmap \
--main-gpu 0 \
--tensor-split 1
```

2. Then, run this script on a different terminal:
`python categorize_triton_kernel_llamacpp.py --catch-all --parallel-id <gpu_rank>`
"""

import os
import re
import glob
import argparse
import requests
import json
import sys
import hashlib
import tempfile
import shutil
import random

def sanitize_path(path):
    """Remove everything before and including "]/" from the path and strip query parameters."""
    # Remove everything before and including "]/"
    if "]/" in path:
        path = path.split("]/", 1)[1]
    
    # Remove query parameters if present
    if '?' in path:
        path = path.split('?')[0]
        
    return path

def extract_triton_kernels(file_path):
    """Extract @triton.jit decorated kernels from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Use a more robust regex-based approach to extract complete kernel functions
        pattern = r'(@triton\.jit[\s\S]*?def\s+\w+\s*\([^)]*\)[^:]*:[\s\S]*?)(?=\n(?:@|\s*def\s+|$))'
        matches = re.finditer(pattern, content + '\n')
        
        kernels = []
        for match in matches:
            kernel_code = match.group(1).strip()
            # Extract the function name
            func_match = re.search(r'def\s+(\w+)', kernel_code)
            if func_match:
                kernel_name = func_match.group(1)
                kernels.append((kernel_name, kernel_code))
        
        return kernels
    except UnicodeDecodeError:
        print(f"Skipping binary file: {file_path}")
        return []
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return []

def remove_think_tags(text):
    """Remove content between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def extract_content_after_think(text):
    """Extract only the content after the last </think> tag."""
    parts = text.split('</think>')
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()

def ask_llamacpp(prompt: str, server_url: str, model_params: dict = None, verbose: bool = False) -> str:
    # Default parameters for the model
    if model_params is None:
        model_params = {
            "stop": ["</s>"],  # Stop at end of output
            "stream": True
        }
    
    # Combine the model parameters with the prompt
    request_data = {
        "prompt": prompt,
        **model_params
    }
    
    response_text = ""
    token_buffer = ""
    
    try:
        response = requests.post(
            server_url,
            json=request_data,
            stream=True,
            timeout=None  # No timeout, since LLM generation can take a long time
        )
        
        for line in response.iter_lines():
            if not line:
                continue
                
            # Handle the SSE format used by llama.cpp server
            line_str = line.decode('utf-8')
            
            # Check if this is a data line
            if line_str.startswith('data: '):
                # Extract the JSON part
                json_str = line_str[6:]  # Remove 'data: ' prefix
                
                try:
                    line_json = json.loads(json_str)
                    
                    # Check if the response has content
                    if 'content' in line_json:
                        token = line_json['content']
                        response_text += token
                        
                        # Print actual tokens in real-time only if verbose
                        if verbose:
                            # Buffer until we get a complete word or punctuation
                            token_buffer += token
                            if token.endswith(' ') or token in ['.', ',', ':', ';', '?', '!', '\n']:
                                sys.stdout.write(token_buffer)
                                sys.stdout.flush()
                                token_buffer = ""
                        
                    # Check if we're done
                    if line_json.get('stop', False):
                        # Flush any remaining buffer if verbose
                        if token_buffer and verbose:
                            sys.stdout.write(token_buffer)
                            sys.stdout.flush()
                        break
                except json.JSONDecodeError:
                    print(f"Error parsing JSON: {json_str}")
        
        return response_text.strip()
    
    except Exception as e:
        print(f"llama.cpp API error: {str(e)}")
        return ""

def get_kernel_categories_from_llm(kernel_code, kernel_name, file_path, args):
    """
    Get categories from llama.cpp LLM.
    
    Uses llama.cpp API to determine the categories of the kernel based on its code and name.
    
    Possible categories include: matmul, attention, activation, element-wise op,
    grid sampling, interpolation, rendering, dtype conversion, quantize, de-quantize,
    rotary embedding, layer norm, cross entropy, jagged tensor, memory ops
    (concat, split, slice, index_select, layout conversion), padding, scatter add,
    block-sparse, scan, cumsum, dropout, etc.
    """
    prompt = f"""
<|user|>
Analyze this Triton kernel function and determine which categories it belongs to. 
Respond ONLY with the categories as a comma-separated list.
NOTE:
1. a kernel with attn / attention in the name doesn't always mean that it belongs to the attention category.
You should pay attention to what the kernel actually does attention operations.
2. a kernel that has "jagged" or "ragged" in the name belongs to the jagged tensor category; otherwise, it doesn't.
3. a kernel that has "mm" / "bmm" / "matmul" / "gemm" / "group_gemm" in the name belongs to the matmul category.
4. if a kernel belongs to "attention", it should also belong to "matmul", and potentially also "softmax".

Kernel name: {kernel_name}
File path: {file_path}

Kernel code:
```python
{kernel_code}
```

Available categories:
- matmul
- attention
- softmax
- activation
- pointwise
- reduction
- convolution
- grid sampling
- interpolation
- rendering
- dtype conversion
- quantize
- de-quantize
- rotary embedding
- layer norm
- cross entropy
- jagged tensor
- memory ops (concat, split, slice, index_select, transpose, layout conversion, etc.)
- padding
- scatter add
- block-sparse
- scan
- cumsum
- dropout
- moe (Mixture of Experts)
- atomic ops (atomic_add, atomic_max, atomic_min, atomic_xor, etc.)

Format your response as ONLY a comma-separated list of applicable categories. If none apply, respond with "unknown".
<|assistant|>
<think>
"""
    
    try:
        # Send request to llama.cpp server
        response_text = ask_llamacpp(
            prompt=prompt,
            server_url=args.server_url,
            model_params={
                "stop": ["</s>"],
                "stream": True,
                "temperature": args.temperature,
                "n_predict": args.max_tokens,
            },
            verbose=args.verbose
        )
        
        if not response_text:
            print(f"Empty response from llama.cpp for {kernel_name}", file=sys.stderr)
            return ["unknown"]
        
        # Extract only content after the last </think> tag
        categories_text = extract_content_after_think(response_text)
        
        # Remove any backticks or code block indicators
        categories_text = re.sub(r'```.*?```', '', categories_text, flags=re.DOTALL)
        categories_text = categories_text.strip('`')
        
        # Extract the last line that contains categories - this is typically the final answer
        # after any explanation the model might have provided
        lines = [line.strip() for line in categories_text.split('\n') if line.strip()]
        if lines:
            categories_text = lines[-1]  # Take the last non-empty line as the answer
        
        categories = [cat.strip() for cat in categories_text.split(',')]
        
        # Make sure we have valid categories
        if not categories or all(c == '' for c in categories):
            print(f"Warning: No valid categories returned for {kernel_name}", file=sys.stderr)
            return ["unknown"]
            
        return categories
    
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to llama.cpp server at {args.server_url}", file=sys.stderr)
        return ["unknown"]
    except requests.exceptions.Timeout:
        print(f"Error: Request to llama.cpp server timed out for {kernel_name}", file=sys.stderr)
        return ["unknown"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error from llama.cpp server for {kernel_name}: {e}", file=sys.stderr)
        return ["unknown"]
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with llama.cpp server for {kernel_name}: {e}", file=sys.stderr)
        return ["unknown"]
    except json.JSONDecodeError:
        print(f"Error parsing response from llama.cpp server for {kernel_name}", file=sys.stderr)
        return ["unknown"]

def get_kernel_filename(kernel_name, source_file_path):
    """
    Create a unique filename for a kernel by combining its name with a hash of its source file path.
    
    Args:
        kernel_name: The name of the kernel function
        source_file_path: The path where the kernel was found
    
    Returns:
        A unique filename in the format kernel_name_hash.py
    """
    # Create a hash of the source file path
    path_hash = hashlib.md5(source_file_path.encode('utf-8')).hexdigest()[:12]
    # Combine the kernel name with the hash
    return os.path.join("kernel_categorization", f"{kernel_name}_{path_hash}.py")

def find_all_unprocessed_kernels(all_paths, args):
    """
    Find all unprocessed Triton kernels from all paths.
    
    Returns:
        List of tuples (file_path, kernel_name, kernel_code) for unprocessed kernels.
    """
    unprocessed_kernels = []
    total_kernels = 0
    
    for path in all_paths:
        sanitized_path = sanitize_path(path)
        full_path = "/home/willfeng/local/fbsource/" + sanitized_path
        
        # Skip if path doesn't exist
        if not os.path.exists(full_path):
            print(f"Path not found, skipping: {full_path}")
            continue
            
        # Process directory or file
        if os.path.isdir(full_path):
            for file_path in glob.glob(os.path.join(full_path, "**"), recursive=True):
                if os.path.isfile(file_path):
                    kernels = extract_triton_kernels(file_path)
                    total_kernels += len(kernels)
                    
                    for kernel_name, kernel_code in kernels:
                        output_file = get_kernel_filename(kernel_name, sanitize_path(file_path))
                        if not os.path.exists(output_file):
                            unprocessed_kernels.append((file_path, kernel_name, kernel_code))
        elif os.path.isfile(full_path):
            kernels = extract_triton_kernels(full_path)
            total_kernels += len(kernels)
            
            for kernel_name, kernel_code in kernels:
                output_file = get_kernel_filename(kernel_name, sanitize_path(full_path))
                if not os.path.exists(output_file):
                    unprocessed_kernels.append((full_path, kernel_name, kernel_code))
    
    print(f"Found {len(unprocessed_kernels)} unprocessed kernels out of {total_kernels} total kernels")
    return unprocessed_kernels

def process_kernel(file_path, kernel_name, kernel_code, args):
    """Process a single kernel and save it to a file."""
    sanitized_path = sanitize_path(file_path)
    output_file = get_kernel_filename(kernel_name, sanitized_path)
    
    # Get kernel categories from LLM
    categories = get_kernel_categories_from_llm(kernel_code, kernel_name, sanitized_path, args)
    
    # Create a temporary file
    fd, temp_file = tempfile.mkstemp(suffix='.py', prefix='kernel_tmp_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(f"# {sanitized_path}\n")
            f.write(f"# {', '.join(categories)}\n")
            f.write(kernel_code)
        
        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Move the temporary file to the final destination
        shutil.move(temp_file, output_file)
        
        print(f"Created kernel file: {output_file}")
        return True
    except Exception as e:
        print(f"Error writing kernel file {output_file}: {e}")
        # Clean up the temporary file if it still exists
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        return False

def process_file(file_path, args):
    """Process a single file to extract and save Triton kernels."""
    sanitized_path = sanitize_path(file_path)
    kernels = extract_triton_kernels(file_path)
    if len(kernels) > 0:
        print(f"Found {len(kernels)} kernels in {sanitized_path}")
    for i, (kernel_name, kernel_code) in enumerate(kernels):
        # Create output filename using the hash-based method
        output_file = get_kernel_filename(kernel_name, sanitized_path)
        
        # Check if the output file already exists
        if os.path.exists(output_file):
            print(f"[{i+1}/{len(kernels)}] Skipping existing kernel file: {output_file}")
            continue
        
        # Process the kernel
        success = process_kernel(file_path, kernel_name, kernel_code, args)
        if success:
            print(f"[{i+1}/{len(kernels)}] Created kernel file: {output_file}")

def process_path(path, args):
    """Process a path which can be a file or directory."""
    sanitized_path = sanitize_path(path)
    sanitized_path = "/home/willfeng/local/fbsource/" + sanitized_path
    
    if os.path.isdir(sanitized_path):
        process_directory(sanitized_path, args)
    elif os.path.isfile(sanitized_path):
        process_file(sanitized_path, args)
    else:
        print(f"Path not found: {sanitized_path}")

def process_directory(dir_path, args):
    """Process all files in a directory to extract and save Triton kernels."""
    sanitized_path = sanitize_path(dir_path)
    print(f"Processing directory: {sanitized_path}")
    
    # Recursively get all files in the directory
    for file_path in glob.glob(os.path.join(sanitized_path, "**"), recursive=True):
        if os.path.isfile(file_path):
            process_file(file_path, args)

def catch_all_mode(args):
    """
    Process Triton kernels in catch-all mode.
    
    Randomly selects unprocessed kernels from all paths in kernel_paths.txt and processes them
    until all kernels have been processed.
    """
    # Read paths from kernel_paths.txt file
    paths_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_paths.txt")
    
    try:
        with open(paths_file, 'r') as f:
            # Filter out empty lines
            all_paths = [line.strip() for line in f if line.strip()]
        
        if not all_paths:
            print(f"Warning: No paths found in {paths_file}")
            return
            
        print(f"Catch-all mode: searching for unprocessed Triton kernels in {len(all_paths)} paths")
        
        # Find all unprocessed kernels
        unprocessed_kernels = find_all_unprocessed_kernels(all_paths, args)
        
        if not unprocessed_kernels:
            print("No unprocessed kernels found. All done!")
            return
            
        # Process kernels in random order until all are processed
        while unprocessed_kernels:
            # Select a random kernel
            kernel_idx = random.randrange(len(unprocessed_kernels))
            file_path, kernel_name, kernel_code = unprocessed_kernels.pop(kernel_idx)
            
            # Check if the output file already exists (someone else might have processed it)
            output_file = get_kernel_filename(kernel_name, sanitize_path(file_path))
            if os.path.exists(output_file):
                print(f"Skipping already processed kernel: {kernel_name} from {sanitize_path(file_path)}")
                continue
            
            print(f"Processing kernel {kernel_name} from {sanitize_path(file_path)}")
            print(f"Remaining unprocessed kernels: {len(unprocessed_kernels)}")
            
            # Process the kernel
            process_kernel(file_path, kernel_name, kernel_code, args)
        
        print("All kernels have been processed!")
        
    except FileNotFoundError:
        print(f"Error: File {paths_file} not found. Please create this file with one path per line.")
        sys.exit(1)
    except Exception as e:
        print(f"Error in catch-all mode: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract Triton kernels from source files.")
    
    parser.add_argument("--parallel-id", type=int, required=False, default=None, choices=range(8),
                        help="Parallel ID (0-7) to 1. determine the llama.cpp server port 2. determine which subset of paths to process in kernel_paths.txt if --path is not provided")
    
    # Add path argument
    parser.add_argument("--path", type=str, default=None,
                        help="Specific file or directory to process")
    
    # Add catch-all mode flag
    parser.add_argument("--catch-all", action="store_true",
                        help="Enable catch-all mode that randomly processes unprocessed kernels until all are done")
    
    # llama.cpp API arguments
    parser.add_argument("--server-url", default=None, 
                       help="The llama.cpp server URL (if not specified, uses http://localhost:<12345+parallel_id>/completion)")
    parser.add_argument("--temperature", type=float, default=0.6, 
                       help="Sampling temperature; lower is more focused (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=999999,
                       help="Maximum number of tokens to generate (default: %(default)s)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print LLM output to console (default: quiet)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("kernel_categorization", exist_ok=True)
    
    # Handle server URL based on parallel_id if not provided
    if args.server_url is None:
        if args.parallel_id is not None:
            port = 12345 + args.parallel_id
        else:
            port = 12345  # Default to first port if no part_id specified
        args.server_url = f"http://localhost:{port}/completion"
    
    print(f"Using server URL: {args.server_url}")
    
    # If catch-all mode is enabled, process in catch-all mode
    if args.catch_all:
        catch_all_mode(args)
        return
    
    # If a specific path is provided, process only that path
    if args.path:
        print(f"Processing single path: {args.path}")
        process_path(args.path, args)
        return
    
    # Otherwise, use the parallel_id logic to process paths from the file
    if args.parallel_id is None:
        print("Error: Either --path, --parallel-id, or --catch-all must be specified")
        sys.exit(1)
    
    # Read paths from kernel_paths.txt file
    paths_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_paths.txt")
    
    try:
        with open(paths_file, 'r') as f:
            # Filter out empty lines
            all_paths = [line.strip() for line in f if line.strip()]
        
        if not all_paths:
            print(f"Warning: No paths found in {paths_file}")
            return
        
        # Split paths into 8 parts
        total_paths = len(all_paths)
        paths_per_part = (total_paths + 7) // 8  # Ceiling division to ensure all paths are covered
        
        # Calculate start and end indices for this part
        start_idx = args.parallel_id * paths_per_part
        end_idx = min(start_idx + paths_per_part, total_paths)
        
        # Get the paths for this part
        part_paths = all_paths[start_idx:end_idx]
        
        print(f"Processing part {args.parallel_id} with {len(part_paths)} paths (out of {total_paths} total)")
        
        for path in part_paths:
            process_path(path, args)
            
    except FileNotFoundError:
        print(f"Error: File {paths_file} not found. Please create this file with one path per line.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {paths_file}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()