"""
This script uses llama.cpp server to analyze C++ code blocks.
Step 1: install required packages: `pip install requests`
Step 2: Start the llama.cpp server (default: http://localhost:12345)
Step 3: Run `python cpp_to_python_dsl_conversion/find_cpp_codeblocks_llm.py <cpp_file>`
"""

import os
import requests
import json
import sys
import re

def remove_think_tags(text):
    """Remove content between <think> and </think> tags."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def extract_content_after_think(text):
    """Extract only the content after the last </think> tag."""
    parts = text.split('</think>')
    if len(parts) > 1:
        return parts[-1].strip()
    return text.strip()

def ask_llamacpp(prompt: str, server_url: str = "http://localhost:12345/completion", model_params: dict = None, verbose: bool = False) -> str:
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

def identify_next_block(code: str) -> tuple:
    """
    Use LLM to identify the next logical code block in the given code.
    Returns a tuple of (block_type, block_content, remaining_code).
    """
    prompt = f"""
<|user|>
Analyze this C++ code and identify the first complete logical block (class, struct, function, etc.).
Return ONLY a JSON object with these fields:
- type: The type of the block (class, struct, function, namespace, include, using, comment, license_header, unknown)
- content: The exact code content of the block
- end_pos: The character position where this block ends (to help extract the remaining code)

Here's the code to analyze:
```cpp
{code}
```

Format your response as a valid JSON object with the specified fields. If you can't identify a block, set type to "unknown" and take the first line.
<|assistant|>
<think>
"""
    
    try:
        # Send request to llama.cpp server
        response_text = ask_llamacpp(
            prompt=prompt,
            model_params={
                "stop": ["</s>"],
                "stream": True,
                "temperature": 0.1,  # Low temperature for more focused responses
                "n_predict": 1000,  # Need more tokens for JSON response
            },
            verbose=False
        )
        
        if not response_text:
            print(f"Empty response from llama.cpp", file=sys.stderr)
            # Default to first line if can't identify block
            first_line_end = code.find('\n')
            if first_line_end == -1:
                return "unknown", code.strip(), ""
            return "unknown", code[:first_line_end].strip(), code[first_line_end:].strip()
        
        # Extract only content after the last </think> tag
        json_str = extract_content_after_think(response_text)
        
        try:
            result = json.loads(json_str)
            block_type = result.get('type', 'unknown').lower().strip()
            block_content = result.get('content', '').strip()
            end_pos = result.get('end_pos', 0)
            
            # Validate block type
            valid_types = {"class", "struct", "function", "namespace", "include", "using", "comment", "license_header", "unknown"}
            if block_type not in valid_types:
                block_type = "unknown"
            
            # If we got valid content, return it with the remaining code
            if block_content:
                # Find the actual end position of the block in the original code
                if block_content in code:
                    actual_end = code.find(block_content) + len(block_content)
                    remaining_code = code[actual_end:].strip()
                else:
                    # If exact match fails, use the suggested end_pos
                    remaining_code = code[end_pos:].strip()
                return block_type, block_content, remaining_code
            
        except json.JSONDecodeError:
            print(f"Error parsing LLM response as JSON: {json_str}", file=sys.stderr)
    
    except Exception as e:
        print(f"Error identifying block: {e}", file=sys.stderr)
    
    # Default to first line if all else fails
    first_line_end = code.find('\n')
    if first_line_end == -1:
        return "unknown", code.strip(), ""
    return "unknown", code[:first_line_end].strip(), code[first_line_end:].strip()

def analyze_cpp_file(file_path: str) -> list:
    """
    Analyze a C++ file and return a list of code blocks with their types.
    Each block is a tuple of (type, content).
    """
    with open(file_path, 'r') as f:
        source_code = f.read()
    
    blocks = []
    remaining_code = source_code.strip()
    
    # Keep processing until we've consumed all the code
    while remaining_code:
        block_type, block_content, remaining_code = identify_next_block(remaining_code)
        if block_content:  # Only add non-empty blocks
            blocks.append((block_type, block_content))
        else:
            break  # Prevent infinite loop if we can't identify any more blocks
    
    return blocks

def main():
    if len(sys.argv) != 2:
        print("Usage: python find_cpp_codeblocks_llm.py <cpp_file>")
        sys.exit(1)
    
    cpp_file = sys.argv[1]
    if not os.path.exists(cpp_file):
        print(f"Error: File {cpp_file} not found")
        sys.exit(1)
    
    print(f"Analyzing {cpp_file}...")
    blocks = analyze_cpp_file(cpp_file)
    
    print("\nFound code blocks:")
    for block_type, content in blocks:
        print(f"\n{'-'*40}")
        print(f"Type: {block_type}")
        print(f"{'-'*40}")
        print(content)
    
    # Statistics
    type_counts = {}
    for block_type, _ in blocks:
        type_counts[block_type] = type_counts.get(block_type, 0) + 1
    
    print("\nBlock type statistics:")
    for block_type, count in sorted(type_counts.items()):
        print(f"{block_type}: {count}")

if __name__ == "__main__":
    main()