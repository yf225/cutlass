"""
This script demonstrates how to use tree-sitter to parse C++ code.
Step 1: install required packages: `pip install tree-sitter`
Step 2: Run `pushd cpp_to_python_dsl_conversion && git clone https://github.com/tree-sitter/tree-sitter-cpp.git && popd`
Step 3: Run `python cpp_to_python_dsl_conversion/find_cpp_codeblocks_tree_sitter.py <cpp_file>`
"""

import os
import tree_sitter
from itertools import zip_longest

def setup_tree_sitter():
    """Set up tree-sitter with the C++ language."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the C++ language
    cpp_language_dir = os.path.join(script_dir, "tree-sitter-cpp")
    cpp_library_path = os.path.join(script_dir, "build", "cpp.so")
    
    # Only build if not already built
    if not os.path.exists(os.path.dirname(cpp_library_path)):
        os.makedirs(os.path.dirname(cpp_library_path))
        
    if not os.path.exists(cpp_library_path):
        tree_sitter.Language.build_library(
            # Store the library in the build directory
            cpp_library_path,
            # Include one or more languages
            [cpp_language_dir]
        )
    
    # Load the C++ language
    CPP_LANGUAGE = tree_sitter.Language(cpp_library_path, 'cpp')
    return CPP_LANGUAGE

# Initialize the C++ language
CPP_LANGUAGE = setup_tree_sitter()

def parse_cpp_file(source_code):
    """Parse a C++ file using tree-sitter and return the syntax tree."""
    parser = tree_sitter.Parser()
    parser.set_language(CPP_LANGUAGE)
    
    # Convert source code to bytes if it's a string
    if isinstance(source_code, str):
        source_code = source_code.encode('utf8')
    
    tree = parser.parse(source_code)
    return tree

def format_cpp_code(source_code):
    """Format C++ code by removing extra whitespace and normalizing newlines."""
    if isinstance(source_code, bytes):
        source_code = source_code.decode('utf8')
    
    # Split into lines and strip whitespace
    lines = source_code.splitlines()
    formatted_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped:
            formatted_lines.append(stripped)
    
    return "\n".join(formatted_lines)

def print_tree(node, source_code, level=0):
    """Print the syntax tree in a readable format."""
    indent = "  " * level
    
    # Print the current node
    if len(node.children) == 0:
        # For leaf nodes, print the actual text
        start_byte = node.start_byte
        end_byte = node.end_byte
        text = source_code[start_byte:end_byte].decode('utf8')
        print(f"{indent}{node.type}: '{text}'")
    else:
        # For non-leaf nodes, just print the type
        print(f"{indent}{node.type}")
    
    # Recursively print children
    for child in node.children:
        print_tree(child, source_code, level + 1)

def get_function_code(node, source_code):
    """Extract the full function code including the definition."""
    if node.type == "function_definition":
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code[start_byte:end_byte].decode('utf8')
    return None

def find_functions_with_code(node, source_code):
    """Find all function declarations in the code and return their names and full code."""
    functions = []
    
    if node.type == "function_definition":
        # Get function name
        declarator = next(
            (child for child in node.children if child.type == "function_declarator"),
            None
        )
        if declarator:
            name = next(
                (child for child in declarator.children if child.type == "identifier"),
                None
            )
            if name:
                start_byte = name.start_byte
                end_byte = name.end_byte
                func_name = source_code[start_byte:end_byte].decode('utf8')
                func_code = get_function_code(node, source_code)
                functions.append((func_name, func_code))
    
    # Recursively search children
    for child in node.children:
        functions.extend(find_functions_with_code(child, source_code))
    
    return functions

def get_comment_block(node, source_code):
    """Extract comment blocks from the code."""
    if node.type == "comment":
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code[start_byte:end_byte].decode('utf8')
    return None

def get_include_statement(node, source_code):
    """Extract #include statements from the code."""
    if node.type == "preproc_include":
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code[start_byte:end_byte].decode('utf8')
    return None

def get_using_statement(node, source_code):
    """Extract using statements from the code."""
    if node.type in ["using_declaration", "alias_declaration"] or (node.type == "declaration" and "using" in source_code[node.start_byte:node.end_byte].decode('utf8')):
        start_byte = node.start_byte
        end_byte = node.end_byte
        text = source_code[start_byte:end_byte].decode('utf8')
        if not text.strip().endswith(';'):
            text = text.strip() + ';'
        return text
    return None

def get_struct_definition(node, source_code):
    """Extract struct definitions from the code."""
    if node.type == "struct_specifier":
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code[start_byte:end_byte].decode('utf8')
    return None

def get_class_definition(node, source_code):
    """Extract class definitions from the code."""
    if node.type == "class_specifier":
        start_byte = node.start_byte
        end_byte = node.end_byte
        return source_code[start_byte:end_byte].decode('utf8')
    return None

def get_using_alias(node, source_code):
    """Extract using alias declarations from the code."""
    # Check for type alias declarations
    if node.type == "declaration":
        start_byte = node.start_byte
        end_byte = node.end_byte
        text = source_code[start_byte:end_byte].decode('utf8')
        if text.strip().startswith('using') and '=' in text:
            return text
    return None

def find_all_code_blocks(node, source_code):
    blocks = []  # List of (type, content, start_byte) tuples
    
    def is_license_header(text):
        return "Copyright" in text and "License" in text
        
    def is_doc_comment(text):
        return text.startswith("/*") and not is_license_header(text)
        
    def get_node_text(node):
        if isinstance(source_code, str):
            return source_code[node.start_byte:node.end_byte]
        return source_code[node.start_byte:node.end_byte].decode('utf8')
        
    def get_full_node_text(node, include_template=True):
        """Get the full text of a node including any template declarations."""
        if not include_template:
            return get_node_text(node)
            
        # Find any template declaration that precedes this node
        prev_sibling = node.prev_sibling
        template_start = node.start_byte
        template_node = None
        
        while prev_sibling:
            if prev_sibling.type == "template_declaration":
                template_start = prev_sibling.start_byte
                template_node = prev_sibling
                break
            elif prev_sibling.type not in ["comment", "preproc_include"]:
                break
            prev_sibling = prev_sibling.prev_sibling
            
        # Get the text from the template declaration (if any) through the end of this node
        if template_node:
            if isinstance(source_code, str):
                return source_code[template_start:node.end_byte]
            return source_code[template_start:node.end_byte].decode('utf8')
        return get_node_text(node)
        
    def process_node(node, inside_class=False):
        node_type = node.type
        
        if node_type == "comment":
            node_text = get_node_text(node)
            if is_license_header(node_text):
                blocks.append(("license", node_text, node.start_byte))
            elif is_doc_comment(node_text):
                blocks.append(("comments", node_text, node.start_byte))
                
        elif node_type == "preproc_include":
            blocks.append(("includes", get_node_text(node), node.start_byte))
            
        elif node_type == "using_declaration":
            blocks.append(("using_statements", get_node_text(node), node.start_byte))
            
        elif node_type == "template_declaration":
            # Skip template declarations as they will be handled with their associated nodes
            pass
            
        elif node_type == "struct_specifier":
            # Get the full struct text including any template declarations
            full_text = get_full_node_text(node)
            blocks.append(("structs", full_text + ";", node.start_byte))
            
        elif node_type == "class_specifier":
            # Get the full class text including any template declarations
            full_text = get_full_node_text(node)
            blocks.append(("classes", full_text + ";", node.start_byte))
            
        elif node_type == "function_definition" and not inside_class:
            # Get the full function text including any template declarations
            full_text = get_full_node_text(node)
            blocks.append(("functions", full_text, node.start_byte))
            
        # Recursively process children
        for child in node.children:
            process_node(child, inside_class or node_type in ["class_specifier", "struct_specifier"])
            
    process_node(node)
    
    # Sort blocks by their original position
    blocks.sort(key=lambda x: x[2])
    
    # Merge adjacent blocks of the same type
    merged_blocks = []
    current_block = None
    
    for block in blocks:
        block_type, content, start_byte = block
        
        if current_block is None:
            current_block = block
        elif current_block[0] == block_type:
            # Merge blocks of the same type
            current_block = (block_type, current_block[1] + "\n" + content, current_block[2])
        else:
            merged_blocks.append(current_block)
            current_block = block
            
    if current_block is not None:
        merged_blocks.append(current_block)
        
    return merged_blocks

def main():
    # Initialize tree-sitter
    CPP_LANGUAGE = setup_tree_sitter()
    
    # Parse the C++ file
    cpp_file = "examples/cute/tutorial/hopper/wgmma_tma_sm90_simple.cu"
    with open(cpp_file, 'r') as f:
        source_code = f.read()
    
    # Format the original code
    formatted_code = format_cpp_code(source_code)
    
    # Parse and find all code blocks
    tree = parse_cpp_file(source_code)
    blocks = find_all_code_blocks(tree.root_node, source_code)
    
    # Concatenate blocks in their original order
    concatenated_code = ""
    prev_block_type = None
    
    # Define block type order and spacing rules
    block_spacing = {
        ("license", "includes"): "\n",
        ("includes", "includes"): "\n",
        ("includes", "using_statements"): "\n\n",
        ("using_statements", "using_statements"): "\n",
        ("using_statements", "structs"): "\n\n",
        ("structs", "structs"): "\n\n",
        ("structs", "classes"): "\n\n",
        ("classes", "classes"): "\n\n",
        ("classes", "functions"): "\n\n",
        ("functions", "functions"): "\n\n",
    }
    
    for block_type, content, _ in blocks:
        # Add appropriate spacing between blocks
        if prev_block_type:
            spacing = block_spacing.get((prev_block_type, block_type), "\n")
            concatenated_code += spacing
        
        concatenated_code += content
        prev_block_type = block_type
    
    # Compare the concatenated code with the original
    concatenated_lines = [line.strip() for line in concatenated_code.splitlines() if line.strip()]
    original_lines = [line.strip() for line in formatted_code.splitlines() if line.strip()]
    
    if concatenated_lines == original_lines:
        print("\nVerification successful: Concatenated code blocks match the formatted original code")
    else:
        print("\nVerification failed: Concatenated code blocks do not match the formatted original code")
        print("\nDifferences between concatenated and original code:\n")
        for i, (concat_line, orig_line) in enumerate(zip_longest(concatenated_lines, original_lines), 1):
            if concat_line != orig_line:
                print(f"Line {i}:")
                print(f"Concatenated: {concat_line}")
                print(f"Original:    {orig_line}\n")
                
if __name__ == "__main__":
    main()