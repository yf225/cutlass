"""
This script demonstrates how to use tree-sitter to parse C++ code.
Step 1: install required packages: `pip install tree-sitter`
Step 2: Run `git clone https://github.com/tree-sitter/tree-sitter-cpp.git`
Step 3: Run `python find_cpp_functions_tree_sitter.py <cpp_file>`
"""

from tree_sitter import Language, Parser
import os

def setup_tree_sitter():
    """Initialize tree-sitter and build the C++ language."""
    assert os.path.exists("tree-sitter-cpp")
    
    # Build the language
    Language.build_library(
        # Store the library in the `build` directory
        "build/my-languages.so",
        # Include the cpp language
        ["tree-sitter-cpp"]
    )
    
    CPP_LANGUAGE = Language("build/my-languages.so", "cpp")
    parser = Parser()
    parser.set_language(CPP_LANGUAGE)
    
    return parser

def parse_cpp_file(parser, file_path):
    """Parse a C++ file and return the syntax tree."""
    with open(file_path, 'rb') as file:
        source_code = file.read()
    
    tree = parser.parse(source_code)
    return tree

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

def main():
    # Initialize tree-sitter
    parser = setup_tree_sitter()
    
    # Example usage with a sample C++ file
    cpp_code = b"""
    #include <iostream>
    
    int add(int a, int b) {
        return a + b;
    }
    
    class Calculator {
    public:
        double multiply(double x, double y) {
            return x * y;
        }
    };
    
    int main() {
        Calculator calc;
        std::cout << "Hello, World!" << std::endl;
        return 0;
    }
    """
    
    # Write example code to a file
    with open("example.cpp", "wb") as f:
        f.write(cpp_code)
    
    # Parse the file
    tree = parse_cpp_file(parser, "example.cpp")
    
    print("Full syntax tree:")
    print_tree(tree.root_node, cpp_code)
    
    print("\nFunctions found:")
    functions = find_functions_with_code(tree.root_node, cpp_code)
    for func_name, func_code in functions:
        print(f"\nFunction: {func_name}")
        print("Code:")
        print(func_code)
        print("-" * 50)

if __name__ == "__main__":
    main()