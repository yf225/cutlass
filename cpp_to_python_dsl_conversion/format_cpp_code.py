import subprocess
import re
from pathlib import Path

def get_clang_format_version():
    """Get the installed clang-format version."""
    try:
        result = subprocess.run(['clang-format', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        # Extract version number using regex
        match = re.search(r'version (\d+\.\d+\.\d+)', result.stdout)
        if match:
            return match.group(1)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def create_clang_format_config(directory, style='google', version=None):
    """
    Create a .clang-format configuration file in the specified directory.
    Also adds a comment about the clang-format version used.
    """
    try:
        config_cmd = ['clang-format', '-style=' + style, '-dump-config']
        config = subprocess.run(config_cmd, capture_output=True, text=True, check=True)
        
        config_path = Path(directory) / '.clang-format'
        with open(config_path, 'w') as f:
            f.write(f"# Generated using clang-format version {get_clang_format_version()}\n")
            if version:
                f.write(f"# Required version: {version}\n")
            f.write(config.stdout)
        
        return True
    except subprocess.CalledProcessError:
        return False

def format_cpp_string(cpp_code):
    """
    Format a C++ code string using clang-format with version checking.
    
    Args:
        cpp_code (str): C++ code string to format
        required_version (str): Required clang-format version (e.g., "15.0.0")
    
    Returns:
        str: Formatted code if successful, None if failed
    """
    current_version = get_clang_format_version()
    if not current_version:
        raise RuntimeError("clang-format is not installed or version cannot be determined")
    
    # Check version in .clang-format file
    clang_format_path = Path('.clang-format')
    if clang_format_path.exists():
        with open(clang_format_path) as f:
            for line in f:
                if 'Required version:' in line:
                    required_version = line.split(':')[1].strip()
                    if current_version != required_version:
                        raise RuntimeError(
                            f"clang-format version mismatch. Required in .clang-format: {required_version}, "
                            f"Found: {current_version}"
                        )
                    break

    try:
        # Format the code string
        result = subprocess.run(
            ['clang-format'],
            input=cpp_code,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error formatting code: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Create a configuration file
    create_clang_format_config("./", style='google', version=get_clang_format_version())
    
    # Example C++ code string
    cpp_code = """
    #include <iostream>
    int main() {
        std::cout<<"Hello World!";
    return 0;
    }
    """
    
    # Format a code string with version checking
    try:
        formatted = format_cpp_string(cpp_code)
        if formatted:
            print("Formatted code:")
            print(formatted)
    except RuntimeError as e:
        print(f"Error: {e}")