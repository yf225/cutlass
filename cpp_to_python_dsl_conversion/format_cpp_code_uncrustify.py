"""
1. `conda install conda-forge::uncrustify==0.80.1`
2. `python cpp_to_python_dsl_conversion/format_cpp_code_uncrustify.py`
"""

import subprocess
import re
from pathlib import Path

def get_uncrustify_version():
    """Get the installed uncrustify version."""
    try:
        result = subprocess.run(['uncrustify', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        # Extract version number using regex
        match = re.search(r'Uncrustify-(\d+\.\d+\.\d+)', result.stdout)
        if match:
            return match.group(1)
        return None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def ensure_uncrustify_config_exists(directory):
    """
    Ensure that the uncrustify.cfg file exists or copy it from the specified directory.
    """
    config_path = Path(directory) / 'uncrustify.cfg'
    if not config_path.exists():
        raise FileNotFoundError(f"No uncrustify.cfg found in {directory}. Please create one first.")
    return config_path.exists()

def format_cpp_string(cpp_code):
    """
    Format a C++ code string using uncrustify with the config file specified.
    
    Args:
        cpp_code (str): C++ code string to format
    
    Returns:
        str: Formatted code if successful, None if failed
    """
    current_version = get_uncrustify_version()
    if not current_version:
        raise RuntimeError("uncrustify is not installed or version cannot be determined")
    
    # Look for uncrustify.cfg in the current directory or parent directory
    config_file = Path('uncrustify.cfg')
    if not config_file.exists():
        # Try the script's directory
        script_dir = Path(__file__).parent
        config_file = script_dir / 'uncrustify.cfg'
        
    if not config_file.exists():
        raise FileNotFoundError(f"uncrustify.cfg file not found in current directory or {script_dir}")

    try:
        # Create a temporary file to hold the input code
        temp_input = Path('temp_input.cpp')
        temp_output = Path('temp_output.cpp')
        
        with open(temp_input, 'w') as f:
            f.write(cpp_code)
        
        # Format the code using uncrustify
        cmd = [
            'uncrustify', 
            '-c', str(config_file),
            '-f', str(temp_input),
            '-o', str(temp_output)
        ]
        
        subprocess.run(cmd, check=True)
        
        # Read the formatted output
        with open(temp_output, 'r') as f:
            formatted_code = f.read()
        
        # Clean up temporary files
        temp_input.unlink()
        temp_output.unlink()
        
        return formatted_code
    except subprocess.CalledProcessError as e:
        print(f"Error formatting code with uncrustify: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during formatting: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Example C++ code string
    cpp_code = """
#include <iostream>
namespace pica { namespace inner {
        int x = 1;
}}
int main() {
    std::cout<<"Hello World!";
return 0;
}

#include <iostream>
using namespace std;

class Outer {
public:
    int publicOuter;  // Public attribute of Outer

    // Constructor for Outer initializing both public and private attributes
    Outer() : publicOuter(1), privateOuter(2) {
        cout << "Outer constructed." << endl;
    }

    // Method to display Outer class attributes
    void displayOuter() {
        cout << "Outer -> publicOuter: " << publicOuter 
             << ", privateOuter: " << privateOuter << endl;
    }

    // Nested class inside Outer
    class Inner {
public:
        int publicInner;  // Public attribute of Inner

        // Constructor for Inner initializing both public and private attributes
        Inner() : publicInner(10), privateInner(20) {
            cout << "Inner constructed." << endl;
        }

        // Method to display Inner class attributes
        void displayInner() {
            cout << "Inner -> publicInner: " << publicInner 
                 << ", privateInner: " << privateInner << endl;
        }

        // Nested class inside Inner
        class Innermost {
public:
            int publicInnermost;  // Public attribute of Innermost

            // Constructor for Innermost initializing both public and private attributes
            Innermost() : publicInnermost(100), privateInnermost(200) {
                cout << "Innermost constructed." << endl;
            }

            // Method to display Innermost class attributes
            void displayInnermost() {
                cout << "Innermost -> publicInnermost: " << publicInnermost 
                     << ", privateInnermost: " << privateInnermost << endl;
            }

private:
            int privateInnermost;  // Private attribute of Innermost
        };

private:
        int privateInner;  // Private attribute of Inner
    };

private:
    int privateOuter;  // Private attribute of Outer
};

int main() {
    // Create an instance of Outer and display its attributes
    Outer outer;
    outer.displayOuter();

    // Create an instance of Inner (nested in Outer) and display its attributes
    Outer::Inner inner;
    inner.displayInner();

    // Create an instance of Innermost (nested in Inner) and display its attributes
    Outer::Inner::Innermost innermost;
    innermost.displayInnermost();

    return 0;
}

template<typename T>
struct is_valid {
  static_assert(sizeof(T) > 1, "Type too small");
  static_assert(std::is_default_constructible_v<T>, "Must be default constructible");
  
  template<typename U>
  static constexpr bool is_convertible = std::is_convertible_v<T, U>;
};

namespace outer {
  namespace inner {
    template <typename T>
    struct Helper {
      static T value;
    };
  }  // namespace inner
using namespace inner;
using inner_helper = inner::Helper<int>;
}  // namespace outer

template <class ProblemShape, class TA, class TmaA, class TC, class TiledMma>
__global__ static __launch_bounds__(decltype(size(
    TiledMma{}))::value) void gemm_device(ProblemShape shape_MNK, TA const* A,
                                          CUTLASS_GRID_CONSTANT TmaA const
                                              tma_a,
                                          TC* C, TiledMma mma) {
  launch(shape_MNK, A, tma_a, C, mma);
}
"""
    
    # Format a code string with uncrustify
    try:
        formatted = format_cpp_string(cpp_code)
        if formatted:
            print("Formatted code:")
            print(formatted)
    except Exception as e:
        print(f"Error: {e}")