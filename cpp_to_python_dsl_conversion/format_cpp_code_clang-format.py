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

def create_clang_format_config_if_not_exists(directory, style='google', version=None):
    """
    Create a .clang-format configuration file in the specified directory.
    Also adds a comment about the clang-format version used.
    """
    try:
        config_cmd = ['clang-format', '-style=' + style, '-dump-config']
        config = subprocess.run(config_cmd, capture_output=True, text=True, check=True)
        
        config_path = Path(directory) / '.clang-format'
        if not config_path.exists():
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
    create_clang_format_config_if_not_exists("./", style='google', version=get_clang_format_version())
    
    # Example C++ code string
    cpp_code = """
#include <iostream>
namespace     pica { namespace  inner {
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

template <class ProblemShape,
          class TA, class TmaA,
          class TC, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TC      * C, TiledMma mma)
{
  launch(shape_MNK, A, tma_a, C, mma);
}
"""
    
    # Format a code string with version checking
    try:
        formatted = format_cpp_string(cpp_code)
        formatted = format_cpp_string(formatted)
        if formatted:
            print("Formatted code:")
            print(formatted)
    except RuntimeError as e:
        print(f"Error: {e}")