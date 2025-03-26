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