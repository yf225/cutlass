
    /* This is a block comment */
    // This is a line comment
    #include <iostream>
    #include "myheader.h"
    
    using namespace std;
    using MyType = int;
    
    struct Point {
        int x;
        int y;
    };
    
    class Calculator {
    public:
        double multiply(double x, double y) {
            return x * y;
        }
    };
    
    int add(int a, int b) {
        return a + b;
    }
    
    int main() {
        Calculator calc;
        std::cout << "Hello, World!" << std::endl;
        return 0;
    }
    