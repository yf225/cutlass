# C++ to Python DSL conversion examples

This file provides examples mapping C++ code to our Python DSL. Each example consists of a C++ snippet and its equivalent DSL snippet.

Guidelines to follow:
- Use 2 spaces for indentation.
- Both `->` and `.` in C++ translate to `.` in Python DSL.
- DO NOT add Public to the Python DSL code for stuff in `struct`, unless they are specifically specified with `public` in the original C++ code.
  - In C++, struct members are public by default, while class members are private by default
  - When translating a C++ struct to Python DSL, do not add explicit Public declarations unless the C++ code has them
  - Example:
    ```cpp
    struct MyStruct {
      int x;  // implicitly public
      public:
        int y;  // explicitly public
    };
    ```
    Translates to:
    ```python
    class MyStruct:
      x: int  # no Public needed
      #[[public]]
      y: int  # explicit Public kept
    ```
- Template specialization using `<...>` in C++ translates to `.spec(...)` in Python DSL.
  For example:
  - C++: `std::vector<int>` -> Python DSL: `std.vector.spec(int)`
  - C++: `MyClass<T, U>` -> Python DSL: `MyClass.spec(T, U)`
  - C++: `uint64_t tma_barrier[size<2>(SmemLayoutA{})]` -> Python DSL: `tma_barrier: List[uint64_t, size.spec(2)(SmemLayoutA)]`
  - C++: `tensor<0>(tBsB)` -> Python DSL: `tensor.spec(0)(tBsB)`

---

## 1. Basic Translation Elements

### C++ Code
```cpp
/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#include <algorithm>
#include "custom/header.h"

#define MAX_SIZE 100
#define COMPUTE(x) ((x) * (x) + 1)

#ifdef DEBUG
#define LOG(x) std::cout << x << std::endl
#else
#define LOG(x)
#endif
```

### Python DSL Code
```python
"""
Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
"""
from algorithm import *
from custom.header import *

@pica.macro
def MAX_SIZE():
  return 100

@pica.macro
def COMPUTE(x):
  return (x) * (x) + 1

if defined(DEBUG()):
  @pica.macro
  def LOG(x):
    std.cout << x << std.endl
else:
  @pica.macro
  def LOG(x):
    pass
```

## 2. Classes and Templates

### C++ Code
```cpp
template <typename T, int N>
class Buffer {
public:
  T data[N];
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  
  CUTLASS_HOST_DEVICE
  T& operator[](int idx) {
    int x = 1;
    CUTE_UNROLL
    for (int i = 0; i < 10; ++i) {
      x = x * i;
    }
    return data[idx];
  }

private:
  static constexpr int size = N;
  T* ptr;
};

class NonTemplateClass {
public:
  static constexpr float PI = 3.14159f;
  
  CUTLASS_HOST_DEVICE
  float compute(float x) const {
    CUTE_UNROLL
    for (int i = 0; i < 10; ++i) {
      x = x * i;
    }
    return x * PI;
  }
};
```

### Python DSL Code
```python
@pica.cls(template=True)
class Buffer:
  def __new__(cls, T: typename, N: int):
    #[[public]]
    data: List[T, N]
    A: pica.type[alignas(128), ArrayEngine.spec(ElementA, cosize_v.spec(SmemLayoutA))]
    
    @pica.fn(CUTLASS_HOST_DEVICE())
    def __getitem__(self, idx: int) -> Ref[T]:
      x: int = 1
      CUTE_UNROLL()
      for i in range(10):
        x = x * i
      return data[idx]

    #[[private]]
    size: Static[ConstExpr[int]]] = N
    ptr: Ptr[T]

@pica.cls
class NonTemplateClass:
  #[[public]]
  PI: Static[ConstExpr[float]] = 3.14159f
  
  @pica.fn(CUTLASS_HOST_DEVICE(), Const)
  def compute(self, x: float) -> float:
    CUTE_UNROLL()
    for i in range(10):
      x = x * i
    return x * self.PI
```

## 3. CUDA Kernels and Device Code

### C++ Code
```cpp
template <typename T>
__global__ void vectorAdd(T* a, T* b, T* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__device__ float deviceFunc(float x) {
  return x * x;
}

// Launch
vectorAdd<<<grid, block>>>(d_a, d_b, d_c, n);
```

### Python DSL Code
```python
@pica.fn(__global__, template=True)
def vectorAdd(T: typename) -> None:
  a: Ptr[T]
  b: Ptr[T] 
  c: Ptr[T]
  n: int

  idx = blockIdx.x * blockDim.x + threadIdx.x
  if idx < n:
    c[idx] = a[idx] + b[idx]

@pica.fn(__device__)
def deviceFunc(x: float) -> float:
  return x * x

# Launch
vectorAdd[grid, block](d_a, d_b, d_c, n)
```

## 4. More CUDA Kernels and Device Code Examples

### C++ Code
```cpp
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
```

### Python Code
```python
@pica.fn(__global__, Static, template=True)
def gemm_device(ProblemShape: typename,
                TA: typename, TmaA: typename,
                TC: typename, TiledMma typename) -> None:
  gemm_device.add_attrs(__launch_bounds__(decltype(size(TiledMma())).value))

  shape_MNK: ProblemShape
  A: Const[Ptr[TA]]
  tma_a: CUTLASS_GRID_CONSTANT[Const[TmaA]]
  C: Ptr[TC]
  mma: TiledMma

  launch(shape_MNK, A, tma_a, C, mma)
```

## 5. Advanced Template Features

### C++ Code
```cpp
template<typename... Args>
struct tuple_wrapper {
  template<typename First, typename... Rest>
  CUTLASS_HOST_DEVICE
  static void process(First&& f, Rest&&... rest) {
    // Process variadic args
  }
};

template<typename T, typename Enable = void>
struct smart_pointer {
  static constexpr bool is_smart_ptr = false;
};

template<typename T>
struct smart_pointer<T, 
  std::enable_if_t<
    std::is_same_v<typename T::element_type*, decltype(std::declval<T>().get())>
  >
> {
  static constexpr bool is_smart_ptr = true;
};
```

### Python DSL Code
```python
@pica.struct(template=True)
class tuple_wrapper:
  def __new__(cls, *Args: typename):
    @pica.fn(Static, CUTLASS_HOST_DEVICE())
    def process(First: typename, *Rest: typename) -> None:
      f: Forward[First]
      rest: Forward[Rest]
      # Process variadic args

@pica.struct(template=True)
class smart_pointer:
  def __new__(cls, T: typename, Enable: typename = void):
    is_smart_ptr: Static[ConstExpr[bool]] = False

@pica.struct(template=True)
class smart_pointer:
  def __new__(cls, T: typename, Enable: typename = std.enable_if_t.spec(
      std.is_same_v.spec(typename(Ptr[T.element_type]), decltype(std.declval.spec(T)().get()))
    )):
    is_smart_ptr: Static[ConstExpr[bool]] = True
```

## 6. Lambda Functions and Captures

### C++ Code
```cpp
int multiplier = 10;
auto lambda1 = [multiplier](int x) { return x * multiplier; };

auto lambda2 = [&](int x) mutable { 
  multiplier *= 2;
  return x * multiplier; 
};

auto lambda3 = [=](int x) mutable { 
  multiplier *= 2;
  return x * multiplier; 
};

auto lambda4 = [=, &x](int y) {
  return x * y * multiplier;
}; 

auto generic_lambda = []<typename T>(T&& x, auto&& y) {
  return std::forward<T>(x) + y;
};
```

### Python DSL Code
```python
multiplier: int = 10

@pica.lambda(captures=[multiplier])
def lambda1(x: int) -> int:
  return x * multiplier

@pica.lambda(captures=[Ref[all]], mutable)
def lambda2(x: int) -> int:
  multiplier *= 2
  return x * multiplier

@pica.lambda(captures=[all], mutable) 
def lambda3(x: int) -> int:
  multiplier *= 2
  return x * multiplier

@pica.lambda(captures=[all, Ref[x]])
def lambda4(y: int) -> int:
  return x * y * multiplier

@pica.lambda(template=True)
def generic_lambda(T: typename)-> Any:
  x: Forward[T]
  y: Forward[auto]

  return std.forward.spec(T)(x) + y
```

## 7. Namespaces and Using Directives

### C++ Code
```cpp
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
```

### Python DSL Code
```python
#[[namespace]] outer {
#[[namespace]] inner {
  @pica.struct(template=True)
  class Helper:
    def __new__(cls, T: typename):
      value: Static[T]
#} [[end namespace]] inner
using(namespace(inner))
inner_helper = using(inner.Helper.spec(int))
#} [[end namespace]] outer
```

## 8. Member Access and Alignment Specifications

### C++ Code
```cpp
template<typename T>
struct alignas(16) AlignedStruct {
  T value;
  
  struct Inner {
    using type = T;
    T* ptr;
  };
  
  typename Inner::type* data;
};
```

### Python DSL Code
```python
@pica.struct(alignas(16), template=True)
class AlignedStruct:
  def __new__(cls, T: typename):
    value: T
    
    @pica.struct
    class Inner:
      type = using(T)
      ptr: Ptr[T]
    
    data: Ptr[Inner.type]
```

## 9. Friend Functions and Operators

### C++ Code
```cpp
template<typename T>
class Container {
  T value;
public:
  template<typename U>
  friend class OtherContainer;
  
  friend std::ostream& operator<<(std::ostream& os, const Container& c) {
    return os << c.value;
  }
  
  Container& operator+=(const Container& rhs) {
    value += rhs.value;
    return *this;
  }
};
```

### Python DSL Code
```python
@pica.cls(template=True)
class Container:
  def __new__(cls, T: typename):
    value: T
    
    @pica.cls(Friend, template=True)
    class OtherContainer:
      def __new__(cls, U: typename):
        ...
    
    @pica.fn(Friend)
    def __lshift__(os: Ref[std.ostream], c: Const[Ref[Container]]) -> Ref[std.ostream]:
      return os << c.value
    
    def __iadd__(self, rhs: Const[Ref[Container]]) -> Ref[Container]:
      self.value += rhs.value
      return deref(self)
```

## 10. Type Traits and Static Assertions

### C++ Code
```cpp
template<typename T>
struct is_valid {
  static_assert(sizeof(T) > 1, "Type too small");
  static_assert(std::is_default_constructible_v<T>, "Must be default constructible");
  
  template<typename U>
  static constexpr bool is_convertible = std::is_convertible_v<T, U>;
};
```

### Python DSL Code
```python
@pica.struct(template=True)
class is_valid:
  def __new__(cls, T: typename):
    static_assert(sizeof(T) > 1, "Type too small")
    static_assert(std.is_default_constructible_v.spec(T), "Must be default constructible")
    
    @pica.cls(template=True)
    class is_convertible:
      def __new__(cls, U: typename):
        return std.is_convertible_v.spec(T, U)
```

## 11. Type Alias and reinterpret_cast

### C++ Code
```cpp
extern __shared__ char shared_memory[];
using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
```

### Python Code
```python
shared_memory: extern[__shared__[List[char]]]
SharedStorage = using(SharedStorage.spec(TA, TB, SmemLayoutA, SmemLayoutB))
smem = deref(reinterpret_cast.spec(Ptr[SharedStorage])(shared_memory))
```

## 12. `&` usage (Address-of Operator)

### C++ Code
```cpp
ZType::wait(&mbar[pipe], write_state.phase());
```

### Python Code
```python
ZType.wait(addr_of(mbar[pipe]), write_state.phase())
```
