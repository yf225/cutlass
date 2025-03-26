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
#[[public]]  # explicit Public kept
  y: int
```
- Template specialization using `<...>` in C++ translates to `.spec(...)` in Python DSL.
  For example:
  - C++: `std::vector<int>` -> Python DSL: `std.vector.spec(int)`
  - C++: `MyClass<T, U>` -> Python DSL: `MyClass.spec(T, U)`
  - C++: `uint64_t tma_barrier[size<2>(SmemLayoutA{})]` -> Python DSL: `tma_barrier: List[uint64_t, size.spec(2)(SmemLayoutA)]`
  - C++: `tensor<0>(tBsB)` -> Python DSL: `tensor.spec(0)(tBsB)`
- C++: `public:` -> Python DSL: `#[[public]]`, and C++: `private:` -> Python DSL: `#[[private]]`.
  And regardless of how much indentation there is for C++ `public:` and `private:`, their corresponding Python DSL must not have any leading spaces.

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

#define MAX_SIZE 100
#define COMPUTE(x) ((x) * (x) + 1)

#ifdef DEBUG
#define LOG(x) std.cout << x << std.endl
#else
#define LOG(x)
#endif
```

## 2. Classes and Templates (template class)

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
```

### Python DSL Code
```python
@pica.template({T: typename, N: int})
class Buffer:
#[[public]]
  data: List[T].len(N)
  A: pica.type[alignas(128), cute::ArrayEngine.spec(ElementA, cosize_v.spec(SmemLayoutA))]
  
  CUTLASS_HOST_DEVICE
  def __getitem__(self, idx: int) -> ref[T]:
    x: int = 1
    CUTE_UNROLL
    for i in range(10):
      x = x * i
    return data[idx]

#[[private]]
  size: static[constexpr[int]] = N
  ptr: ptr[T]
```

## 2.1 Classes and Templates (non-template class)

### C++ Code
```cpp
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
class NonTemplateClass:
#[[public]]
  PI: static[constexpr[float]] = 3.14159

  CUTLASS_HOST_DEVICE
  @pica.attrs(const)
  def compute(self, x: float) -> float:
    CUTE_UNROLL
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
@pica.template({T: typename})
@pica.attrs(__global__)
def vectorAdd(a: ptr[T], b: ptr[T], c: ptr[T], n: int) -> None:
  idx: int = blockIdx.x * blockDim.x + threadIdx.x
  if idx < n:
    c[idx] = a[idx] + b[idx]

@pica.attrs(__device__)
def deviceFunc(x: float) -> float:
  return x * x

# Launch
vectorAdd[grid, block](d_a, d_b, d_c, n)
```

## 4. More CUDA Kernels and Device Code Examples

### C++ Code
```cpp
template <class ProblemShape, class TA, class TmaA, class TC, class TiledMma>
__global__ static __launch_bounds__(decltype(size(
    TiledMma{}))::value) void gemm_device(ProblemShape shape_MNK, TA const* A,
                                          CUTLASS_GRID_CONSTANT TmaA const
                                              tma_a,
                                          TC* C, TiledMma mma) {
  launch(shape_MNK, A, tma_a, C, mma);
}
```

### Python Code
```python
@pica.template({ProblemShape: class_,
                TA: class_, TmaA: class_,
                TC: class_, TiledMma: class_})
@pica.attrs(__global__, static)
@pica.attrs(__launch_bounds__(decltype(size(TiledMma())).value))
def gemm_device(shape_MNK: ProblemShape,
                A: ptr[Const[TA]], tma_a: CUTLASS_GRID_CONSTANT[Const[TmaA]],
                C: ptr[TC], mma: TiledMma) -> None:
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
def lambda1(x: int) -> Any:
  return x * multiplier

@pica.lambda(captures=[ref[all]], mutable)
def lambda2(x: int) -> Any:
  multiplier *= 2
  return x * multiplier

@pica.lambda(captures=[all], mutable) 
def lambda3(x: int) -> Any:
  multiplier *= 2
  return x * multiplier

@pica.lambda(captures=[all, ref[x]])
def lambda4(y: int) -> Any:
  return x * y * multiplier

@pica.lambda(template={T: typename})
def generic_lambda(x: Forward[T], y: Forward[Any])-> Any:
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

namespace outer {
namespace inner {
struct Helper2 {
  static T value2;
};
}  // namespace inner
}  // namespace outer
```

### Python DSL Code
```python
#[[namespace outer]] 
#[[namespace inner]] 
@pica.template(T: typename)
class Helper(struct):
  value: static[T]
#[[end namespace inner]] 
using(namespace(inner))
inner_helper = using(inner.Helper.spec(int))
#[[end namespace outer]] 

#[[namespace outer]]
#[[namespace inner]] 
class Helper2(struct):
  value2: static[T]
#[[end namespace inner]]
#[[end namespace outer]]
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
@pica.template({T: typename})
class AlignedStruct(struct[alignas(16)]):
  value: T
    
  class Inner(struct):
    type = using(T)
    ptr: ptr[T]
  
  data: ptr[typename(Inner.type)]
```

## 9. Friend Functions and Operators

### C++ Code
```cpp
template<typename U>
friend class OtherContainer;

friend std::ostream& operator<<(std::ostream& os, const Container& c) {
  return os << c.value;
}
```

### Python DSL Code
```python
@pica.template({U: typename})
@pica.cls(friend)
class OtherContainer:
  ...

@pica.fn(friend)
def __lshift__(os: ref[std.ostream], c: Const[ref[Container]]) -> ref[std.ostream]:
  return os << c.value  
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
@pica.template({T: typename})
class is_valid(struct):
  static_assert(sizeof(T) > 1, "Type too small")
  static_assert(std.is_default_constructible_v.spec(T), "Must be default constructible")
  
  pica.template_var({U: typename})
  is_convertible: static[constexpr[bool]] = std.is_convertible_v.spec(T, U)
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
smem = deref(reinterpret_cast.spec(ptr[SharedStorage])(shared_memory))
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
