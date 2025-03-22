# C++ to Python DSL translation examples

This file provides examples mapping C++ code to our Python DSL. Each example consists of a C++ snippet and its equivalent DSL snippet. Follow these examples to translate additional C++ files.

Guidelines to follow:
- Use 2 spaces for indentation.
- Both `->` and `.` in C++ translate to `.` in Python DSL.
- Template specialization using <...> in C++ translates to .spec(...) in Python DSL.
  For example:
  - C++: `std::vector<int>` -> Python DSL: `std.vector.spec(int)`
  - C++: `MyClass<T, U>` -> Python DSL: `MyClass.spec(T, U)`
  - C++: `uint64_t tma_barrier[size<2>(SmemLayoutA{})]` -> Python DSL: `tma_barrier: List[uint64_t, size.spec(2)(SmemLayoutA)]`

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
  
  CUTLASS_HOST_DEVICE
  T& operator[](int idx) {
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
    return x * PI;
  }
};
```

### Python DSL Code
```python
@pica.cls(template=True)
class Buffer:
  def __new__(_, T: typename, N: int):
    data: Public[List[T, N]]
    
    @attrs(Public, CUTLASS_HOST_DEVICE())
    def __getitem__(self, idx: int) -> Ref[T]:
      return data[idx]

    size: Private[Static[ConstExpr[int]]] = N
    ptr: Private[Ptr[T]]

@pica.cls
class NonTemplateClass:
  PI: Public[Static[ConstExpr[float]]] = 3.14159f
  
  @attrs(Public, CUTLASS_HOST_DEVICE(), Const)
  def compute(self, x: float) -> float:
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
@pica.func(template=True)
@attrs(__global__)
def vectorAdd(T: typename) -> None:
  a: Ptr[T]
  b: Ptr[T] 
  c: Ptr[T]
  n: int

  idx = blockIdx.x * blockDim.x + threadIdx.x
  if idx < n:
    c[idx] = a[idx] + b[idx]

@attrs(__device__)
def deviceFunc(x: float) -> float:
  return x * x

# Launch
vectorAdd[grid, block](d_a, d_b, d_c, n)
```

## 4. Advanced Template Features

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
  def __new__(_, *Args: typename):
    @attrs(Static, CUTLASS_HOST_DEVICE())
    def process(First: typename, *Rest: typename) -> None:
      f: Forward[First]
      rest: Forward[Rest]
      # Process variadic args

@pica.struct(template=True)
class smart_pointer:
  def __new__(_, T: typename, Enable: typename = void):
    is_smart_ptr: Static[ConstExpr[bool]] = False

@pica.struct(template=True)
class smart_pointer:
  def __new__(_, T: typename, Enable: typename = std.enable_if_t.spec(
      std.is_same_v.spec(typename(Ptr[T.element_type]), decltype(std.declval.spec(T)().get()))
    )):
    is_smart_ptr: Static[ConstExpr[bool]] = True
```

## 5. Lambda Functions and Captures

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

@pica.lambda(captures=Ref[all], mutable)
def lambda2(x: int) -> int:
  multiplier *= 2
  return x * multiplier

@pica.lambda(captures=all, mutable) 
def lambda3(x: int) -> int:
  multiplier *= 2
  return x * multiplier

@pica.lambda(template=True)
def generic_lambda(T: typename)-> Any:
  x: Forward[T]
  y: Forward[auto]

  return std.forward.spec(T)(x) + y
```

## 6. Namespaces and Using Directives

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
    def __new__(_, T: typename):
      value: Static[T]
#} [[end namespace]] inner
#[[using namespace]] inner
#[[using]] inner_helper = inner.Helper.spec(int)
#} [[end namespace]] outer
```

## 7. Member Access and Alignment Specifications

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
@pica.struct(template=True)
class AlignedStruct:
  @attrs(alignas(16))
  def __new__(_, T: typename):
    value: T
    
    @pica.struct
    class Inner:
      type: typename = T
      ptr: Ptr[T]
    
    data: Ptr[Inner.type]
```

## 8. Friend Functions and Operators

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
  def __new__(_, T: typename):
    value: Private[T]
    
    @attrs(Friend, Template)
    class OtherContainer:
      U: typename
    
    @attrs(Friend)
    def __lshift__(os: Ref[std.ostream], c: Const[Ref[Container]]) -> Ref[std.ostream]:
      return os << c.value
    
    def __iadd__(self, rhs: Const[Ref[Container]]) -> Ref[Container]:
      self.value += rhs.value
      return self.deref()
```

## 9. Type Traits and Static Assertions

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
  def __new__(_, T: typename):
    static_assert(sizeof(T) > 1, "Type too small")
    static_assert(std.is_default_constructible_v.spec(T), "Must be default constructible")
    
    @pica.cls(template=True)
    class is_convertible:
      def __new__(_, U: typename):
        return std.is_convertible_v.spec(T, U)
```

## 10. Operator Overloading Examples

### 10.1 Arithmetic Operators
cpp```operator+``` => python```__add__```
cpp```operator-``` => python```__sub__```
cpp```operator*``` => python```__mul__```
cpp```operator/``` => python```__truediv__```
cpp```operator%``` => python```__mod__```
cpp```operator+()``` => python```__pos__```  // unary plus
cpp```operator-()``` => python```__neg__```  // unary minus

### 10.2 Compound Assignment
cpp```operator+=``` => python```__iadd__```
cpp```operator-=``` => python```__isub__```
cpp```operator*=``` => python```__imul__```
cpp```operator/=``` => python```__idiv__```
cpp```operator%=``` => python```__imod__```

### 10.3 Increment/Decrement
cpp```operator++()``` => python```__preinc__```     // pre-increment
cpp```operator--()``` => python```__predec__```     // pre-decrement
cpp```operator++(int)``` => python```__postinc__```  // post-increment
cpp```operator--(int)``` => python```__postdec__```  // post-decrement

### 10.4 Comparison Operators
cpp```operator==``` => python```__eq__```
cpp```operator!=``` => python```__ne__```
cpp```operator<``` => python```__lt__```
cpp```operator<=``` => python```__le__```
cpp```operator>``` => python```__gt__```
cpp```operator>=``` => python```__ge__```

### 10.5 Logical Operators
cpp```operator&&``` => python```__and__```
cpp```operator||``` => python```__or__```
cpp```operator!``` => python```__not__```

### 10.6 Bitwise Operators
cpp```operator&``` => python```__band__```
cpp```operator|``` => python```__bor__```
cpp```operator^``` => python```__xor__```
cpp```operator~``` => python```__invert__```
cpp```operator<<``` => python```__lshift__```
cpp```operator>>``` => python```__rshift__```
cpp```operator&=``` => python```__iband__```
cpp```operator|=``` => python```__ibor__```
cpp```operator^=``` => python```__ixor__```
cpp```operator<<=``` => python```__ilshift__```
cpp```operator>>=``` => python```__irshift__```

### 10.7 Array Subscript Operator
cpp```operator[]``` => python```__getitem__```

### 10.8 Function Call Operator
cpp```operator()``` => python```__call__```

### 10.9 Member Access Operators
cpp```operator->``` => python```__arrow__```
cpp```operator*``` => python```__deref__```

### 10.10 Stream Operators
cpp```operator<<``` => python```__lshift__```
cpp```operator>>``` => python```__rshift__```