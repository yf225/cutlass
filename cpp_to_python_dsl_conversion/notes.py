"""
NOTE(yf225): how to implement CUTLASS C++ -> PICA conversion:
It should be driven by LLM, but verified with our PICA -> CUTLASS C++ convertor,
to ensure that the PICA code can be used to generate back the exact same CUTLASS C++ code (after C++ formatting).
The LLM-driven C++ -> PICA conversion process should be as follows:
0. Run clang-format==20.1.0 and then add_missing_namespace_comments.py on the C++ file.
1. Use LLM to split the C++ file into logical blocks: split_cpp_file_into_codeblocks.prompt, save each block into a separate file.
2. Use LLM to convert each block into PICA code: convert_cpp_codeblock_to_pica.prompt, save each block into a separate file.
3. Combine the PICA code blocks, and verify the code by using the PICA -> CUTLASS C++ convertor to check if the generated PICA code (entire file) can be used to
generate back the exact same CUTLASS C++ code (clang-format==20.1.0 formatted). If they don't match, identify the mismatched lines, and think how to fix them / ask LLM to regenerate them based on the mismatch.
4. Repeat the process until the generated PICA code can be used to generate back the exact same CUTLASS C++ code.

NOTE(yf225): how C++ namespace is handled in PICA
1. Translation from C++ to Python
Case 1:
```
namespace cute {
...
}
```
=> 
```
#[[namespace]] cute {
...
#} [[end namespace]]
```

Case 2:
`#include <cute/tensor.hpp>` => Must specify what's being used from the header file, e.g. `from cute.tensor import X`
During the CUTLASS C++ -> PICA conversion, we should be able to know what's being used from the header file
by searching unknown symbols through all the PICA-managed `#include`-ed header files
(we always convert starting from the leaf header files, so for downstream files, we should only need to do symbol searching
within already-converted PICA-managed header files which are Python files (so easier to AST parse and find symbols)).

Case 3:
`using namespace cute;` => PICA-managed files will not have this line - it will be generated during C++ codegen by looking up what namespace each symbol in the file belongs to.
Specifically, we will maintain a list of symbols under each namespace (written within cute/__namespace__.py, cutlass/__namespace__.py, cute/detail/__namespace__.py, etc. files).
These __namespace__.py files are checked in to the repo (not dynamically generated), and will be used to
do symbol lookup and populate the `using namespace X` line during C++ codegen.
Q: How are __namespace__.py files generated?
A: We will have an additional "populate" step after any PICA-managed files are changed (as an offline step before committing the changes),
which walks through the changed file to detect new additions to the namespace, and
add the new symbols to the corresponding __namespace__.py files.
"""