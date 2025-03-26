"""
python cpp_to_python_dsl_conversion/add_missing_namespace_comments.py
"""

import re

def add_missing_namespace_comments(cpp_code: str) -> str:
    lines = cpp_code.splitlines()
    namespace_stack = []
    result_lines = lines.copy()
    
    for i, line in enumerate(lines):
        # Check for namespace declarations
        namespace_match = re.match(r'\s*namespace\s+(\w+)\s*{?', line)
        if namespace_match:
            namespace_name = namespace_match.group(1)
            namespace_stack.append(namespace_name)
        
        # Check for closing braces
        if '}' in line:
            # If there's already a namespace comment, verify it matches and pop
            existing_comment = re.search(r'//\s*namespace\s+(\w+)', line)
            if existing_comment:
                if namespace_stack and existing_comment.group(1) == namespace_stack[-1]:
                    namespace_stack.pop()
            # If no comment and we have namespaces in the stack, add a comment
            elif namespace_stack:
                # Only split the line if it's a standalone closing brace
                if line.strip() == '}':
                    result_lines[i] = line.rstrip() + f"  // namespace {namespace_stack.pop()}"
                else:
                    # For lines with code, only add the comment if it's a namespace closing brace
                    # Count opening and closing braces before this one
                    text_before_brace = line[:line.rindex('}')]
                    open_count = text_before_brace.count('{')
                    close_count = text_before_brace.count('}')
                    
                    # If this is a namespace closing brace (brace counts match)
                    if open_count == close_count:
                        result_lines[i] = line.rstrip() + f"  // namespace {namespace_stack.pop()}"
    
    return '\n'.join(result_lines)

def test_add_missing_namespace_comments():
    # Test case 1: Basic nested namespaces without end comments
    test_input1 = """namespace pica {
namespace inner {
int x = 1;
}
}  // namespace pica"""
    expected_output1 = """namespace pica {
namespace inner {
int x = 1;
}  // namespace inner
}  // namespace pica"""
    
    result1 = add_missing_namespace_comments(test_input1)
    assert result1 == expected_output1, f"\nExpected:\n{expected_output1}\nGot:\n{result1}"
    
    # Test case 2: Multiple nested namespaces
    test_input2 = """namespace outer {
namespace middle {
namespace inner {
void func() {}
}
}
}  // namespace outer"""
    expected_output2 = """namespace outer {
namespace middle {
namespace inner {
void func() {}
}  // namespace inner
}  // namespace middle
}  // namespace outer"""
    
    result2 = add_missing_namespace_comments(test_input2)
    assert result2 == expected_output2, f"\nExpected:\n{expected_output2}\nGot:\n{result2}"
    
    # Test case 3: Namespace with existing end comment
    test_input3 = """namespace test {
void func() {}
}  // namespace test"""
    expected_output3 = """namespace test {
void func() {}
}  // namespace test"""
    
    result3 = add_missing_namespace_comments(test_input3)
    assert result3 == expected_output3, f"\nExpected:\n{expected_output3}\nGot:\n{result3}"
    
    # Test case 4: Code on same line as closing brace
    test_input4 = """namespace test {
void func() { return; }
}"""
    expected_output4 = """namespace test {
void func() { return; }
}  // namespace test"""
    
    result4 = add_missing_namespace_comments(test_input4)
    assert result4 == expected_output4, f"\nExpected:\n{expected_output4}\nGot:\n{result4}"
    
    # Test case 5: Function with multiple braces
    test_input5 = """namespace test {
void func() { if (true) { return; } }
}"""
    expected_output5 = """namespace test {
void func() { if (true) { return; } }
}  // namespace test"""
    
    result5 = add_missing_namespace_comments(test_input5)
    assert result5 == expected_output5, f"\nExpected:\n{expected_output5}\nGot:\n{result5}"

    # Test case 7: Complex inline functions and brace patterns
    test_input7 = """namespace complex {
template<typename T>
class Test { public: void func() { if (true) { while(true) { do { } while(true); } } } };
}"""
    expected_output7 = """namespace complex {
template<typename T>
class Test { public: void func() { if (true) { while(true) { do { } while(true); } } } };
}  // namespace complex"""

    result7 = add_missing_namespace_comments(test_input7)
    assert result7 == expected_output7, f"\nExpected:\n{expected_output7}\nGot:\n{result7}"

    # Test case 8: Namespaces with comments and preprocessor directives
    test_input8 = """namespace preprocessor {
#ifdef DEBUG
namespace debug {
void debug_func() {}
} // namespace debug
#endif
namespace release {
void release_func() {}
}
}"""
    expected_output8 = """namespace preprocessor {
#ifdef DEBUG
namespace debug {
void debug_func() {}
} // namespace debug
#endif
namespace release {
void release_func() {}
}  // namespace release
}  // namespace preprocessor"""

    result8 = add_missing_namespace_comments(test_input8)
    assert result8 == expected_output8, f"\nExpected:\n{expected_output8}\nGot:\n{result8}"

    # Test case 10: Mixed existing and missing comments
    test_input10 = """namespace outer {
namespace middle {  // with inline comment
namespace inner {
int x = 0;
}  // namespace inner
}
}"""
    expected_output10 = """namespace outer {
namespace middle {  // with inline comment
namespace inner {
int x = 0;
}  // namespace inner
}  // namespace middle
}  // namespace outer"""

    result10 = add_missing_namespace_comments(test_input10)
    assert result10 == expected_output10, f"\nExpected:\n{expected_output10}\nGot:\n{result10}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_add_missing_namespace_comments()
