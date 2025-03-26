"""
1. `python cpp_to_python_dsl_conversion/format_cpp_code_oneline.py`
"""

import re
import sys
import unittest
import tempfile
import os
from typing import List, Tuple

"""
Formatting rules:

First Pass - Comment Conversion:
1. Process file from beginning to end
2. When encountering a single-line comment "// XYZ\n" (where \n is the line ending):
   - Convert it to multi-line comment format "/* XYZ*/"
   - Preserve any leading whitespace
   - Comments in string literals should not be converted

Second Pass - Line Joining:
1. Process file from beginning to end
2. Special cases to preserve:
   - Multi-line comments (/* ... */): Copy as-is
   - Empty lines (\n\n): Preserve as-is
   - Lines within a #if / #ifdef ... #else ... #endif block: Preserve as-is
3. Line joining rules:
   - Join consecutive lines by removing newlines
   - `{...}` where `...` is either empty or only whitespace: should be replaced with `{}`.
   - Make sure there is a newline after: `{` or `public:` or `private:` or `protected:` or `;`.
   - Make sure there is a newline before: `}` or `public:` or `private:` or `protected:`.
   - Error if encountering `/* ... \n` pattern during joining
   - When joining multiple lines together:
     - Keep the indentation level of the first line
     - Remove indentation from subsequent joined lines
     - This ensures consistent indentation in the output
     - For example:
       "    line1\n        line2" becomes "    line1line2"
"""

class FormatError(Exception):
    """Custom exception for formatting errors."""
    pass

def convert_single_line_comments(code: str) -> str:
    """Convert single-line comments to multi-line format while preserving string literals."""
    result = []
    i = 0
    in_string = False
    string_char = None
    in_preprocessor = False
    
    while i < len(code):
        # Handle preprocessor directives
        if not in_string and code[i] == '#':
            in_preprocessor = True
            result.append(code[i])
            i += 1
            continue
            
        # Handle string literals
        if not in_string and (code[i] == '"' or code[i] == "'"):
            in_string = True
            string_char = code[i]
            result.append(code[i])
            i += 1
            continue
        elif in_string:
            if code[i] == '\\' and i + 1 < len(code):
                # Skip escaped characters in strings
                result.extend(code[i:i+2])
                i += 2
                continue
            elif code[i] == string_char:
                in_string = False
            result.append(code[i])
            i += 1
            continue
            
        # Look for single-line comments
        if code[i:i+2] == '//':
            # Find the end of the comment (newline or EOF)
            comment_start = i
            i += 2  # Skip the //
            
            # Collect the comment text
            comment_text = []
            while i < len(code) and code[i] != '\n':
                comment_text.append(code[i])
                i += 1
                
            # Convert the comment, preserving exact whitespace
            if in_preprocessor:
                result.append('//')
                result.extend(comment_text)
            else:
                result.append('/*')
                result.extend(comment_text)
                result.append('*/')
            
            # Preserve newline if it exists
            if i < len(code):
                result.append(code[i])
                in_preprocessor = False
                i += 1
        else:
            result.append(code[i])
            i += 1
            
    return ''.join(result)

def format_code(code: str) -> str:
    """Format C++ code according to the specified rules."""
    # Convert single-line comments to multi-line comments
    code = convert_single_line_comments(code)

    # Split into lines and trim whitespace
    lines = [line.strip() for line in code.split('\n')]

    # Remove empty lines at start and end
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # Check for invalid multi-line comments
    for line in lines:
        if "/*" in line:
            if "*/" not in line:
                # Check if this is a valid multi-line comment start
                if line.strip() != line[line.index("/*"):].strip():
                    raise FormatError("Invalid multi-line comment detected")
            else:
                # Check if there's code after the comment
                after_comment = line[line.index("*/") + 2:].strip()
                if after_comment and not after_comment.startswith('{') and not after_comment.startswith('}'):
                    raise FormatError("Invalid multi-line comment detected")

    # Handle multi-line comments
    i = 0
    formatted_lines = []
    while i < len(lines):
        line = lines[i].strip()
        
        # Handle multi-line comments
        if "/*" in line and "*/" not in line:
            comment_lines = [line]
            j = i + 1
            while j < len(lines) and "*/" not in lines[j]:
                comment_lines.append(lines[j].strip())
                j += 1
            if j < len(lines):
                comment_lines.append(lines[j].strip())
                formatted_lines.extend(comment_lines)
                i = j + 1
            else:
                raise FormatError("Unclosed multi-line comment")
            continue
        
        formatted_lines.append(line)
        i += 1

    # Process lines for joining and formatting
    i = 0
    result = []
    while i < len(formatted_lines):
        current_line = formatted_lines[i]

        # Skip empty lines
        if not current_line:
            result.append(current_line)
            i += 1
            continue

        # Handle preprocessor directives
        if current_line.startswith('#'):
            result.append(current_line)
            i += 1
            continue

        # Handle complete multi-line comments
        if current_line.startswith('/*') and current_line.endswith('*/'):
            result.append(current_line)
            i += 1
            continue

        # Handle template declarations
        if current_line.startswith('template'):
            template_line = current_line
            j = i + 1
            while j < len(formatted_lines) and not formatted_lines[j].strip().endswith('{'):
                if not formatted_lines[j].strip().startswith('#'):
                    template_line += ' ' + formatted_lines[j].strip()
                else:
                    result.append(template_line)
                    template_line = formatted_lines[j].strip()
                j += 1
            if j < len(formatted_lines) and formatted_lines[j].strip().endswith('{'):
                template_line += ' ' + formatted_lines[j].strip()
            result.append(template_line)
            i = j + 1
            continue

        # Handle namespace declarations
        if current_line.startswith('namespace'):
            namespace_line = current_line
            j = i + 1
            while j < len(formatted_lines) and not formatted_lines[j].strip().endswith('{'):
                namespace_line += ' ' + formatted_lines[j].strip()
                j += 1
            if j < len(formatted_lines):
                namespace_line += ' ' + formatted_lines[j].strip()
            result.append(namespace_line)
            i = j + 1
            continue

        # Handle line joining for declarations and statements
        if i + 1 < len(formatted_lines):
            next_line = formatted_lines[i + 1]
            
            # Don't join lines ending with semicolon
            if current_line.endswith(';'):
                result.append(current_line)
                result.append('')  # Add empty line after semicolon
                i += 1
                continue

            # Don't join lines starting with access specifiers
            if next_line.strip() in ['public:', 'private:', 'protected:']:
                result.append(current_line)
                i += 1
                continue

            # Join lines that should be joined
            if (not current_line.endswith('{') and
                not current_line.endswith('}') and
                not current_line.endswith(':') and
                not next_line.startswith('{')):
                current_line += ' ' + next_line
                i += 2
                result.append(current_line)
                continue

        # Handle remaining lines
        result.append(current_line)
        i += 1

    # Format braces and add proper line breaks
    formatted_result = []
    i = 0
    while i < len(result):
        line = result[i].strip()

        # Handle empty blocks
        if '{' in line and '}' in line and line.count('{') == line.count('}'):
            # Format empty blocks with proper spacing
            parts = line.split('{')
            prefix = parts[0].rstrip()
            if '}' in parts[-1]:
                suffix = parts[-1].split('}')[1].lstrip()
                formatted_result.append(f"{prefix} {{}}{suffix}")
            else:
                formatted_result.append(f"{prefix} {{}}")
            i += 1
            continue

        # Handle opening braces
        if line.endswith('{'):
            formatted_result.append(line)
            i += 1
            continue

        # Handle closing braces
        if line.startswith('}'):
            if i > 0 and formatted_result[-1].strip():
                formatted_result.append('')
            formatted_result.append(line)
            if i + 1 < len(result) and result[i + 1].strip():
                formatted_result.append('')
            i += 1
            continue

        # Handle semicolons
        if line.endswith(';'):
            formatted_result.append(line)
            if i + 1 < len(result) and not result[i + 1].startswith('}'):
                formatted_result.append('')
            i += 1
            continue

        formatted_result.append(line)
        i += 1

    # Clean up empty lines
    while formatted_result and not formatted_result[-1].strip():
        formatted_result.pop()

    # Join lines and ensure proper spacing
    final_result = []
    for line in formatted_result:
        if line.strip():
            # Ensure proper spacing around braces
            line = re.sub(r'\s+}', '}', line)
            line = re.sub(r'{\s+', '{ ', line)
            line = re.sub(r'\s+{', ' {', line)
            line = re.sub(r'}\s+', '} ', line)
            
            # Handle special cases for templates and operators
            if 'template' in line:
                line = re.sub(r'template\s*<', 'template <', line)
                line = re.sub(r'>\s*class', '> class', line)
            if 'operator' in line:
                line = re.sub(r'operator\s*([^\w\s])', r'operator\1', line)
            
            final_result.append(line)
        else:
            final_result.append('')

    # Clean up consecutive empty lines
    i = 0
    while i < len(final_result) - 1:
        if not final_result[i].strip() and not final_result[i + 1].strip():
            final_result.pop(i)
        else:
            i += 1

    return '\n'.join(final_result)

class NameGenerator:
    """Generates random but valid C++ identifiers."""
    def __init__(self):
        import random
        import string
        self.random = random
        self.string = string
        self.used_names = set()
        
    def _generate_base(self, prefix=''):
        length = self.random.randint(3, 10)
        name = prefix + ''.join(self.random.choices(self.string.ascii_uppercase + self.string.ascii_lowercase, k=length))
        if name in self.used_names:
            return self._generate_base(prefix)
        self.used_names.add(name)
        return name
        
    def get_type_name(self):
        return self._generate_base('T')
        
    def get_var_name(self):
        return self._generate_base('v')
        
    def get_class_name(self):
        return self._generate_base('C')
        
    def get_template_name(self):
        return self._generate_base('Tmpl')
        
    def reset(self):
        self.used_names.clear()

# Unit Tests
class TestCppCodeFormatter(unittest.TestCase):
    maxDiff = None
    
    def setUp(self):
        self.name_gen = NameGenerator()
        
    def test_single_line_comment_conversion(self):
        """Test conversion of single-line comments."""
        test_cases = [
            # Basic comment conversion
            ("// Simple comment", "/* Simple comment*/"),
            ("code; // Comment", "code; /* Comment*/"),
            ("    // Indented comment", "    /* Indented comment*/"),
            
            # Comments in strings should not be converted
            ('printf("// Not a comment");', 'printf("// Not a comment");'),
            ("std::string s = \"// Still not a comment\";", "std::string s = \"// Still not a comment\";"),
            
            # Multiple comments
            ("code // Comment 1\nmore // Comment 2", "code /* Comment 1*/\nmore /* Comment 2*/"),
            
            # Empty comments
            ("//", "/**/"),
            ("code //", "code /**/"),
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = convert_single_line_comments(input_code)
                self.assertEqual(result.strip(), expected.strip())

    def test_format_code(self):
        """Test code formatting rules."""
        test_cases = [
            # Basic formatting
            (
                "int main() {\nreturn 0;}",
                "int main() {\nreturn 0;\n}"
            ),
            
            # Empty lines preservation
            (
                "class A {\n\npublic:\n\nint x;}",
                "class A {\n\npublic:\n\nint x;\n}"
            ),
            
            # Line joining
            (
                "int x;\nint y;\nint z;",
                "int x;\nint y;\nint z;"
            ),
            
            # Multi-line comments
            (
                "/* Multi\nline\ncomment */\nint x;",
                "/* Multi\nline\ncomment */\nint x;"
            ),

            # Multi-line comments
            (
                "/*******\nMulti\nline\ncomment\n*******/\nint x;",
                "/*******\nMulti\nline\ncomment\n*******/\nint x;"
            ),
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = format_code(input_code)
                # Normalize whitespace for comparison
                result_normalized = ' '.join(result.strip().split())
                expected_normalized = ' '.join(expected.strip().split())
                self.assertEqual(result_normalized, expected_normalized)

    def test_format_error_cases(self):
        """Test cases that should raise FormatError."""
        error_cases = [
            # Multi-line comment in middle of line
            "int x = 1; /* Multi\nline comment */",
            
            # Multi-line comment in middle of line
            "call<int, int>(/* Multi\nline comment */1, 2);",
            
            # Unclosed multi-line comment
            "/* Unclosed\ncomment",
        ]
        
        for error_case in error_cases:
            with self.subTest(error_case=error_case):
                with self.assertRaises(FormatError):
                    format_code(error_case)

    def test_end_to_end(self):
        """Test complete formatting process with file I/O."""
        test_code = """
        // Header comment
        class MyClass {
            public:
            // Method comment
            void method() {
                int x; // Variable comment
                // Another comment
            }
        };
        """
        
        expected = """
        /* Header comment*/
        class MyClass {
            public:
            /* Method comment*/
            void method() {
                int x; /* Variable comment*/
                /* Another comment*/
            }
        };
        """
        
        # Create temporary files for testing
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as input_file:
            input_file.write(test_code)
            input_path = input_file.name
            
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as output_file:
            output_path = output_file.name
            
        try:
            # Run the formatter
            main(input_path, output_path)
            
            # Read and verify the output
            with open(output_path, 'r') as f:
                result = f.read()
                
            # Compare content while ignoring whitespace differences
            expected_normalized = ''.join(expected.strip().split())
            result_normalized = ''.join(result.strip().split())
            self.assertEqual(result_normalized, expected_normalized)
            
        finally:
            # Clean up temporary files
            os.unlink(input_path)
            os.unlink(output_path)

    def test_line_joining_rules(self):
        """Test line joining according to formatting rules."""
        test_cases = [
            # Basic line joining
            (
                "int x\nint y\nint z;",
                "int x int y int z;"
            ),
            
            # Line breaks after specific tokens
            (
                "class MyClass {\npublic:\nint x;};",
                "class MyClass {\npublic:\nint x;\n};"
            ),
            (
                "class MyClass {\nprivate:\nint x;};",
                "class MyClass {\nprivate:\nint x;\n};"
            ),
            (
                "class MyClass {\nprotected:\nint x;};",
                "class MyClass {\nprotected:\nint x;\n};"
            ),
            
            # Multiple statements
            (
                "int x = 1;int y = 2;int z = 3;",
                "int x = 1;\nint y = 2;\nint z = 3;"
            ),
            
            # Empty lines preservation
            (
                "int x;\n\nint y;\n\nint z;",
                "int x;\n\nint y;\n\nint z;"
            ),

            # Nested namespaces
            (
                "namespace A{namespace B {int x;}}",
                "namespace A {\nnamespace B {\nint x;\n}\n}"
            ),

            # Deeply nested curly braces
            (
                "{\nint x = 1;\n{\nint y = 2;\n{\nint z = 3;}\n}}",
                "{\nint x = 1;\n{\nint y = 2;\n{\nint z = 3;\n}\n}\n}"
            ),
            
            # Complex case with multiple rules
            (
                "template<typename T>\nclass Container {\npublic:\nT value;\nvoid set(T v) {\nvalue = v;}};",
                "template<typename T>class Container {\npublic:\nT value;\nvoid set(T v) {\nvalue = v;\n}\n};"
            )
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = format_code(input_code)
                self.assertEqual(result, expected)

    def test_invalid_multiline_comment_during_joining(self):
        """Test error cases for multi-line comments during line joining."""
        name_gen = self.name_gen
        
        def create_error_case_1():
            type1, type2 = [name_gen.get_type_name() for _ in range(2)]
            var1 = name_gen.get_var_name()
            return f"int {var1}\n/* Comment\n*/ int {type2};"
            
        def create_error_case_2():
            type1 = name_gen.get_type_name()
            var1 = name_gen.get_var_name()
            return f"template<typename {type1}\n/* Comment\n*/> class {var1};"
            
        def create_error_case_3():
            type1, type2 = [name_gen.get_type_name() for _ in range(2)]
            var1, var2 = [name_gen.get_var_name() for _ in range(2)]
            class1 = name_gen.get_class_name()
            return f"{class1}<{type1}, {type2}>({var1}, /* Multi\nline comment */{var2});"

        error_case_generators = [
            create_error_case_1,
            create_error_case_2,
            create_error_case_3
        ]
        
        for i, create_error_case in enumerate(error_case_generators):
            with self.subTest(f"Invalid multiline comment case {i+1}"):
                name_gen.reset()  # Reset used names for each test case
                error_case = create_error_case()
                with self.assertRaises(FormatError):
                    format_code(error_case)

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and corner scenarios."""
    maxDiff = None
    def test_nested_comments(self):
        """Test handling of nested-looking comments."""
        test_cases = [
            # Comment containing //
            ("// Contains // another comment", "/* Contains // another comment*/"),
            
            # Comment containing /* */
            ("// Contains /* inline */ comment", "/* Contains /* inline */ comment*/"),
            
            # Multiple // in one line
            ("code // First // Second", "code /* First // Second*/"),
            
            # Comment with special regex characters
            ("// Comment with * and + and [brackets]", "/* Comment with * and + and [brackets]*/"),

            # Comment with only slashes
            (
                "////////////////////////////////////////////////////////////////////////////////////////////////////",
                "/*//////////////////////////////////////////////////////////////////////////////////////////////////*/"
            )
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = convert_single_line_comments(input_code)
                self.assertEqual(result.strip(), expected.strip())

    def test_unicode_handling(self):
        """Test handling of Unicode characters in comments."""
        test_cases = [
            # Unicode in comments
            ("// 你好世界", "/* 你好世界*/"),
            ("// Café", "/* Café*/"),
            
            # Emoji in comments
            ("// Comment with 😊 emoji", "/* Comment with 😊 emoji*/"),
            
            # Mixed Unicode and ASCII
            ("int x = 42; // Value = 值", "int x = 42; /* Value = 值*/"),
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = convert_single_line_comments(input_code)
                self.assertEqual(result.strip(), expected.strip())

    def test_whitespace_handling(self):
        """Test handling of various whitespace scenarios."""
        test_cases = [
            # Tabs in comments
            ("//\tTabbed comment", "/*\tTabbed comment*/"),
            
            # Multiple spaces
            ("//    Multiple    spaces    ", "/*    Multiple    spaces    */"),
            
            # Mixed tabs and spaces
            ("//\t  \t  Mixed\tspaces\t  ", "/*\t  \t  Mixed\tspaces\t  */"),
            
            # No space after //
            ("//No space after slashes", "/*No space after slashes*/"),
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = convert_single_line_comments(input_code)
                self.assertEqual(result.strip(), expected.strip())

class TestCppSyntaxScenarios(unittest.TestCase):
    """Test handling of complex C++ syntax scenarios."""
    maxDiff = None
    
    def setUp(self):
        self.name_gen = NameGenerator()
        
    def test_template_syntax(self):
        """Test handling of C++ template syntax."""
        test_cases = [
            # Template declaration with comments
            (
                "template<typename T> // Template parameter\nclass Container { };",
                "template<typename T> /* Template parameter*/\nclass Container {};"
            ),
            
            # Nested templates
            (
                "template<template<typename> class T> // Nested template\nclass Wrapper {};",
                "template<template<typename> class T> /* Nested template*/\nclass Wrapper {};"
            ),
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = convert_single_line_comments(input_code)
                self.assertEqual(result.strip(), expected.strip())

    def test_complex_template_declaration(self):
        """Test handling of complex template function declarations with line breaks, comments, and includes."""
        name_gen = self.name_gen
        
        def create_test_case_1():
            type1, type2, type3, type4, type5 = [name_gen.get_type_name() for _ in range(5)]
            var1, var2, var3, var4 = [name_gen.get_var_name() for _ in range(4)]
            class1 = name_gen.get_class_name()
            return f"""
            template <class {type1}, class {type2}, class {type3}, class {type4}, class {type5}>
            __global__ static __launch_bounds__(decltype(size(
                {type5}{{}})::value)) void {class1}({type1} {var1}, {type2} const* {var2},
                                              CUTLASS_GRID_CONSTANT {type3} const
                                                  {var3},
                                              {type4}* {var4}, {type5} {var1}) {{
                std::cout << "Hello, World!" << std::endl;
            }}
                """, f"""
            template <class {type1}, class {type2}, class {type3}, class {type4}, class {type5}>__global__ static __launch_bounds__(decltype(size({type5}{{}})::value)) void {class1}({type1} {var1}, {type2} const* {var2}, CUTLASS_GRID_CONSTANT {type3} const {var3}, {type4}* {var4}, {type5} {var1}) {{
                std::cout << "Hello, World!" << std::endl;
            }}
                """

        def create_test_case_2():
            type1, type2 = [name_gen.get_type_name() for _ in range(2)]
            var1, var2 = [name_gen.get_var_name() for _ in range(2)]
            func = name_gen.get_var_name()
            return f"""
            template <class {type1}, // First template parameter
                     class {type2}>  // Second template parameter
            void {func}({type1} {var1}, 
                    {type2} {var2}) {{ // Function body starts here
                // Implementation code
            }}
                """, f"""
            template <class {type1}, /* First template parameter*/class {type2}>  /* Second template parameter*/ void {func}({type1} {var1}, {type2} {var2}) {{ /* Function body starts here*/
                /* Implementation code*/
            }}
                """

        def create_test_case_3():
            type1 = name_gen.get_type_name()
            class1 = name_gen.get_class_name()
            var1, var2 = [name_gen.get_var_name() for _ in range(2)]
            return f"""
            #include <vector>
            #include <algorithm>
            
            template <typename {type1}>
            class {class1} {{
                // Vector to store elements
                std::vector<{type1}> {var1};
            public:
                void {var2}({type1} element) {{
                    {var1}.push_back(element);
                }}
            }};
                """, f"""
            #include <vector>
            #include <algorithm>
            
            template <typename {type1}> class {class1} {{
                /* Vector to store elements*/
                std::vector<{type1}> {var1};
            public:
                void {var2}({type1} element) {{
                    {var1}.push_back(element);
                }}
            }};
                """

        def create_test_case_4():
            type1 = name_gen.get_type_name()
            class1, class2 = [name_gen.get_class_name() for _ in range(2)]
            var1 = name_gen.get_var_name()
            return f"""
            template <
                typename... {type1}s, // Parameter pack
                template <typename...> class {class1}, // Template template parameter
                int {var1} = 42 // Non-type template parameter with default
            >
            struct {class2} {{
                // Implementation details
                template <typename {type1}>
                void method() {{
                    // Nested method
                }}
            }};
                """, f"""
            template <typename... {type1}s, /* Parameter pack*/template <typename...> class {class1}, /* Template template parameter*/int {var1} = 42 /* Non-type template parameter with default*/>struct {class2} {{
                /* Implementation details*/
                template <typename {type1}>void method() {{
                    /* Nested method*/
                }}
            }};
                """

        def create_test_case_5():
            type1, type2 = [name_gen.get_type_name() for _ in range(2)]
            class1 = name_gen.get_class_name()
            return f"""
            #if defined(DEBUG2)
            // Debug version
            template <typename {type2}>
            #endif

            #ifdef DEBUG
            // Debug version
            template <typename {type1}>
            #else
            // Release version
            template <typename {type1}, typename {type2}>
            #endif
            class {class1} {{
                // Implementation
            }};
                """, f"""
            #if defined(DEBUG2)
            /* Debug version*/
            template <typename {type2}>
            #endif

            #ifdef DEBUG
            /* Debug version*/
            template <typename {type1}>
            #else
            /* Release version*/
            template <typename {type1}, typename {type2}>
            #endif
            class {class1} {{
                /* Implementation*/
            }};
                """

        def create_test_case_6():
            type1 = name_gen.get_type_name()
            func = name_gen.get_var_name()
            var1 = name_gen.get_var_name()
            return f"""
            template <
                typename {type1},
                typename = typename std::enable_if<
                    std::is_integral<{type1}>::value && // Only for integral types
                    !std::is_same<{type1}, bool>::value, // But not for bool
                    void // Default type
                >::type
            >
            void {func}({type1} {var1}) {{
                // Implementation
            }}
                """, f"""
            template <typename {type1}, typename = typename std::enable_if<std::is_integral<{type1}>::value && /* Only for integral types*/ !std::is_same<{type1}, bool>::value, /* But not for bool*/ void /* Default type*/>::type>void {func}({type1} {var1}) {{
                /* Implementation*/
            }}
                """

        test_case_generators = [
            create_test_case_1,
            create_test_case_2,
            create_test_case_3,
            create_test_case_4,
            create_test_case_5,
            create_test_case_6
        ]
        
        for i, create_test_case in enumerate(test_case_generators):
            with self.subTest(f"Complex template test case {i+1}"):
                name_gen.reset()  # Reset used names for each test case
                input_code, expected = create_test_case()
                
                # First pass: comment conversion
                intermediate = convert_single_line_comments(input_code)
                # Second pass: formatting
                result = format_code(intermediate)
                
                # Compare without whitespace and newlines for easier comparison
                self.assertEqual(result, expected)

    def test_operator_overloading(self):
        """Test handling of operator overloading syntax."""
        test_cases = [
            # Member operator with comment
            (
                "void operator/(int x) // Divide operator\n{/* impl */ }",
                "void operator/(int x) /* Divide operator*/{\n/* impl */ \n}"
            ),
        ]
        
        for input_code, expected in test_cases:
            with self.subTest(input_code=input_code):
                result = convert_single_line_comments(input_code)
                self.assertEqual(result.strip(), expected.strip())

class TestErrorHandling(unittest.TestCase):
    """Test error handling scenarios."""
    maxDiff = None
    def test_invalid_content(self):
        """Test handling of invalid content."""
        test_cases = [
            # Binary content
            b"\x00\x01\x02".decode('utf-8', errors='ignore'),
            
            # Invalid UTF-8 sequences
            "// Comment with \uD800 surrogate",  # Unpaired surrogate
            
            # Extremely long lines
            "// " + "x" * 10000,
        ]
        
        for test_case in test_cases:
            with self.subTest(test_case=test_case[:20] + "..."):
                # Should not raise any exceptions
                result = convert_single_line_comments(test_case)
                self.assertIsInstance(result, str)

def main(input_file: str, output_file: str = None) -> None:
    """
    Main function to format C++ code
    
    Args:
        input_file (str): Path to input C++ file
        output_file (str): Optional path to output file. If not provided, prints to stdout
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        PermissionError: If file access is denied
        FormatError: If code formatting fails
    """
    try:
        with open(input_file, 'r') as f:
            content = f.read()
            
        # First pass: Convert single-line comments
        intermediate = convert_single_line_comments(content)
        
        # Second pass: Format code
        formatted = format_code(intermediate)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(formatted)
        else:
            print(formatted)
            
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found", file=sys.stderr)
        raise
    except PermissionError:
        print(f"Error: Permission denied accessing file '{input_file}'", file=sys.stderr)
        raise
    except FormatError as e:
        print(f"Formatting error: {str(e)}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        main(input_file, output_file)
    else:
        print("Running tests...")
        # Explicitly run each test class
        test_classes = [
            TestCppCodeFormatter,
            TestEdgeCases,
            TestCppSyntaxScenarios,
            TestErrorHandling
        ]
        
        suite = unittest.TestSuite()
        for test_class in test_classes:
            tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
            
        runner = unittest.TextTestRunner()
        runner.run(suite)
