import os
import re
import sys
from typing import Set, List, Tuple, Dict
from collections import defaultdict

# List of known external dependencies to exclude from in-repo analysis
# Default include directories to search for headers
INCLUDE_DIRS = [
    'include/',
    'tools/util/include/',
    # Add more include directories as needed
]

EXTERNAL_DEPS = [
    # CUDA headers
    'cuda_runtime.h',
    'cuda_runtime_api.h',
    'cuda.h',
    'device_launch_parameters.h',
    'cuda_fp16.h',
    'curand.h',
    'cublas_v2.h',
    'cusparse.h',

    # CUTLASS headers
    'cutlass/version_extended.h',
    'cutlass/arch/array.h',
    'cutlass/arch/numeric_types.h',

    # Standard C++ headers (common ones)
    'vector',
    'string',
    'iostream',
    'memory',
    'utility',
    'algorithm',
    'type_traits',
    'cstdint',
    'cstddef',
    'cstdlib',
    'cassert',
    'functional',
    'array',
    'tuple',
    'limits',
    'chrono',
    'thread',
    'mutex',
    'atomic',
    
    # Add more external dependencies as needed
]

# List of regex patterns for path-based external dependency matching
# Each pattern will be matched against the normalized include path
EXTERNAL_PATH_PATTERNS = [
    r'^cuda/std/',  # Matches any path starting with cuda/std/
    # Add more patterns as needed, for example:
    # r'^external/',  # Match paths starting with external/
    # r'.*\.internal\.h$',  # Match paths ending with .internal.h
]

# Compile the patterns for better performance
EXTERNAL_PATH_REGEXES = [re.compile(pattern) for pattern in EXTERNAL_PATH_PATTERNS]

def is_header_file(file_path: str) -> bool:
    """Check if a file is a header file."""
    return file_path.endswith(('.h', '.hpp', '.cuh', '.hxx', '.inl'))

def find_includes(file_path: str, visited: Set[str], dependencies: Dict[str, Set[str]]) -> List[str]:
    """Recursively find all header files included by file_path."""
    headers = []
    if not os.path.exists(file_path) or file_path in visited:
        return headers

    visited.add(file_path)
    normalized_path = normalize_include_path(file_path)
    # print(f"\nProcessing file: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # Regex to match both quoted and angle-bracket includes
            match = re.search(r'^\s*#\s*include\s+(?:"([^"]+)"|<([^>]+)>)', line)
            if match:
                # Get the header file name (from either "..." or <...>)
                include_file = match.group(1) or match.group(2)
                # print(f"  Found include: {include_file}")
                
                # For quoted includes, try multiple search paths
                if match.group(1):
                    found = False
                    # First try relative to current file
                    search_paths = [os.path.dirname(file_path)]
                    # Then try all configured include directories
                    search_paths.extend(INCLUDE_DIRS)
                    
                    for base_dir in search_paths:
                        include_path = os.path.normpath(os.path.join(base_dir, include_file))
                        if os.path.exists(include_path):
                            # Store the path with appropriate include directory prefix
                            if base_dir in INCLUDE_DIRS:
                                stored_path = include_path  # Keep full path for display
                            else:
                                # For files relative to current file, try to determine which include dir they belong to
                                for inc_dir in INCLUDE_DIRS:
                                    if os.path.exists(os.path.join(inc_dir, include_file)):
                                        stored_path = os.path.join(inc_dir, include_file)
                                        break
                                else:
                                    stored_path = include_path
                            
                            headers.append(stored_path)
                            # Add dependency relationship using normalized path
                            norm_include = normalize_include_path(stored_path)
                            dependencies[normalized_path].add(norm_include)
                            headers.extend(find_includes(include_path, visited, dependencies))
                            found = True
                            break
                    
                    if not found:
                        if not is_external_dependency(include_file):
                            raise Exception(f"Could not find {include_file} in any of the include directories")
                        else:
                            # For external dependencies, just record the name as a system include
                            system_include = f"<{include_file}>"
                            headers.append(system_include)
                            # Add system include as a dependency
                            dependencies[normalized_path].add(system_include)
                else:
                    # For system includes, just record the name
                    system_include = f"<{include_file}>"
                    headers.append(system_include)
                    # Add system include as a dependency
                    dependencies[normalized_path].add(system_include)
    
    return headers

def find_all_headers(directory: str) -> List[str]:
    """Find all header files in the given directory recursively."""
    headers = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_header_file(file):
                headers.append(os.path.join(root, file))
    return headers

def normalize_include_path(include: str) -> str:
    """Normalize include paths by removing angle brackets and standardizing format."""
    # Remove angle brackets if present
    if include.startswith('<') and include.endswith('>'):
        include = include[1:-1]
    # Convert Windows-style paths to forward slashes
    include = include.replace('\\', '/')
    # For dependency tracking, we still want to normalize the path
    # by removing any include directory prefix
    for inc_dir in INCLUDE_DIRS:
        if include.startswith(inc_dir):
            include = include[len(inc_dir):]
            break
    return include

def has_header_extension(path: str) -> bool:
    """Check if a path has a header file extension."""
    # Use the same extensions as is_header_file
    return path.endswith(('.h', '.hpp', '.cuh', '.hxx', '.inl'))

def format_include_for_display(include: str) -> str:
    """Format include path for display in summary."""
    # For paths without header extensions, keep the angle brackets
    if include.startswith('<') and include.endswith('>'):
        inner_path = include[1:-1]
        if not has_header_extension(inner_path):
            return include
        include = inner_path

    # Keep the include directory prefix for display
    include = include.replace('\\', '/')
    return include

def count_file_lines(file_path: str) -> int:
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error counting lines in {file_path}: {str(e)}")
        return 0

def analyze_dependencies(dependencies: Dict[str, Set[str]]) -> List[Tuple[str, int, bool, int]]:
    """Analyze dependencies to find leaf headers and dependency counts, only counting in-repo header files."""
    # Calculate total dependencies (direct + indirect) for each header
    total_deps = {}
    leaf_headers = set()
    line_counts = {}  # Store line counts for each header
    
    def is_in_repo_header(dep: str) -> bool:
        """Check if dependency is an in-repo header file."""
        # Remove angle brackets if present for comparison
        clean_dep = dep.strip('<>')
        
        # Check against path-based patterns
        for regex in EXTERNAL_PATH_REGEXES:
            if regex.search(clean_dep):
                return False
                
        # Check other conditions
        return (has_header_extension(dep) and 
                not dep.startswith('<') and 
                clean_dep not in EXTERNAL_DEPS)
    
    # First, filter dependencies to only include in-repo headers and map normalized paths back to full paths
    normalized_to_full_path = {}  # Keep track of full paths
    filtered_deps = {}
    
    for header, deps in dependencies.items():
        # Try to find the full path with include directory
        full_header_path = header
        for inc_dir in INCLUDE_DIRS:
            if os.path.exists(os.path.join(inc_dir, header)):
                full_header_path = os.path.join(inc_dir, header).replace('\\', '/')
                break
        
        normalized_to_full_path[header] = full_header_path
        filtered_deps[header] = {dep for dep in deps if is_in_repo_header(dep)}
        
        # Count lines in the file if it's a real file path
        if not header.startswith('<'):
            # Try each include directory
            for inc_dir in INCLUDE_DIRS:
                file_path = os.path.join(inc_dir, header)
                if os.path.exists(file_path):
                    line_counts[header] = count_file_lines(file_path)
                    break
            
        # If a header has no in-repo dependencies, it's a leaf
        if not filtered_deps[header]:
            leaf_headers.add(header)
        
        # Count total in-repo dependencies
        visited = set()
        def count_deps(h: str) -> None:
            if h in visited:
                return
            visited.add(h)
            for dep in filtered_deps.get(h, set()):
                if is_in_repo_header(dep):
                    count_deps(dep)
        
        count_deps(header)
        # Subtract 1 to not count the header itself
        total_deps[header] = len(visited) - 1
    
    # Create list of (header, dep_count, is_leaf, line_count) tuples
    # Use full paths with include directories for display
    result = []
    for header in filtered_deps:
        is_leaf = header in leaf_headers
        line_count = line_counts.get(header, 0)
        full_path = normalized_to_full_path[header]
        result.append((full_path, total_deps[header], is_leaf, line_count))
    
    # Sort by dependency count (descending) and then by header name
    return sorted(result, key=lambda x: (-x[1], x[0]))

def is_external_dependency(include_file: str) -> bool:
    """Check if a file is an external dependency."""
    # Check against exact matches
    if include_file in EXTERNAL_DEPS:
        return True
        
    # Check against path patterns
    for regex in EXTERNAL_PATH_REGEXES:
        if regex.search(include_file):
            return True
            
    return False

def main():
    # Define the directories to search
    cute_dir = "include/cute"
    cutlass_dir = "include/cutlass"
    
    # Find all header files in both directories
    all_headers = []
    for directory in [cute_dir, cutlass_dir]:
        if os.path.exists(directory):
            headers = find_all_headers(directory)
            all_headers.extend(headers)
    
    if not all_headers:
        print("No header files found in cute/ or cutlass/ directories!")
        sys.exit(1)

    # Process each header file
    visited = set()
    dependencies = defaultdict(set)  # Track dependencies for each header
    
    for header in all_headers:
        find_includes(header, visited, dependencies)

    # Analyze dependencies
    dep_analysis = analyze_dependencies(dependencies)
    
    # Calculate the maximum length of header paths for proper spacing
    max_header_length = max(len(header) for header, _, _, _ in dep_analysis)
    header_col_width = max(50, max_header_length + 2)  # Minimum 80 chars, or longer if needed
    total_width = header_col_width + 35  # Add space for other columns
    
    print("\nHeader Dependency Analysis (In-Repository Headers Only):")
    print("=" * total_width)
    print(f"{'Header':<{header_col_width}} {'In-Repo Deps':<12} {'Leaf?':<8} {'Lines':<8}")
    print("-" * total_width)
    
    total_lines = 0
    leaf_lines = 0
    leaf_count = 0
    
    for header, dep_count, is_leaf, line_count in dep_analysis:
        leaf_status = "Yes" if is_leaf else "No"
        print(f"{header:<{header_col_width}} {dep_count:<12} {leaf_status:<8} {line_count:<8}")
        total_lines += line_count
        if is_leaf:
            leaf_lines += line_count
            leaf_count += 1
    
    print("\nSummary:")
    print(f"Total headers analyzed: {len(dep_analysis)} (Total lines: {total_lines})")
    print(f"Leaf headers (no in-repo dependencies): {leaf_count} (Total lines: {leaf_lines})")
    print(f"Headers with in-repo dependencies: {len(dep_analysis) - leaf_count} (Total lines: {total_lines - leaf_lines})")
    print("\nNote: Excluding the following external dependencies:")
    print("\nExact matches:")
    for dep in sorted(EXTERNAL_DEPS):
        print(f"  - {dep}")
    print("\nPath patterns:")
    for pattern in EXTERNAL_PATH_PATTERNS:
        print(f"  - {pattern}")

if __name__ == "__main__":
    main()
