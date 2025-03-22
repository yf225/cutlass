import os

# Specify the folder path
folder = "include/"

file_line_counts = []

# Walk through the directory recursively
for root, dirs, files in os.walk(folder):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            with open(file_path, 'r', errors='ignore') as f:
                # Count the number of lines in the file
                line_count = sum(1 for _ in f)
                file_line_counts.append((file_path, line_count))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Sort files by line count in descending order
sorted_files = sorted(file_line_counts, key=lambda x: x[1], reverse=True)

# Print the ranked list
print("Ranked Files by Number of Lines:")
for rank, (file_path, count) in enumerate(sorted_files, start=1):
    print(f"{rank}. {file_path}: {count} lines")

# Total line count in all files
total_lines = sum(count for _, count in file_line_counts)
print(f"\nTotal lines in all files: {total_lines}")
