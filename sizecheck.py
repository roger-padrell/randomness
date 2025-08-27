import os
from alive_progress import alive_bar

# Maximum file size in bytes (50 MB)
MAX_SIZE = 50 * 1024 * 1024  

def get_total_files(base_dir):
    total = 0
    for root, dirs, files in os.walk(base_dir):
        # Remove hidden directories from traversal
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        total += len(files)
    return total

def check_file_sizes(base_dir):
    noerror=True;
    total_files = get_total_files(base_dir)

    with alive_bar(total_files, title="Scanning files") as bar:
        for root, dirs, files in os.walk(base_dir):
            # Remove hidden directories from traversal
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    if size > MAX_SIZE:
                        print(f"\n⚠️ WARNING: {file_path} is {size / (1024 * 1024):.2f} MB (exceeds 50 MB)")
                        noerror=False
                except OSError as e:
                    print(f"\n❌ Could not access {file_path}: {e}")
                bar()  # advance progress bar
    if noerror == True:
        print("No file is larger than max size (50 MB)")

if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"Scanning directory: {current_dir}")
    check_file_sizes(current_dir)