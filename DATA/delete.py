import os
from alive_progress import alive_bar

def clean_selected_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__));
    print(script_dir)
    
    # List directories in the current directory
    dirs = [d for d in os.listdir(script_dir) if os.path.isdir(os.path.join(script_dir, d))]

    if not dirs:
        print("No directories found in {}.", script_dir)
        return

    # Show directories with index
    print("Available directories:")
    for i, d in enumerate(dirs, start=1):
        print(f"{i}. {d}")

    # Get user choice
    while True:
        try:
            choice = int(input("Select a directory by number: "))
            if 1 <= choice <= len(dirs):
                selected_dir = os.path.join(script_dir, dirs[choice - 1])
                break
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Confirm selection
    confirm = input(f"Are you sure you want to delete all files inside '{selected_dir}'? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation canceled.")
        return

    # Get list of files inside the selected directory
    files = [os.path.join(selected_dir, f) for f in os.listdir(selected_dir) if os.path.isfile(os.path.join(selected_dir, f))]

    # Delete files with progress bar
    with alive_bar(len(files), title=f"Cleaning '{selected_dir}'") as bar:
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error deleting {f}: {e}")
            bar()

    print(f"All files inside '{selected_dir}' have been removed.")

clean_selected_directory()
