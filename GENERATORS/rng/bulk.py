import subprocess
from pathlib import Path
from alive_progress import alive_bar

FILES = int(input("How many files: "));

# Get the directory of this script
script_dir = Path(__file__).resolve().parent

# Run the delete.py script with piped input
delete_script = script_dir / "../../DATA/delete.py"
subprocess.run(["uv", "run", str(delete_script)], text=True)

# Run generate.py 100 times with input "1024"
generate_script = script_dir / "generate.py"
with alive_bar(FILES) as bar: 
	for _ in range(FILES):
   	 subprocess.run(["uv", "run", str(generate_script)], input="1024\n", text=True)
   	 bar()

