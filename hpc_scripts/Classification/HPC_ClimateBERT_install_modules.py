import subprocess, sys
print("Modules installing..")
subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "datasets", "pandas"])
print("All dependencies installed successfully\n")