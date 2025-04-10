import subprocess, sys
print("Modules installing..")
subprocess.check_call([sys.executable, "-m", "pip", "install", "atproto"])
