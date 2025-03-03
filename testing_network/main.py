# main.py
import time
import nest
import sys
print(f"Python interpreter: {sys.executable}")



# ✅ Only install if it's not already loaded
installed_modules = nest.GetKernelStatus().get("loaded_modules", [])
if "felixmodule" not in installed_modules:
    nest.Install('felixmodule')
else:
    print("Felix Module is already installed, skipping installation.")
from testing_function.test_runner import run_all_tests


if __name__ == "__main__":
    print("🚀 Starting FelixNet Testing...")
    run_all_tests()
    print("✅ All tests completed successfully!")
