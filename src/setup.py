import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = ['requests', 'numpy', 'numba']

print("🔧 Installing required packages...")
for package in required_packages:
    try:
        __import__(package)
        print(f"✅ {package} already installed")
    except ImportError:
        print(f"📦 Installing {package}...")
        install_package(package)
        print(f"✅ {package} installed successfully")

print("\n🎉 Setup complete! You can now run the Wordle solver.")
