import subprocess
import sys
import pkg_resources

# 定义所需的依赖及其版本
REQUIRED_PACKAGES = {
    "transformers": "4.45.2",
    "datasets": "3.0.1",
    "torch": "2.4.1",
    "pandas": "2.2.3",
    "scikit-learn": "1.5.2"
}

def check_and_install_packages():
    print("Checking and installing required packages...")
    
    # 检查已安装的包
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    for package, version in REQUIRED_PACKAGES.items():
        if package not in installed:
            print(f"Installing {package}=={version}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        elif pkg_resources.parse_version(installed[package]) < pkg_resources.parse_version(version):
            print(f"Upgrading {package} to version {version}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}", "--upgrade"])
        else:
            print(f"{package}=={version} is already installed.")

def main():
    try:
        check_and_install_packages()
        print("\nEnvironment setup complete! All required packages are installed.")
    except Exception as e:
        print(f"Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()