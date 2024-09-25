import sys
import importlib.resources
import importlib.metadata
from packaging import requirements


RED = '\033[91m'


def dep_error(prompt: str):
    print(f"❌ {prompt}")
    sys.exit()


def check_skip(req: str, skip_list: list[str]) -> bool:
    for skip in skip_list:
        if skip in req:
            return True
    return False


def check_dependencies():
    print("Checking Dependencies...")

    try:
        with open("requirements.txt", "r") as f_in:
            req_list = f_in.read().splitlines()
    except FileNotFoundError:
        print(f"Requirements file 'requirements.txt' not found")
        return

    try:
        import torch
        import torchvision
    except ImportError as e:
        dep_error("PyTorch/TorchVision are not installed!\n{e}")
        
    print("✅ PyTorch/Torchvision")

    if not torch.cuda.is_available():
        dep_error("PyTorch CUDA is not installed!\n"
                "Make sure Nvidia CUDA Toolkit is installed (visit https://developer.nvidia.com/cuda-toolkit)\n"
                "Please install PyTorch with cuda enabled (visit https://pytorch.org/)\n")

    print("✅ CUDA")

    skip_list = ["torch", "torchvision"]
    for req in req_list:
        req = req.strip()
        if check_skip(req, skip_list) or req.startswith("#"):
            continue

        try:
            req_parsed = requirements.Requirement(req)
            pkg_name = req_parsed.name
            req_version = req_parsed.specifier
            installed_version = importlib.metadata.version(pkg_name)

            if req_version.contains(installed_version):
                print(f"✅ {pkg_name} == {installed_version} {req_version}")
            else:
                dep_error(f"{pkg_name} Version conflict\nRequired {req_version}, Installed {installed_version}")
        except Exception as e:
            print(f"{req}\n{e}")
    
    print("All Dependencies are installed correctly!")


if __name__ == "__main__":
    check_dependencies()
