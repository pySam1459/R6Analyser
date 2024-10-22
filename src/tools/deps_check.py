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

    for req in req_list:
        req = req.strip()
        if req.startswith("#"):
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
