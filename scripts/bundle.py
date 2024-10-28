import argparse
import json
import os
import shutil
from sys import exit
from pathlib import Path
from zipfile import ZipFile
from typing import Any


DEFAULT_EXE          = "R6Analyser.exe"

DEFAULT_BUILD_CONFIG = "build_config.json"
DEFAULT_BUNDLE       = Path("build/bundle")
DEFAULT_DIST         = Path("dist")
DEFAULT_ZIP          = Path("dist/R6Analyser.zip")


def load_build_config(config: Path) -> dict[str, Any]:
    try:
        with open(args.config, "r") as f_in:
            return json.load(f_in)
    except Exception as e:
        print(f"Error, could not load build config {config}\n{e}")
        exit()


def bundle_distexe(bconfig: dict[str, Any], bundle: Path, dist: Path) -> None:
    if not dist.exists():
        raise ValueError("dist directory does not exist")
    
    exes = [file for file in os.listdir(dist) if file.endswith(".exe")]
    if len(exes) == 0:
        raise ValueError("dist/*.exe was not found")
    elif len(exes) > 1:
        raise ValueError("# dist/*.exe > 1  found. Cannot determine correct .exe to zip")
    
    print(f"Bundling {dist / exes[0]}")
    shutil.copyfile(dist / exes[0], bundle / bconfig.get("exe", DEFAULT_EXE))


def bundle_includes(bconfig: dict[str, Any], bundle: Path) -> None:
    for file in bconfig.get("include", []):
        print(f"Bundling {file}")
        file = Path(file)
        if not file.exists():
            raise ValueError(f"bundle includes: {file} does not exist")
        if file.is_dir():
            shutil.copytree(file, bundle / file)
        else:
            out = bundle / file
            if not out.parent.exists():
                os.makedirs(out.parent)
            shutil.copy(file, out)


def zip_directory(directory_path: Path, output_zip_path: Path):
    print(f"Zipping Bundle to {output_zip_path}")
    with ZipFile(output_zip_path, 'w') as zipf:
        # Recursively walk through the directory
        for foldername, _, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, directory_path))


def bundle(args: argparse.Namespace) -> None:
    bundle: Path = args.bundle
    dist:   Path = args.dist
    if bundle.exists():
        print("Info: Cleaning bundle tree")
        shutil.rmtree(bundle)
    os.makedirs(bundle, exist_ok=True)

    bconfig = load_build_config(args.config)
    
    bundle_distexe(bconfig, bundle, dist)
    bundle_includes(bconfig, bundle)

    zip_directory(bundle, args.output)
    print("Build Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="publish.py",
                                     description="Creates the final zip file containg prog files")
    parser.add_argument("-c", "--config",
                        type=Path,
                        default=DEFAULT_BUILD_CONFIG,
                        help="configuration file for build")
    parser.add_argument("-b", "--bundle",
                        type=Path,
                        default=DEFAULT_BUNDLE,
                        help="path to build bundle")
    parser.add_argument("-d", "--dist",
                        type=Path,
                        default=DEFAULT_DIST,
                        help="path to dist")
    parser.add_argument("-O", "--output",
                        type=Path,
                        default=DEFAULT_ZIP,
                        help="output dst for zip file")

    args = parser.parse_args()
    bundle(args)
