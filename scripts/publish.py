import argparse
import json
import os
from zipfile import ZipFile, ZIP_DEFLATED

def zip_wrapper(zipf: ZipFile, file_path, arcname, v):
    if v > 0:
        print(f"Zipping {file_path}")
    zipf.write(file_path, arcname)


def publish(args: argparse.Namespace) -> None:
    is_pyc = any([file.endswith(".pyc") for file in os.listdir(args.files)])

    with open(args.build_config, "r") as f_in:
        bconfig: dict = json.load(f_in)
    
    assert bconfig.get("include", None), "build config must include a `include` field"
    assert bconfig.get("include-no-exe", None) or not is_pyc, "no-exe builds must have a `include-no-exe` field"

    with ZipFile(args.output, 'w', ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.files):            
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, args.files)
                zip_wrapper(zipf, file_path, arcname, args.verbose)
        
        includes = bconfig.get("include", [])
        if is_pyc:
            includes += bconfig.get("include-no-exe", [])

        for inc_file in includes:
            zip_wrapper(zipf, inc_file, inc_file, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="publish.py", description="Creates the final zip file containg prog files")
    parser.add_argument("-f", "--files", type=str, default="dist/R6Analyser", help="Program files to zip")
    parser.add_argument("-b", "--build-config", type=str, default="build_config.json", help="configuration file for build process")
    parser.add_argument("-O", "--output", type=str, default="dist/R6Analyser.zip", help="output dst for zip file")
    parser.add_argument("-v", "--verbose", type=int, default=0, help="verbosity of output")

    args = parser.parse_args()
    publish(args)
