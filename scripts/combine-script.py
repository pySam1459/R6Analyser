## disclaimer: this script was mainly written by claude 3.5 sonnet
import argparse
import ast
import json
import os
import py_compile
import re
import tempfile
import toml
import warnings
from datetime import date
from pathlib import Path
from typing import Callable


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)


def extract_imports(content: str) -> tuple[list[str], list[str]]:
    import_lines = []
    non_import_lines = []
    for line in content.split('\n'):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            import_lines.append(line.strip())
        else:
            non_import_lines.append(line)
    return import_lines, non_import_lines

def clean_imports(import_lines: list[str]) -> list[str]:
    pure_imports: list[str] = []
    from_all_imports: list[str] = []
    from_imports_map: dict[str, set] = {}

    for line in import_lines:
        if line.startswith("import"):
            pure_imports.append(line)
            continue
        
        re_expr = re.match(r"from ([\.|\w]+) import ([\w|\s|,|\*]+)", line)
        if re_expr is None:
            print(line)
            continue
        
        module_name, imports = re_expr.groups()
        module_name = module_name.strip()
        if "*" in imports:
            from_all_imports.append(module_name)
        else:
            import_stuff = [item.strip() for item in imports.split(",")]

            if module_name not in from_imports_map:
                from_imports_map[module_name] = set()
            from_imports_map[module_name] |= set(import_stuff)
    
    from_imports_str = {key: "*" for key in from_all_imports} | \
                       {key: ",".join(map(str,list(value))) for key, value in from_imports_map.items()}

    pure_imports.sort()
    from_imports = [f"from {key} import {values}" for key, values in from_imports_str.items()]
    from_imports.sort()

    return pure_imports + from_imports


def is_local_import(import_line: str, file_list: list[str]) -> bool:
    for file in file_list:
        module_name = os.path.splitext(os.path.basename(file))[0]
        if re.match(rf"import (\w+\.)?{module_name}", import_line) or re.match(rf"from (\w+\.)?{module_name}", import_line):
            return True
    return False


def remove_main_block(content: str) -> str:
    # Use regex to find and remove the if __name__ == "__main__": block
    pattern = r'\nif\s+__name__\s*==\s*["\']__main__["\']\s*:\s*\n(?:.*\n)*$'
    return re.sub(pattern, '', content)


class RemoveAllTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == '__all__':
            return None
        return node

def remove_all_variable(in_content: str) -> str:
    # Parse the source code into an AST
    tree = ast.parse(in_content)

    # Transform the AST
    transformer = RemoveAllTransformer()
    modified_tree = transformer.visit(tree)

    # Generate the modified source code
    return ast.unparse(modified_tree)


class CleanupTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Name) and node.targets[0].id == '__all__':
            return None
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Str):
            return None  # Remove docstrings
        if isinstance(node.value, ast.Ellipsis):
            return node  # Preserve ellipsis
        return node

    def visit_FunctionDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            node.body = node.body[1:]  # Remove function docstrings
        return self.generic_visit(node)

    def visit_ClassDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            node.body = node.body[1:]  # Remove class docstrings
        return self.generic_visit(node)


def remove_code_elements(source_code):
    # Parse the source code into an AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"SyntaxError: {str(e)}")
        return source_code

    # Transform the AST
    cleanup = CleanupTransformer()
    try:
        modified_tree = cleanup.visit(tree)
    except Exception as e:
        print(f"Error during AST transformation: {str(e)}")
        return source_code

    # Generate the modified source code
    try:
        modified_source = ast.unparse(modified_tree)
    except Exception as e:
        print(f"Error during unparsing: {str(e)}")
        return source_code

    # Remove empty lines
    cleaned_source = '\n'.join(line for line in modified_source.split('\n') if line.strip())

    return cleaned_source


def remove_type_hints(body_lines: list[str]) -> list[str]:
    pattern = re.compile(r"(\s+def \w+\((.+)\)) -> .+:")
    out = []
    for line in body_lines:
        if (expr := re.match(pattern, line)):
            out.append(expr.group(1)+":")
        else:
            out.append(line)

    return out


def combine_python_files(file_list: list[str]):
    all_imports = set()
    combined_content = []

    for file_name in file_list:
        with open(os.path.join("src", file_name), 'r') as f_in:
            content = f_in.read()
            content = remove_main_block(content)

            import_lines, body_lines = extract_imports(content)

            # Filter out imports of files from file_list
            filtered_imports = [imp for imp in import_lines if not is_local_import(imp, file_list)]
            all_imports.update(filtered_imports)
            
            body_lines = remove_type_hints(body_lines)
            body = "\n".join(body_lines)
            body = remove_all_variable(body)
            combined_content.append(body)
    
    file_content = '\n'.join(clean_imports(list(all_imports))) + '\n\n' + '\n\n'.join(combined_content)
    file_content = remove_code_elements(file_content)
    return file_content


def compile_and_save(source: str, output_path: str) -> None:
    # Create a temporary .py file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(source)
        temp_file_name = temp_file.name

    try:
        # Compile the temporary file
        py_compile.compile(temp_file_name, cfile=output_path)
    finally:
        # Clean up the temporary file
        os.unlink(temp_file_name)


def nocompile_and_save(source: str, output_path: str) -> None:
    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(output_path, "w") as f_out:
        f_out.write(source)


def get_func_ext(no_compile: bool) -> tuple[Callable, str]:
    if no_compile:
        return (nocompile_and_save, ".py")
    else:
        return (compile_and_save, ".pyc")


def read_version():
    toml_file = Path(__file__).parent.parent / "pyproject.toml"
    if toml_file.exists() and toml_file.is_file():
        data = toml.load(toml_file)
        if "project" in data and "version" in data["project"]:
            return data["project"]["version"].replace(".", "_")

    return "0_0_0"


def write_run(dst_dir: str, output: str) -> None:
    with open(os.path.join(dst_dir, "run.py"), "w") as f_out:
        f_out.write(f"import {output};{output}.main()")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="combine-script.py", description="combines multiple python files into a single .pyc")
    parser.add_argument("-b", "--build-config", type=str, default="build_config.json", help="configuration file for build process")
    parser.add_argument("--no-compile", action="store_true", help="Doesn't compile the final .py final into bytecode")
    parser.add_argument("-O", "--output", type=str, default="./build/bundle/", dest="output", help="Output directory")

    args = parser.parse_args()

    with open(args.build_config, "r") as f_in:
        build_spec: dict = json.load(f_in)
    
    assert (combine_spec := build_spec.get("combine", None)), "No combine-spec in build-spec"
    assert (order := combine_spec.get("src-order", None)), "No src-order in combine-spec"

    combined_contents = combine_python_files(order)

    cas_func, ext = get_func_ext(args.no_compile)
    version = read_version()
    date_str = date.today().isoformat().replace("-", "_")
    module_name = f"r6a_module_{version}_{date_str}"
    dst_path = os.path.join(args.output, f"{module_name}{ext}")
    cas_func(combined_contents, dst_path)

    write_run(args.output, module_name)
