import argparse

from controller import Controller
from utils.cli import AnalyserArgs


__all__ = [ "main" ]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="R6 Analyser",
        description="A Rainbow Six Siege VOD Analyser to record live information from a game.")

    parser.add_argument("-c", "--config",
                        type=str,
                        help="JSON configuration file located in . or ./configs",
                        dest="config_path")
    parser.add_argument("-k", "--key",
                        type=str,
                        help="Software Key",
                        dest="key")
    parser.add_argument("-v", "--verbose",
                        type=int,
                        help="Determines how detailed the console output is, 0-nothing, 1-some, 2-all, 3-debug",
                        dest="verbose")

    parser.add_argument("--check-regions", 
                        action="store_true",
                        help="Does not perform data extract but saves the regions of interest as images for quality check",
                        dest="check_regions")
    parser.add_argument("--test-regions",
                        action="store_true",
                        help="Performs data extract for the instance when the program is run",
                        dest="test_regions")
    parser.add_argument("--check",
                        action="store_true",
                        help="Runs a script to check program validity and function",
                        dest="deps_check")

    parser.add_argument("--region-tool",
                        action="store_true",
                        help="Runs the Region tool instead of R6Analyser.",
                        dest="region_tool")
    parser.add_argument("-d", "--delay",
                        type=int,
                        help="Time delay between program launch and start",
                        dest="delay")
    parser.add_argument("--display",
                        type=int,
                        help="When using the `--region-tool`, Which display to capture",
                        dest="display")

    return parser


def main():
    parser = create_parser()
    parser_args = {k:v for k,v in vars(parser.parse_args()).items() if v is not None}
    args = AnalyserArgs(**parser_args)

    Controller(args).run()
