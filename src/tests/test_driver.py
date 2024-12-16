#!/usr/bin/env python3
import pytest
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', message='Covariance.*') # filtering out a single meaningless numpy warning
warnings.filterwarnings('ignore', category=UserWarning, message='FigureCanvasAgg.*') # prevents warnings from .show() calls with the Agg backend

def run_tests():
    """Run all test files in the current directory with coverage reporting."""
    # get the directory containing this script
    current_dir = Path(__file__).parent.absolute()
    parent_dir = current_dir.parent

    # add the parent directory to Python path to find the modules
    sys.path.append(str(parent_dir))

    # run pytest with coverage on all test files
    pytest_args = [
        f'--cov={parent_dir}',  # coverage for parent directory
        '--cov-report=term-missing',  # Show missing lines in terminal
        '--cov-report=html',  # Generate HTML report
        '--cov-fail-under=80',  # ensures >=80% coverage
        '-v',
        str(current_dir)  # run all tests in current directory
    ]
    exit_code = pytest.main(pytest_args)

    return exit_code

if __name__ == "__main__":
    sys.exit(run_tests())
