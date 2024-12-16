#!/usr/bin/env python3
import pytest
import sys
import os
from pathlib import Path

def run_tests():
    """Run the test suite with coverage reporting."""
    # get the directory containing this script
    current_dir = Path(__file__).parent.absolute()

    # add the parent directory to Python path to find the module
    sys.path.append(str(current_dir.parent))

    # run pytest with coverage
    pytest_args = [
        '--cov=clean_data',
        '--cov-report=term-missing',
        '--cov-fail-under=80', # ensures >=80% coverage
        '-v',
        str(current_dir / 'test_clean_data.py')
    ]
    exit_code = pytest.main(pytest_args)

    return exit_code

if __name__ == "__main__":
    sys.exit(run_tests())
