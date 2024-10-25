# run_tests.py
import sys
import os
import unittest

# Add the src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Run unittest discover
unittest.main(module=None, argv=['first-arg-is-ignored', 'discover', '-s', 'tests', '-p', '*_test.py'])
