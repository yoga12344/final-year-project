# pytest conftest.py – shared fixtures for unit and integration tests
import sys
import os

# Ensure src/ is on the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
