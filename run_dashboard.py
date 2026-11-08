#!/usr/bin/env python3
"""Launcher for the natHacks '26 Robotic Sorting Arm Dashboard."""

import os
import sys

# Ensure current directory is on pythonpath
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.__main__ import main

if __name__ == "__main__":
    main()
