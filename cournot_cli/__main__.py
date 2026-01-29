"""
Module execution entry point.

Allows running with: python -m cournot_cli
"""

import sys
from cournot_cli.main import main

if __name__ == "__main__":
    sys.exit(main())
