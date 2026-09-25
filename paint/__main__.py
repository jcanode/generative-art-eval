import sys

from .cli import main

if __name__ == "__main__":  # required: spawn-based workers (macOS/Windows) re-import this module
    sys.exit(main())
