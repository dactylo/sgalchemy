"""Run the supported `python -m sgalchemy` command-line entry point."""

import sys

from sgalchemy._cli import main

if __name__ == '__main__':
    sys.exit(main())
