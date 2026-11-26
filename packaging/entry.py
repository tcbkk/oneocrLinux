"""PyInstaller entry point (keeps package-relative imports working)."""
import sys

from oneocr_linux.cli import main

if __name__ == "__main__":
    sys.exit(main())