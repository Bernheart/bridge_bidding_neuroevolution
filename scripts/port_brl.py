"""Download + convert the brl baseline checkpoints. Same as `uv run bridgebid baselines`."""

import sys

from bridgebid.models.brl import download, main

if __name__ == "__main__":
    if "--download" in sys.argv:
        download()
    main()
