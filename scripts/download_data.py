#!/usr/bin/env python
"""One-time download of the real iWildCam2020-WILDS data. Separate from the
training scripts so the (large, slow) download is an explicit, deliberate
step rather than something that silently kicks off the first time you run
training with data=iwildcam_full or data=iwildcam_subset.

    python scripts/download_data.py --root data/iwildcam

Warning: this is a multi-GB download. Verify you have disk space and are on
a connection you're fine tying up before running it.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/iwildcam", help="Where to store the dataset")
    args = parser.parse_args()

    try:
        from wilds import get_dataset
    except ImportError as e:
        raise ImportError("pip install wilds first (already in environment.yml).") from e

    print(f"Downloading iWildCam2020-WILDS to {args.root} ...")
    get_dataset(dataset="iwildcam", download=True, root_dir=args.root)
    print("Done.")


if __name__ == "__main__":
    main()
