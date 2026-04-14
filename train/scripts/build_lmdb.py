from __future__ import annotations

import argparse
from pprint import pprint

from train.datasets.lmdb_builder import build_lmdb_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--label_file", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    pprint(build_lmdb_manifest(args.data_dir, args.label_file, args.output_dir))


if __name__ == "__main__":
    main()
