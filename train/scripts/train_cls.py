from __future__ import annotations

import argparse
from pprint import pprint

from train.trainers.cls_trainer import ClsTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    trainer = ClsTrainer(args.config, args.data_dir)
    pprint(
        {
            "config": args.config,
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            "resume": args.resume,
            "debug_batch": trainer.build_debug_batch(),
        }
    )


if __name__ == "__main__":
    main()
