from __future__ import annotations

import argparse
import json
from pathlib import Path

from conflict_graphv2_api_common import API_DATASET_ROOT, TEMPLATE_DATASET_ROOT, bootstrap_api_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the parallel API-generated conflict_graphv2 dataset root.")
    parser.add_argument("--template-root", type=Path, default=TEMPLATE_DATASET_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=API_DATASET_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = bootstrap_api_dataset(
        template_root=args.template_root,
        api_root=args.dataset_root,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
