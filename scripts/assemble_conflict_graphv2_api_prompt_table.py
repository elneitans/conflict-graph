from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from conflict_graphv2_api_common import (
    API_DATASET_ROOT,
    build_review_queue_rows,
    build_v0_prompt_row,
    build_variant_prompt_row,
    cache_path,
    family_map,
    load_json,
    load_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble prompt_table.jsonl for the API-generated conflict_graphv2 dataset.")
    parser.add_argument("--dataset-root", type=Path, default=API_DATASET_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt_table_path = args.dataset_root / "prompt_table.jsonl"
    if prompt_table_path.exists() and not args.overwrite:
        raise SystemExit(f"{prompt_table_path} already exists. Pass --overwrite to replace it.")

    families = load_jsonl(args.dataset_root / "family_sheets.jsonl")
    family_lookup = family_map(args.dataset_root)
    prompt_rows: list[dict] = []
    for family in families:
        prompt_rows.append(build_v0_prompt_row(family))
        payload = load_json(cache_path(args.dataset_root, family["family_id"]))
        variants_by_id = {variant["variant_id"]: variant for variant in payload["variants"]}
        for variant_id in ["V1", "V2", "V3", "V4"]:
            prompt_rows.append(build_variant_prompt_row(family, variants_by_id[variant_id]))

    write_jsonl(prompt_table_path, prompt_rows)

    review_rows = build_review_queue_rows(args.dataset_root)
    write_csv(args.dataset_root / "api_generation" / "review_queue.csv", review_rows)

    manifest_path = args.dataset_root / "manifests" / "api_generation_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "status": "assembled",
            "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "generated_family_count": len(list((args.dataset_root / "api_generation" / "variant_cache").glob("*.json"))),
            "assembled_prompt_count": len(prompt_rows),
            "review_queue_count": len(review_rows),
        }
    )
    write_json(manifest_path, manifest)

    print(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root.resolve()),
                "prompt_table": str(prompt_table_path.resolve()),
                "prompt_count": len(prompt_rows),
                "review_queue": str((args.dataset_root / 'api_generation' / 'review_queue.csv').resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
