from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from conflict_graphv2_api_common import (
    API_DATASET_ROOT,
    EXPECTED_VARIANT_IDS,
    VARIANT_TYPE_BY_ID,
    build_v0_prompt_row,
    cache_path,
    family_map,
    load_json,
    load_jsonl,
    sha256_file,
    validate_cached_family,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated API variants for the parallel conflict_graphv2 dataset root.")
    parser.add_argument("--dataset-root", type=Path, default=API_DATASET_ROOT)
    parser.add_argument("--allow-partial", action="store_true", help="Allow missing variant cache files for smoke-test/bootstrap validation.")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_prompt_table(dataset_root: Path, families: dict[str, dict]) -> dict[str, int]:
    prompt_table_path = dataset_root / "prompt_table.jsonl"
    if not prompt_table_path.exists():
        return {"prompt_count": 0}

    prompt_rows = load_jsonl(prompt_table_path)
    prompts_by_family: dict[str, list[dict]] = defaultdict(list)
    for row in prompt_rows:
        prompts_by_family[row["family_id"]].append(row)

    require(len(prompt_rows) == len(families) * 5, f"Expected {len(families) * 5} prompt rows, found {len(prompt_rows)}")
    for family_id, family in families.items():
        family_rows = prompts_by_family[family_id]
        require(len(family_rows) == 5, f"{family_id} must have 5 prompt rows in prompt_table")
        variant_ids = {row["variant_id"] for row in family_rows}
        require(variant_ids == {"V0", "V1", "V2", "V3", "V4"}, f"{family_id} variant ids mismatch in prompt_table")
        v0_row = next(row for row in family_rows if row["variant_id"] == "V0")
        require(v0_row == build_v0_prompt_row(family), f"{family_id} V0 row mismatch in prompt_table")
        for variant_id in EXPECTED_VARIANT_IDS:
            row = next(item for item in family_rows if item["variant_id"] == variant_id)
            require(row["variant_type"] == VARIANT_TYPE_BY_ID[variant_id], f"{family_id} {variant_id} variant_type mismatch in prompt_table")
    return {"prompt_count": len(prompt_rows)}


def main() -> None:
    args = parse_args()
    families = family_map(args.dataset_root)
    manifest_path = args.dataset_root / "manifests" / "api_generation_manifest.json"
    require(manifest_path.exists(), f"Missing API generation manifest: {manifest_path}")
    manifest = load_json(manifest_path)

    template_root = Path(manifest["template_root"])
    template_hash_errors: list[str] = []
    for rel_path, recorded_hash in manifest.get("template_source_hashes", {}).items():
        current_path = template_root / rel_path
        if not current_path.exists():
            template_hash_errors.append(f"missing template source file: {rel_path}")
            continue
        current_hash = sha256_file(current_path)
        if current_hash != recorded_hash:
            template_hash_errors.append(f"template source hash changed: {rel_path}")
    require(not template_hash_errors, json.dumps({"template_hash_errors": template_hash_errors}, indent=2))

    generated_count = 0
    missing_count = 0
    invalid_count = 0
    errors_by_family: dict[str, list[str]] = {}

    for family_id, family in families.items():
        current_cache = cache_path(args.dataset_root, family_id)
        if not current_cache.exists():
            missing_count += 1
            if not args.allow_partial:
                errors_by_family[family_id] = ["missing cache file"]
            continue

        payload = load_json(current_cache)
        errors = validate_cached_family(family, payload)
        if errors:
            invalid_count += 1
            errors_by_family[family_id] = errors
        else:
            generated_count += 1

    prompt_stats = validate_prompt_table(args.dataset_root, families)

    if errors_by_family:
        preview = {family_id: messages[:3] for family_id, messages in list(errors_by_family.items())[:10]}
        raise AssertionError(json.dumps({"error_count": len(errors_by_family), "examples": preview}, indent=2))

    if not args.allow_partial:
        require(generated_count == len(families), f"Expected cache for all {len(families)} families, found {generated_count}")

    print(
        json.dumps(
            {
                "status": "ok",
                "dataset_root": str(args.dataset_root.resolve()),
                "family_count": len(families),
                "generated_cache_count": generated_count,
                "missing_cache_count": missing_count,
                "invalid_cache_count": invalid_count,
                "prompt_count": prompt_stats["prompt_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
