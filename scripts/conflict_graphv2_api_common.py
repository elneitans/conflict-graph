from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_DATASET_ROOT = ROOT / "data" / "conflict_graphv2_provisional"
API_DATASET_ROOT = ROOT / "data" / "conflict_graphv2_provisional_api"

STRUCTURAL_RELATIVE_PATHS = [
    "clause_registry.csv",
    "pair_registry.csv",
    "directional_rubric_registry.csv",
    "family_sheets.jsonl",
    "manifests/pilot_manifest.json",
]

VARIANT_TYPE_BY_ID = {
    "V0": "canonical",
    "V1": "paraphrase-lexical",
    "V2": "paraphrase-structural",
    "V3": "paraphrase-pragmatic",
    "V4": "innocuous-counterfactual",
}

EXPECTED_VARIANT_IDS = {"V1", "V2", "V3", "V4"}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "by",
    "do",
    "for",
    "from",
    "he",
    "her",
    "him",
    "his",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "remains",
    "set",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "wants",
    "what",
    "who",
    "with",
    "without",
    "you",
    "your",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_api_dataset(
    template_root: Path = TEMPLATE_DATASET_ROOT,
    api_root: Path = API_DATASET_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    if not template_root.exists():
        raise FileNotFoundError(f"Template dataset root not found: {template_root}")

    copied_files: list[dict[str, str]] = []
    source_hashes: dict[str, str] = {}
    for rel_path in STRUCTURAL_RELATIVE_PATHS:
        src = template_root / rel_path
        dst = api_root / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Missing template artifact: {src}")
        ensure_dir(dst.parent)
        if overwrite or not dst.exists():
            shutil.copy2(src, dst)
        copied_files.append({"relative_path": rel_path, "source": str(src.resolve()), "target": str(dst.resolve())})
        source_hashes[rel_path] = sha256_file(src)

    ensure_dir(api_root / "api_generation" / "variant_cache")
    ensure_dir(api_root / "manifests")

    manifest = {
        "dataset_name": "conflict_graphv2_provisional_api",
        "status": "bootstrapped",
        "created_at_utc": utc_now(),
        "template_root": str(template_root.resolve()),
        "api_root": str(api_root.resolve()),
        "copied_structural_files": copied_files,
        "template_source_hashes": source_hashes,
        "generated_family_count": 0,
        "assembled_prompt_count": 0,
    }
    write_json(api_root / "manifests" / "api_generation_manifest.json", manifest)
    return manifest


def load_family_sheets(dataset_root: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(dataset_root / "family_sheets.jsonl")
    if not rows:
        raise ValueError(f"No family sheets found in {dataset_root}")
    return rows


def family_map(dataset_root: Path) -> dict[str, dict[str, Any]]:
    return {row["family_id"]: row for row in load_family_sheets(dataset_root)}


def cache_path(dataset_root: Path, family_id: str) -> Path:
    return dataset_root / "api_generation" / "variant_cache" / f"{family_id}.json"


def build_v0_prompt_row(family: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": f"{family['family_id']}_V0",
        "family_id": family["family_id"],
        "variant_id": "V0",
        "split": family["split"],
        "target_type": family["target_type"],
        "target_clauses": family["target_clauses"],
        "prompt_text": family["canonical_prompt"],
        "variant_type": VARIANT_TYPE_BY_ID["V0"],
        "determinate_status": family["determinate_status"],
        "admissible_actions": family["admissible_actions"],
        "domain": family["domain"],
        "stakes": family["stakes"],
        "role_asymmetry": family["role_asymmetry"],
    }


def build_variant_prompt_row(family: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt_id": f"{family['family_id']}_{variant['variant_id']}",
        "family_id": family["family_id"],
        "variant_id": variant["variant_id"],
        "split": family["split"],
        "target_type": family["target_type"],
        "target_clauses": family["target_clauses"],
        "prompt_text": variant["prompt_text"],
        "variant_type": variant["variant_type"],
        "determinate_status": family["determinate_status"],
        "admissible_actions": family["admissible_actions"],
        "domain": family["domain"],
        "stakes": family["stakes"],
        "role_asymmetry": family["role_asymmetry"],
    }


def tokenize_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_'-]+", text.lower())
    return [token for token in tokens if len(token) >= 4 and token not in STOPWORDS]


def protected_fact_errors(family: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    prompt_text = variant["prompt_text"].strip()
    preservation = variant.get("preservation_notes", {})
    notes_blob = json.dumps(preservation, ensure_ascii=False)
    search_space = f"{prompt_text}\n{notes_blob}".lower()
    errors: list[str] = []
    canonical_prompt = family["canonical_prompt"].strip()

    if not prompt_text:
        errors.append("prompt_text is empty")
    if prompt_text == canonical_prompt:
        errors.append("prompt_text is identical to canonical_prompt")

    for fact in family["protected_facts"]:
        keywords = tokenize_keywords(fact)
        if not keywords:
            continue
        required_overlap = min(2, len(keywords))
        overlap = sum(1 for token in set(keywords) if token in search_space)
        if overlap < required_overlap:
            errors.append(f"protected_fact_overlap_failed: {fact}")
    return errors


def validate_cached_family(family: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("family_id") != family["family_id"]:
        errors.append("family_id mismatch")
    variants = payload.get("variants")
    if not isinstance(variants, list):
        return ["variants missing or not a list"]
    if len(variants) != 4:
        errors.append(f"expected 4 generated variants, found {len(variants)}")

    by_id: dict[str, dict[str, Any]] = {}
    for variant in variants:
        variant_id = variant.get("variant_id")
        if variant_id in by_id:
            errors.append(f"duplicate variant_id {variant_id}")
            continue
        by_id[variant_id] = variant

    if set(by_id.keys()) != EXPECTED_VARIANT_IDS:
        errors.append(f"variant ids mismatch: {sorted(by_id.keys())}")

    for variant_id in sorted(by_id):
        variant = by_id[variant_id]
        if variant.get("variant_type") != VARIANT_TYPE_BY_ID[variant_id]:
            errors.append(f"{variant_id} variant_type mismatch")
        prompt_text = variant.get("prompt_text")
        if not isinstance(prompt_text, str) or not prompt_text.strip():
            errors.append(f"{variant_id} prompt_text empty")
        preservation_notes = variant.get("preservation_notes")
        if not isinstance(preservation_notes, dict):
            errors.append(f"{variant_id} preservation_notes must be an object")
        else:
            if not isinstance(preservation_notes.get("preserved_facts"), list) or not preservation_notes.get("preserved_facts"):
                errors.append(f"{variant_id} preserved_facts missing")
            if not isinstance(preservation_notes.get("changed_surface_features"), list) or not preservation_notes.get("changed_surface_features"):
                errors.append(f"{variant_id} changed_surface_features missing")
        errors.extend(f"{variant_id} {msg}" for msg in protected_fact_errors(family, variant))
    return errors


def build_review_queue_rows(dataset_root: Path) -> list[dict[str, Any]]:
    families = load_family_sheets(dataset_root)
    rows: list[dict[str, Any]] = []
    for family in families:
        payload = load_json(cache_path(dataset_root, family["family_id"]))
        variant = next(item for item in payload["variants"] if item["variant_id"] == "V1")
        rows.append(
            {
                "family_id": family["family_id"],
                "sample_variant_id": "V1",
                "sample_variant_type": variant["variant_type"],
                "canonical_prompt": family["canonical_prompt"],
                "generated_prompt": variant["prompt_text"],
                "review_status": "",
                "notes": "",
            }
        )
    return rows
