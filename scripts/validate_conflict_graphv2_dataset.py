from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "conflict_graphv2_provisional"


EXPECTED_PAIR_SLOTS = {"D1", "D2", "D3", "D4", "T1", "T2"}
EXPECTED_CONTROL_SLOTS = {"S1", "S2"}
EXPECTED_ACTIONS = {"ALLOW", "REFUSE", "CLARIFY", "CONSTRAINED_HELP"}


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    clause_rows = load_csv(DATA_DIR / "clause_registry.csv")
    pair_rows = load_csv(DATA_DIR / "pair_registry.csv")
    rubric_rows = load_csv(DATA_DIR / "directional_rubric_registry.csv")
    family_rows = load_jsonl(DATA_DIR / "family_sheets.jsonl")
    prompt_rows = load_jsonl(DATA_DIR / "prompt_table.jsonl")
    manifest = json.loads((DATA_DIR / "manifests" / "dataset_manifest.json").read_text(encoding="utf-8"))

    require(len(clause_rows) == 8, f"Expected 8 clauses, found {len(clause_rows)}")
    require(len(pair_rows) == 28, f"Expected 28 pairs, found {len(pair_rows)}")
    require(len(rubric_rows) == 28, f"Expected 28 directional rubrics, found {len(rubric_rows)}")
    require(len(family_rows) == 184, f"Expected 184 families, found {len(family_rows)}")
    require(len(prompt_rows) == 920, f"Expected 920 prompts, found {len(prompt_rows)}")

    clause_ids = {row["clause_id"] for row in clause_rows}
    pair_ids = {row["pair_id"] for row in pair_rows}
    rubric_ids = {row["rubric_id"] for row in rubric_rows}
    require(len(clause_ids) == 8, "Clause IDs must be unique")
    require(len(pair_ids) == 28, "Pair IDs must be unique")
    require(rubric_ids == {f"R_{pair_id}" for pair_id in pair_ids}, "Rubric IDs must match all pair IDs")

    families_by_id = {row["family_id"]: row for row in family_rows}
    require(len(families_by_id) == 184, "Family IDs must be unique")

    prompts_by_family: dict[str, list[dict]] = defaultdict(list)
    prompt_ids = set()
    for row in prompt_rows:
        prompt_ids.add(row["prompt_id"])
        prompts_by_family[row["family_id"]].append(row)
        require(set(row["admissible_actions"]).issubset(EXPECTED_ACTIONS), f"Unexpected action in prompt row {row['prompt_id']}")
    require(len(prompt_ids) == 920, "Prompt IDs must be unique")

    pair_slot_coverage: dict[str, set[str]] = defaultdict(set)
    control_slot_coverage: dict[str, set[str]] = defaultdict(set)
    pair_domains: dict[str, dict[str, str]] = defaultdict(dict)
    control_domains: dict[str, dict[str, str]] = defaultdict(dict)
    split_counts = Counter()

    for family in family_rows:
        family_id = family["family_id"]
        target_type = family["target_type"]
        target_clauses = family["target_clauses"]
        admissible_actions = family["admissible_actions"]
        protected_facts = family["protected_facts"]
        directional_cues = family["family_specific_directional_cues"]
        split_counts[(target_type, family["split"])] += 1

        require(len(prompts_by_family[family_id]) == 5, f"{family_id} must have exactly 5 prompt variants")
        require({row["variant_id"] for row in prompts_by_family[family_id]} == {"V0", "V1", "V2", "V3", "V4"}, f"{family_id} must contain V0-V4")
        require(bool(protected_facts), f"{family_id} must have non-empty protected_facts")
        require(bool(admissible_actions), f"{family_id} must have non-empty admissible_actions")
        require(bool(family["pair_rationale"]), f"{family_id} must have a non-empty pair_rationale")

        if target_type == "pair":
            require(len(target_clauses) == 2, f"{family_id} must target exactly two clauses")
            pid = family["pair_id"]
            require(pid in pair_ids, f"{family_id} references unknown pair_id {pid}")
            pair_slot_coverage[pid].add(family["family_slot"])
            pair_domains[pid][family["family_slot"]] = family["domain"]
            require(family["materially_implicated_extra_clause"] == "none" or family["materially_implicated_extra_clause"] in clause_ids, f"{family_id} has invalid extra clause")
            require(bool(directional_cues), f"{family_id} must have non-empty directional cues")
            require(len(admissible_actions) == (1 if family["determinate_status"] == "determinate" else 2), f"{family_id} has wrong admissible action count")
        else:
            require(len(target_clauses) == 1, f"{family_id} control must target exactly one clause")
            clause_id = target_clauses[0]
            require(clause_id in clause_ids, f"{family_id} references unknown clause {clause_id}")
            control_slot_coverage[clause_id].add(family["family_slot"])
            control_domains[clause_id][family["family_slot"]] = family["domain"]
            require(family["family_specific_directional_cues"] == [], f"{family_id} control should not have directional cues")
            require(family["determinate_status"] == "determinate", f"{family_id} control must be determinate")
            require(len(admissible_actions) == 1, f"{family_id} control must have exactly one admissible action")

    require(set(pair_slot_coverage.keys()) == pair_ids, "Every pair must be represented in family sheets")
    require(set(control_slot_coverage.keys()) == clause_ids, "Every clause must have controls")
    for pid, slots in pair_slot_coverage.items():
        require(slots == EXPECTED_PAIR_SLOTS, f"{pid} pair family slots mismatch: {sorted(slots)}")
        require(pair_domains[pid]["T1"] not in {pair_domains[pid][slot] for slot in ["D1", "D2", "D3", "D4"]}, f"{pid} T1 domain must be held out from discovery")
        require(pair_domains[pid]["T2"] not in {pair_domains[pid][slot] for slot in ["D1", "D2", "D3", "D4"]}, f"{pid} T2 domain must be held out from discovery")
    for clause_id, slots in control_slot_coverage.items():
        require(slots == EXPECTED_CONTROL_SLOTS, f"{clause_id} control slots mismatch: {sorted(slots)}")
        require(control_domains[clause_id]["S1"] != control_domains[clause_id]["S2"], f"{clause_id} S2 domain must differ from S1")

    require(split_counts[("pair", "discovery")] == 112, f"Expected 112 discovery pair families, found {split_counts[('pair', 'discovery')]}")
    require(split_counts[("pair", "test")] == 56, f"Expected 56 test pair families, found {split_counts[('pair', 'test')]}")
    require(split_counts[("control", "discovery")] == 8, f"Expected 8 discovery controls, found {split_counts[('control', 'discovery')]}")
    require(split_counts[("control", "test")] == 8, f"Expected 8 test controls, found {split_counts[('control', 'test')]}")

    require(manifest["clause_count"] == 8, "Manifest clause_count mismatch")
    require(manifest["pair_count"] == 28, "Manifest pair_count mismatch")
    require(manifest["pair_family_count"] == 168, "Manifest pair_family_count mismatch")
    require(manifest["control_family_count"] == 16, "Manifest control_family_count mismatch")
    require(manifest["family_count"] == 184, "Manifest family_count mismatch")
    require(manifest["prompt_count"] == 920, "Manifest prompt_count mismatch")
    require(len(manifest["pilot_family_ids"]) == 36, "Pilot manifest should contain 36 family IDs")

    print(
        json.dumps(
            {
                "status": "ok",
                "clause_count": len(clause_rows),
                "pair_count": len(pair_rows),
                "family_count": len(family_rows),
                "prompt_count": len(prompt_rows),
                "split_counts": {
                    "pair_discovery": split_counts[("pair", "discovery")],
                    "pair_test": split_counts[("pair", "test")],
                    "control_discovery": split_counts[("control", "discovery")],
                    "control_test": split_counts[("control", "test")],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
