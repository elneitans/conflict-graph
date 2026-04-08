# Conflict Graph v2 Provisional Dataset

This folder contains the provisional generic-spec dataset artifacts for `conflict_graphv2`.

Artifacts:
- `clause_registry.csv`: fixed 8-clause provisional node set.
- `pair_registry.csv`: all 28 unordered clause pairs plus pair metadata.
- `directional_rubric_registry.csv`: one base directional rubric per pair.
- `family_sheets.jsonl`: one family-sheet record per family.
- `prompt_table.jsonl`: one prompt row per variant.
- `manifests/dataset_manifest.json`: dataset counts and pilot manifest.
- `audits/*.csv`: audit templates for action-tag/content and directional scoring checks.
- `collections/dry_collection_template.jsonl`: collection schema for model dry runs.

Generation:
- Run `python3 scripts/generate_conflict_graphv2_dataset.py`.
- Validate with `python3 scripts/validate_conflict_graphv2_dataset.py`.

Notes:
- Action-instability remains the primary graph signal.
- Directional scores are auxiliary, auditable, and may be treated as exploratory if scorer-audit agreement is poor.
