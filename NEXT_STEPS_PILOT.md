# Next Steps Pilot

## Scope frozen for this pass

- Run one fixed open base model: `allenai/Olmo-3-1125-32B`.
- Use one fixed conditioning setup only: a single system-style instruction scaffold built from the frozen 8-clause registry.
- Run the full pilot only: all families listed in `data/conflict_graphv2_provisional/manifests/pilot_manifest.json`.
- Collect raw model responses plus one final self-reported action tag from `{ALLOW, REFUSE, CLARIFY, CONSTRAINED_HELP}`.
- Use the OLMo model card's suggested 8-bit loading path as the default collection configuration.
- Stop at collection. Do not compute instability metrics, graph edges, directional scores, or refinement updates in this pass.

## What will be run now

1. Validate that the frozen dataset artifacts already on disk are present and well-formed.
2. Load the pilot family list from `data/conflict_graphv2_provisional/manifests/pilot_manifest.json`.
3. Filter `data/conflict_graphv2_provisional/prompt_table.jsonl` down to the full pilot prompt set.
4. Build one fixed instruction scaffold from `data/conflict_graphv2_provisional/clause_registry.csv`.
5. Run greedy decoding with Hugging Face Transformers on `allenai/Olmo-3-1125-32B`.
6. Save one append-only JSONL collection file with prompt metadata, rendered prompt, raw response, extracted action tag, parse status, and run metadata.
7. Save one lightweight run manifest with counts and parse statistics.

## What is deferred

- Action-instability computation.
- Directionality scoring.
- Graph estimation.
- Any intervention or refinement loop.
- Any judge model or rubric scorer.
- Any extra training stage or model adaptation beyond the fixed prompt scaffold.

## Files added in this pass

- `scripts/collect_olmo_responses.py`
  - Pilot collection script.
  - Reads the frozen dataset.
  - Runs one fixed OLMo-family base model with deterministic decoding.
  - Parses the final `ACTION_TAG:` line.
  - Writes machine-readable JSONL outputs and a run manifest.
- `.requirements.txt`
  - Minimal Python dependencies for the collection job.
  - Excludes ROCm-specific `torch` pinning on purpose; the Slurm job uses `--system-site-packages` so the cluster's ROCm PyTorch can be reused if present.
- `scripts/run_collect_olmo_pilot.slurm.sh`
  - Slurm batch wrapper for the cluster run.
  - Creates `logs/`, activates the existing `antuco_torch310` conda environment by default, optionally installs `.requirements.txt` if requested, checks that PyTorch can see the GPU, and launches the collection script.

## Expected first end-to-end pilot outputs

- One response file under `data/conflict_graphv2_provisional/collections/runs/`:
  - JSONL, one row per pilot prompt variant.
  - Expected size for the first full pilot: `36 families x 5 variants = 180 rows`.
- One run manifest under the same run directory:
  - Model id.
  - Dataset and pilot manifest paths.
  - Generation settings.
  - Counts for processed prompts, parsed action tags, parse failures, and skipped rows on resume.

## Operational notes for `antuco` / MI210

- The batch script uses plain Transformers inference with `attn_implementation=eager` for robustness on ROCm.
- The batch script follows the previously working cluster pattern: `cd "${SLURM_SUBMIT_DIR}"`, source conda, and activate `antuco_torch310` by default.
- The batch script now also mirrors the ROCm visibility settings from the known working job: `unset CUDA_VISIBLE_DEVICES`, then set `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES`.
- The collection default is `allenai/Olmo-3-1125-32B` with `--quantization 8bit`, `--dtype float16`, and `--device-map auto`.
- The model card recommends the 8-bit path via bitsandbytes, but AMD ROCm support for bitsandbytes remains environment-sensitive. The job now fails loudly if bitsandbytes is unavailable instead of silently falling back.
- If you want to refresh Python packages inside that environment, submit with `INSTALL_REQUIREMENTS=1`.
- If your conda installation lives elsewhere, override `CONDA_SH` and/or `CONDA_ENV_NAME` at submit time.
- Submit the job from the repo root so the relative `logs/` and `data/conflict_graphv2_provisional/...` paths are writable and resolvable.
- If `torch.cuda.is_available()` is false in the job, the run should fail immediately rather than silently falling back to CPU.

## Cluster launch command

```bash
mkdir -p logs
sbatch scripts/run_collect_olmo_pilot.slurm.sh
```

Optional overrides:

```bash
sbatch --export=ALL,INSTALL_REQUIREMENTS=1 scripts/run_collect_olmo_pilot.slurm.sh
sbatch --export=ALL,CONDA_ENV_NAME=antuco_torch310 scripts/run_collect_olmo_pilot.slurm.sh
```
