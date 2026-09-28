# Wound analyzer - handoff after round v3 (closed 2026-09-27)

## Production (unchanged this round)
MRC Miracle site -> Vercel project wound-analyzer-vercel (owned by the "MRC App" session).
- Code: experiments/v2/production_snapshot/predict.py (read-only copy; SHA256SUMS inside)
- Classifier: wound_model.tflite sha256 eef2f3d3fe716f075c2a5bc48357b80759e66fb07da98c2670048c065607dd68
- Out-of-scope gate: out_of_scope_gate.tflite sha256 49201744fb8a... (full hash in ebis_frozen.json)
- Rule: unknown if out_of_scope wins, gate >= 0.5 or confidence < 0.60 (a configurable operating
  threshold, not a safety standard); then a centred 70% crop retry accepted at >= 0.80.

## Backups of trained artifacts
- ~/Downloads/wound-analyzer-artifacts/v3-2026-09-27/ - 159 files, 341 MB, SHA256SUMS verified
  (all v3 checkpoints, RE3 TFLite members + manifests, data manifests, saved predictions, production
  snapshot, deployed Keras pair). README.txt explains the layout.
- ~/Downloads/wound-analyzer-artifacts/v3-2026-09-27.zip - the same, one file, 283 MB,
  sha256 2f3218dd041fc7c4866a8e73a99fe54cf0333c4e922c4fc10bc54b9ae3913cb7.
- Google Drive folder "wound-analyzer-artifacts":
  https://drive.google.com/drive/folders/1iIPvvztKoUDE3YM9fUS02F28n9sleU9s - created; the zip must be
  uploaded by hand (the Drive connector cannot carry 283 MB). Check the upload's size against the above.
- The repo's own models/ files are tracked in Git.

## Frozen EBIS candidates (EBIS_PROTOCOL.md, ebis_frozen.json - do not change)
| | Classifier | Output | Threshold (chosen on val) |
|---|---|---|---|
| P0 | deployed | 6 classes, any burn degree = burn | 0.60 |
| P1 | deployed | generic burn (3 burn probabilities summed) | 0.7637 |
| P2 | RE3 ensemble (experiments/v3/artifacts/RE3) | 6 classes | 0.60 |
| P3 | RE3 ensemble | generic burn | 0.7886 |
Primary: burn recognised, P1 - P0 paired. All with the shipped gate and the production crop retry.

## When EBIS access arrives
1. Read the VEDAs Lab reply; write what it permits into experiments/v3/ebis_permission.txt. If it does
   not permit use for this project, stop. Keep EBIS images outside the repo; never commit them.
2. Run (from the worktree, on branch claude/wound-v3):
   ```
   cd /Users/vihaa/Downloads/Wound-Analyzer/.claude/worktrees/silly-wilson-17d01a/experiments/v3
   ../../.venv/bin/python ebis_eval.py --images /path/to/EBIS/images --masks /path/to/EBIS/masks
   ```
   (add `--splits file.csv` with columns file,split to report EBIS's own splits). It verifies the frozen
   model hashes, excludes images overlapping any existing collection, and writes ebis_results.json.
   If the worktree was cleaned up: check out claude/wound-v3 and restore models from the backup.
3. Do not retune anything on the results. Report generic-burn recognition separately from
   burn-degree classification (EBIS has no degree labels) and from non-burn false alarms (EBIS has
   none; use the existing sets). An EBIS improvement alone does not show the whole analyzer improved.
4. Apply the decision rule in EBIS_PROTOCOL.md. A pass means an experimental preview with rollback -
   after a sourced general burn first-aid text exists - not an automatic deployment.

## Open decisions for Vihaan (evidence in REPORT.md)
- Crop retry: +9 correct answers vs +4 wrong names (one more 3rd-degree shown as abrasion) and more
  false labels on non-wounds. Keep or remove is a product choice.
- 0.60 threshold: a stricter single-model threshold (0.625) matched the ensemble except on the
  Lower Limb healthy feet; any change should follow measured trade-offs.
- RE3 ensemble: fewer healthy-foot false alarms on that one dataset; failed its declared rule.
