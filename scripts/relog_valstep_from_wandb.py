"""Re-log a run's validation scalars onto a clean `val_step` axis.

The training code logs val metrics with `logger.log_metrics(data, val_step)` where
`val_step = (step+1)//val_check_interval` is passed as wandb's *step*. Because
wandb's internal `_step` is monotonic and dominated by the train-loss stream
(logged at global step), those small val_steps get placed on the global-step
axis — so val metrics don't sit on a clean 0,1,2,… validation counter.

This script does NOT touch training code. It reads a source run's val-metric
history via the wandb API, groups it into validation rounds (each metric fires
once per round, in order), and re-logs into a NEW run
(`<name>_relog`, id `<src_id>_relog`) using `wandb.define_metric` so every val
metric is plotted against a dedicated `val_step` axis (= round index, matching
`(step+1)//val_check_interval`: round 0 = sanity/step0, round i = step i*interval).

Usage:
    WANDB_API_KEY=<key> python scripts/relog_valstep_from_wandb.py \
        --project scenetok \
        --runs exp_lagernvs_va-wan_dl3dv_recon_stage_a exp_scenetok_mvB1_va-wan_dl3dv_recon_baseline ...
    # --entity defaults to the API key's default entity.
"""
import argparse

import wandb


# Substrings that identify a validation-metric column (everything else = train).
VAL_KEYS = ("psnr", "ssim", "lpips", "fvd")


def is_val_col(col: str) -> bool:
    c = col.lower()
    if c.startswith("_") or c.startswith("loss") or c.startswith("grad") or c.startswith("step_tracker"):
        return False
    return any(k in c for k in VAL_KEYS)


def relog_one(api: wandb.Api, entity: str, project: str, src_id: str, dry_run: bool) -> None:
    src = api.run(f"{entity}/{project}/{src_id}")
    hist = src.history(samples=200000)
    val_cols = [c for c in hist.columns if is_val_col(c)]
    if not val_cols:
        print(f"[{src_id}] no val columns found — skip")
        return

    # Per-column ordered non-null values (each = one validation round, in order).
    ordered = {c: hist[c].dropna().tolist() for c in val_cols}
    n_rounds = max(len(v) for v in ordered.values())
    counts = {c: len(v) for c, v in ordered.items()}
    print(f"[{src_id}] name={src.name!r} | val cols={len(val_cols)} | rounds={n_rounds}")
    if len(set(counts.values())) > 1:
        print(f"[{src_id}]   ⚠️ uneven round counts across metrics (aligning by index): {counts}")

    if dry_run:
        for i in range(n_rounds):
            row = {c: round(ordered[c][i], 4) for c in val_cols if i < len(ordered[c]) and "psnr" in c}
            print(f"    val_step={i}: {row}")
        return

    dst_id = f"{src_id}_relog"
    run = wandb.init(
        project=project, entity=entity, id=dst_id, name=f"{src.name} [relog]",
        resume="allow", reinit=True, tags=list(src.tags) + ["relog", "valstep"],
        notes=f"val scalars from {src_id} re-logged onto a clean val_step axis.",
    )
    # Bind every val metric to the dedicated val_step axis.
    wandb.define_metric("val_step")
    for c in val_cols:
        wandb.define_metric(c, step_metric="val_step")

    for i in range(n_rounds):
        row = {c: ordered[c][i] for c in val_cols if i < len(ordered[c])}
        row["val_step"] = i
        run.log(row)
    run.finish()
    print(f"[{src_id}] → re-logged {n_rounds} rounds to run id {dst_id}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="scenetok")
    ap.add_argument("--entity", default=None, help="defaults to API key's default entity")
    ap.add_argument("--runs", nargs="+", required=True, help="source run ids (e.g. exp_<exp_name>)")
    ap.add_argument("--dry_run", action="store_true", help="print rounds, don't create runs")
    args = ap.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    print(f"entity={entity} project={args.project} dry_run={args.dry_run}")
    for src_id in args.runs:
        try:
            relog_one(api, entity, args.project, src_id, args.dry_run)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[{src_id}] FAILED: {e!r}")


if __name__ == "__main__":
    main()
