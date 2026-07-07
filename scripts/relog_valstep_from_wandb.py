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


def relog_one(
    api: wandb.Api,
    entity: str,
    project: str,
    src_id: str,
    dry_run: bool,
    step_mode: str,
    val_check_interval: int,
    delete_existing: bool,
    dest_suffix: str,
) -> None:
    src = api.run(f"{entity}/{project}/{src_id}")
    hist = src.history(samples=200000).sort_values("_step").reset_index(drop=True)
    val_cols = [c for c in hist.columns if is_val_col(c)]
    if not val_cols:
        print(f"[{src_id}] no val columns found — skip")
        return

    axis = "trainer/global_step" if step_mode == "global_step" else "val_step"
    # Group val metrics into rounds keyed by the ACTUAL training step at which
    # each validation ran: read the real `trainer/global_step` (0, 4999, 9999, …
    # — note it's step-1, not a synthetic i*interval) and forward-fill it so
    # every metric row in a val burst inherits its round's true global step.
    if step_mode == "global_step":
        if "trainer/global_step" not in hist.columns:
            print(f"[{src_id}] no `trainer/global_step` column — skip")
            return
        gstep = hist["trainer/global_step"].ffill()
        # {round_key(=true global step) : {metric: value}}
        rounds: dict[int, dict] = {}
        for c in val_cols:
            for idx in hist.index[hist[c].notna()]:
                g = gstep.iloc[idx]
                if g != g:  # NaN
                    continue
                rounds.setdefault(int(g), {})[c] = float(hist[c].iloc[idx])
        keys = sorted(rounds)
    else:
        # val_step mode: order rounds by appearance, assign 0,1,2,…
        ordered = {c: hist[c].dropna().tolist() for c in val_cols}
        n = max(len(v) for v in ordered.values())
        rounds = {i: {c: float(ordered[c][i]) for c in val_cols if i < len(ordered[c])} for i in range(n)}
        keys = list(range(n))

    print(f"[{src_id}] name={src.name!r} | val cols={len(val_cols)} | rounds={len(keys)} | axis={axis}")

    if dry_run:
        for k in keys:
            p = {c: round(v, 3) for c, v in rounds[k].items() if "psnr" in c}
            print(f"    {axis}={k}: {p}")
        return

    # Destination run id. `dest_suffix` picks the target: default `_relog_vs`
    # (fresh run — both Step and trainer/global_step end up 0,1,2,…) or e.g.
    # `_valfix` to APPEND val into the existing training run (then Step keeps the
    # run's high global step — wandb `_step` is monotonic — but the
    # `trainer/global_step` COLUMN is still 0,1,2,… for x-axis=trainer/global_step).
    dst_id = f"{src_id}{dest_suffix}"
    exists = False
    try:
        api.run(f"{entity}/{project}/{dst_id}")
        exists = True
    except Exception:
        exists = False
    if delete_existing and exists:
        try:
            api.run(f"{entity}/{project}/{dst_id}").delete()
            print(f"[{src_id}]   deleted existing {dst_id}")
            exists = False
        except Exception:
            pass

    init_kw = dict(
        project=project, entity=entity, id=dst_id, resume="allow", reinit=True,
        settings=wandb.Settings(init_timeout=180),
    )
    if not exists:  # only name/tag/notes a freshly-created run; keep existing run's identity
        init_kw.update(name=f"{src.name} [relog]", tags=list(src.tags) + ["relog", step_mode],
                       notes=f"val scalars from {src_id} re-logged on the {axis} axis.")
    run = wandb.init(**init_kw)

    if step_mode == "val_step":
        # Values 0,1,2,… (= validation counter (step+1)//interval). Write a
        # `trainer/global_step`=k column so x-axis=trainer/global_step shows
        # 0,1,2,…. For a fresh run also pin wandb step=k so the built-in "Step"
        # axis is 0,1,2,…; when appending to an existing run we can't rewind
        # `_step`, so omit it (Step stays the run's global step).
        for k in keys:
            row = {**rounds[k], "trainer/global_step": int(k)}
            if exists:
                run.log(row)
            else:
                run.log(row, step=int(k))
    else:
        # global_step: dedicated `trainer/global_step` axis (real 0,4999,… values)
        # via define_metric, so val plots against the true training step.
        wandb.define_metric(axis)
        for c in val_cols:
            wandb.define_metric(c, step_metric=axis)
        for k in keys:
            run.log({**rounds[k], axis: k})
    run.finish()
    print(f"[{src_id}] → re-logged {len(keys)} rounds to {dst_id} ({axis}; keys={keys})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="scenetok")
    ap.add_argument("--entity", default=None, help="defaults to API key's default entity")
    ap.add_argument("--runs", nargs="+", required=True, help="source run ids (e.g. exp_<exp_name>)")
    ap.add_argument("--step_mode", choices=["global_step", "val_step"], default="global_step",
                    help="global_step: plot vs the run's real trainer/global_step (0,4999,…). "
                         "val_step: plot vs the (step+1)//interval counter (0,1,2,…).")
    ap.add_argument("--val_check_interval", type=int, default=5000,
                    help="only used by val_step mode's counter labeling (kept for reference)")
    ap.add_argument("--delete_existing", action="store_true",
                    help="delete a prior dest run before recreating (e.g. switching axis)")
    ap.add_argument("--dest_suffix", default="_relog_vs",
                    help="dest run id = <src_id><dest_suffix>. Default fresh `_relog_vs` "
                         "(Step & trainer/global_step both 0,1,2). Use e.g. `_valfix` to "
                         "APPEND into the existing training run (Step stays high; "
                         "trainer/global_step column still 0,1,2).")
    ap.add_argument("--dry_run", action="store_true", help="print rounds, don't create runs")
    args = ap.parse_args()

    api = wandb.Api()
    entity = args.entity or api.default_entity
    print(f"entity={entity} project={args.project} mode={args.step_mode} dest_suffix={args.dest_suffix} dry_run={args.dry_run}")
    for src_id in args.runs:
        try:
            relog_one(api, entity, args.project, src_id, args.dry_run,
                      args.step_mode, args.val_check_interval, args.delete_existing, args.dest_suffix)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[{src_id}] FAILED: {e!r}")


if __name__ == "__main__":
    main()
