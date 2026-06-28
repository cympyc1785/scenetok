"""Re-log locally-saved multi-dataset validation videos into the ORIGINAL wandb
run (same id), on a clean `val_step` x-axis, so all validations (1..N) show up.

Why: the training was restarted (resume) which split logging across two local run
dirs sharing one wandb id; wandb _step collisions hid later vals on the dashboard.
The video files are all on disk. We resume the run and re-log them under a
`relog/` prefix against a dedicated `val_step` metric (no collision with the
existing messy _step panels; non-destructive).

Run AFTER the training process is stopped (avoids concurrent-writer conflict).
"""
import argparse
import re
from pathlib import Path

REPO = Path(".").resolve()
WBROOT = REPO / "exp/va-wan-ti2v_multi_dynamic_controlnet_scene_camera_no_lora/wandb"
# (run dir, list of val occurrences are ordered by wstep). Old run = val 1.., new appends.
RUN_DIRS = [
    "run-20260625_164402-exp_va-wan-ti2v_multi_dynamic_controlnet_scene_camera_no_lora",  # val 1-4
    "run-20260626_142428-exp_va-wan-ti2v_multi_dynamic_controlnet_scene_camera_no_lora",  # val 5-11
]
PANELS = ["standard", "unseen"]
WANDB_ID = "exp_va-wan-ti2v_multi_dynamic_controlnet_scene_camera_no_lora"
WANDB_PROJECT = "scenetok"


def group_videos(run_dir):
    """{panel: {kind: [ [8 paths], ... ]}} — Sampled & Original are at DIFFERENT
    consecutive wsteps within one val, so group by kind then by wstep; the i-th
    Sampled group pairs with the i-th Original group = the i-th validation."""
    base = WBROOT / run_dir / "files/media/videos"
    out = {}
    for panel in PANELS:
        pdir = base / panel
        if not pdir.is_dir():
            continue
        by_kind = {"Sampled": {}, "Original": {}}
        for f in sorted(pdir.glob("*.mp4")):
            m = re.match(r"(Sampled|Original) Video_(\d+)_", f.name)
            if not m:
                continue
            kind, wstep = m.group(1), int(m.group(2))
            by_kind[kind].setdefault(wstep, []).append(f)
        out[panel] = {k: [grp for _, grp in sorted(d.items())] for k, d in by_kind.items()}
    return out


def build_val_sequence():
    """Ordered list of vals across both runs: [{panel: {Sampled:[8], Original:[8]}}]."""
    vals = []
    for run_dir in RUN_DIRS:
        g = group_videos(run_dir)
        n = max((len(g.get(p, {}).get("Sampled", [])) for p in PANELS), default=0)
        for i in range(n):
            entry = {}
            for p in PANELS:
                gp = g.get(p, {})
                e = {}
                if i < len(gp.get("Sampled", [])):
                    e["Sampled"] = gp["Sampled"][i]
                if i < len(gp.get("Original", [])):
                    e["Original"] = gp["Original"][i]
                if e:
                    entry[p] = e
            vals.append(entry)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    vals = build_val_sequence()
    print(f"[relog] {len(vals)} validations found")
    for i, entry in enumerate(vals, start=1):
        desc = {p: {k: len(v) for k, v in entry[p].items()} for p in entry}
        print(f"  val {i}: {desc}")
    if args.dry_run:
        print("[relog] dry run — nothing logged.")
        return

    import wandb
    run = wandb.init(project=WANDB_PROJECT, id=WANDB_ID, resume="allow")
    wandb.define_metric("relog/val_step")
    wandb.define_metric("relog/*", step_metric="relog/val_step")
    for i, entry in enumerate(vals, start=1):
        payload = {"relog/val_step": i}
        for p in PANELS:
            if p not in entry:
                continue
            for kind in ("Sampled", "Original"):
                paths = entry[p][kind]
                if paths:
                    payload[f"relog/{p}/{kind}"] = [wandb.Video(str(x)) for x in paths]
        wandb.log(payload)
        print(f"[relog] logged val {i}: { {k: (len(v) if isinstance(v,list) else v) for k,v in payload.items()} }")
    wandb.finish()
    print("[relog] done.")


if __name__ == "__main__":
    main()
