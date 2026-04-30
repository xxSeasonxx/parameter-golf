#!/usr/bin/env python3
"""
Post-run analysis and bookkeeping for parameter-golf experiments.
Parses PyTorch and MLX training logs, archives artifacts,
generates comparison plots, and produces analysis.md.

Usage: python3 analyze.py [log_path]
"""

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
LAB_DIR = ROOT / ".lab"
RESULTS_TSV = LAB_DIR / "results.tsv"
RUN_LOG = ROOT / "run.log"
TRAIN_SCRIPT = ROOT / "train_gpt_mlx.py"

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(cmd):
    try:
        return subprocess.check_output(
            ["git"] + cmd.split(), stderr=subprocess.DEVNULL, cwd=str(ROOT)
        ).decode().strip()
    except Exception:
        return None

def get_commit_hash(short=False):
    flag = "--short" if short else ""
    return git(f"rev-parse {flag} HEAD".strip())

def get_commit_message():
    return git("log -1 --pretty=%s")


def select_log_path(explicit_path=None, root=ROOT):
    """Resolve the log to analyze: explicit path, newest runs/*.log, then run.log."""
    root = Path(root)
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            raise FileNotFoundError(f"Log path not found: {path}")
        return path

    runs_dir = root / "runs"
    run_logs = []
    if runs_dir.exists():
        run_logs = [p for p in runs_dir.glob("*.log") if p.is_file()]
    if run_logs:
        return max(run_logs, key=lambda p: (p.stat().st_mtime_ns, p.name))

    root_log = root / "run.log"
    if root_log.exists():
        return root_log

    raise FileNotFoundError(
        "No run log found. Pass a log path, create runs/*.log, or create run.log in the project root."
    )

# ---------------------------------------------------------------------------
# Parse train_gpt_mlx.py log output
# ---------------------------------------------------------------------------

def parse_log(log_path):
    """Parse the structured log output from train_gpt_mlx.py.

    Log lines follow these patterns (MLX adds tok_s: field):
      step:N/TOTAL train_loss:X.XXXX train_time:NNNms step_avg:NN.NNms tok_s:NNNN
      step:N/TOTAL val_loss:X.XXXX val_bpb:X.XXXX train_time:NNNms step_avg:NN.NNms
      final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX eval_time:NNNms
      final_int8_zlib_roundtrip_exact val_loss:X.XXXXXXXX val_bpb:X.XXXXXXXX
      stopping_early: wallclock_cap train_time:NNNms step:N/TOTAL
    Also handles PyTorch format (peak memory, serialized model lines).
    """
    if not log_path.exists():
        return None

    text = log_path.read_text()
    lines = text.splitlines()

    train_steps = []   # [{step, total, train_loss, train_time_ms, step_avg_ms}]
    val_steps = []     # [{step, total, val_loss, val_bpb, train_time_ms}]
    config_lines = []  # meaningful config lines (not source code dump)
    summary = {}

    # MLX logs dump the full source code first, then config lines like
    # "run_id:", "model_params:", "optimizer:", etc. before training starts.
    # We capture only the config lines (colon-separated key:value format).
    CONFIG_PREFIXES = (
        "run_id:", "mlx_version:", "train_loader:", "val_loader:",
        "tokenizer_path:", "model_params:", "iterations:", "optimizer:",
        "val_bpb:", "compute_dtype:", "dtypes ", "mlx_max_microbatch",
        "WARNING:", "feature_flags:", "deep_supervision:", "layer_growth:",
        "world_size:", "sdp_backends:", "attention_mode:", "tie_embeddings:",
        "seed:",
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("run_id:"):
            summary["run_id"] = line.split(":", 1)[1].strip()

        # Training step: step:N/TOTAL train_loss:X.XXXX ... (optional tok_s:)
        m = re.match(
            r"step:(\d+)/(\d+)\s+train_loss:([\d.]+)\s+train_time:(\d+)ms\s+step_avg:([\d.]+)ms(?:\s+tok_s:(\d+))?",
            line,
        )
        if m:

            entry = {
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "train_loss": float(m.group(3)),
                "train_time_ms": int(m.group(4)),
                "step_avg_ms": float(m.group(5)),
            }
            if m.group(6):
                entry["tok_s"] = int(m.group(6))
            train_steps.append(entry)
            continue

        # Validation step: step:N/TOTAL val_loss:X.XXXX val_bpb:X.XXXX ...
        m = re.match(
            r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+train_time:(\d+)ms",
            line,
        )
        if m:

            val_steps.append({
                "step": int(m.group(1)),
                "total": int(m.group(2)),
                "val_loss": float(m.group(3)),
                "val_bpb": float(m.group(4)),
                "train_time_ms": int(m.group(5)),
            })
            continue

        # Final int8 roundtrip exact (authoritative metric)
        m = re.match(
            r"final_int8_[\w.+-]+_roundtrip_exact\s+(?:eval_stride:\d+\s+)?val_loss:([\d.]+)\s+val_bpb:([\d.]+)",
            line,
        )
        if m:
            summary["final_val_loss"] = float(m.group(1))
            summary["final_val_bpb"] = float(m.group(2))
            continue

        # Final int8 roundtrip (with eval time)
        m = re.match(
            r"final_int8_[\w.+-]+_roundtrip\s+(?:eval_stride:\d+\s+)?val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+eval_time:(\d+)ms",
            line,
        )
        if m:
            summary["roundtrip_val_loss"] = float(m.group(1))
            summary["roundtrip_val_bpb"] = float(m.group(2))
            summary["roundtrip_eval_time_ms"] = int(m.group(3))
            continue

        # Final TTT LoRA result
        m = re.match(
            r"final_int8_ttt_lora\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+eval_time:(\d+)ms",
            line,
        )
        if m:
            summary["ttt_val_loss"] = float(m.group(1))
            summary["ttt_val_bpb"] = float(m.group(2))
            summary["ttt_eval_time_ms"] = int(m.group(3))
            continue

        # Peak memory
        m = re.match(r"peak memory allocated:\s*(\d+)\s*MiB", line)
        if m:
            summary["peak_memory_mib"] = int(m.group(1))
            continue

        # Submission size (int8+compression)
        m = re.match(r"Serialized model int8\+[\w.+-]+:\s*(\d+)\s*bytes", line)
        if m:
            summary["artifact_bytes"] = int(m.group(1))
            continue

        m = re.match(r"serialized_model_int8_[\w.+-]+:\s*(\d+)(?:\s*bytes)?", line)
        if m:
            summary["artifact_bytes"] = int(m.group(1))
            continue

        m = re.match(r"Total submission size int8\+[\w.+-]+:\s*(\d+)\s*bytes", line)
        if m:
            summary["total_submission_bytes"] = int(m.group(1))
            continue

        # Stopping early
        m = re.match(r"stopping_early.*train_time:(\d+)ms\s+step:(\d+)/(\d+)", line)
        if m:
            summary["stopped_early"] = True
            summary["final_train_time_ms"] = int(m.group(1))
            summary["final_step"] = int(m.group(2))
            summary["total_iterations"] = int(m.group(3))
            continue

        # Config lines (meaningful key:value lines, not source code dump)
        if any(line.startswith(p) for p in CONFIG_PREFIXES):
            config_lines.append(line)

    # Derive final training time from last train step if not from stopping_early
    if "final_train_time_ms" not in summary and train_steps:
        summary["final_train_time_ms"] = train_steps[-1]["train_time_ms"]
        summary["final_step"] = train_steps[-1]["step"]

    if train_steps:
        summary["total_iterations"] = summary.get("total_iterations", train_steps[0]["total"])

    return {
        "train_steps": train_steps,
        "val_steps": val_steps,
        "config_lines": config_lines,
        "summary": summary,
    }


def get_val_bpb(parsed):
    """Get the best val_bpb from parsed log, preferring final roundtrip metric."""
    s = parsed["summary"]
    # Priority: TTT > exact roundtrip > roundtrip > last val step
    for key in ["ttt_val_bpb", "final_val_bpb", "roundtrip_val_bpb"]:
        if key in s:
            return s[key]
    if parsed["val_steps"]:
        return parsed["val_steps"][-1]["val_bpb"]
    return None

# ---------------------------------------------------------------------------
# Results TSV management
# ---------------------------------------------------------------------------

def load_results():
    if not RESULTS_TSV.exists():
        return []
    results = []
    with open(RESULTS_TSV) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                results.append(dict(zip(header, parts)))
    return results

def append_result(commit, val_bpb, artifact_bytes, status, description):
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write("commit\tval_bpb\tartifact_bytes\tstatus\tdescription\n")
    with open(RESULTS_TSV, "a") as f:
        f.write(f"{commit}\t{val_bpb:.6f}\t{artifact_bytes}\t{status}\t{description}\n")

# ---------------------------------------------------------------------------
# Loss trajectory statistics
# ---------------------------------------------------------------------------

def compute_trajectory_stats(parsed):
    train_steps = parsed["train_steps"]
    val_steps = parsed["val_steps"]
    if not train_steps:
        return {}

    losses = [s["train_loss"] for s in train_steps]
    stats = {
        "num_train_steps": len(losses),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "min_loss_step": train_steps[losses.index(min(losses))]["step"],
        "max_loss": max(losses),
    }

    if val_steps:
        bpbs = [s["val_bpb"] for s in val_steps]
        stats["first_val_bpb"] = bpbs[0]
        stats["final_val_bpb"] = bpbs[-1]
        stats["min_val_bpb"] = min(bpbs)
        stats["min_val_bpb_step"] = val_steps[bpbs.index(min(bpbs))]["step"]

    # End-of-run slope (last 10% of steps)
    if len(losses) > 20:
        tail_start = int(len(losses) * 0.9)
        tail = losses[tail_start:]
        slope = (tail[-1] - tail[0]) / len(tail) if len(tail) > 1 else 0
        stats["end_slope"] = slope
        stats["still_improving"] = slope < -1e-4

    # Step timing
    if len(train_steps) > 5:
        steady = train_steps[5:]
        stats["avg_step_ms"] = sum(s["step_avg_ms"] for s in steady) / len(steady)

    return stats

# ---------------------------------------------------------------------------
# Archive run
# ---------------------------------------------------------------------------

def _slug(value):
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return safe.strip("-") or "run"


def archive_run(commit_short, parsed, log_path=None, train_script=None, lab_dir=None):
    lab_dir = LAB_DIR if lab_dir is None else Path(lab_dir)
    log_path = RUN_LOG if log_path is None else Path(log_path)
    train_script = TRAIN_SCRIPT if train_script is None else Path(train_script)

    run_id = parsed.get("summary", {}).get("run_id")
    parts = [_slug(commit_short)]
    if run_id:
        parts.append(_slug(run_id))
    parts.append(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir = lab_dir / "_".join(parts)
    suffix = 1
    while run_dir.exists():
        run_dir = lab_dir / ("_".join(parts) + f"_{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy log
    if log_path.exists():
        shutil.copy2(log_path, run_dir / "run.log")

    # Snapshot the training script
    if train_script.exists():
        shutil.copy2(train_script, run_dir / train_script.name)

    # Write parsed metrics as JSONL for easy programmatic access
    with open(run_dir / "metrics.jsonl", "w") as f:
        for s in parsed["train_steps"]:
            f.write(json.dumps({"type": "train", **s}) + "\n")
        for s in parsed["val_steps"]:
            f.write(json.dumps({"type": "val", **s}) + "\n")

    # Write summary
    with open(run_dir / "summary.json", "w") as f:
        json.dump(parsed["summary"], f, indent=2)

    return run_dir

# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _load_prev_history(results):
    """Load the most recent archived run's metrics for comparison."""
    for r in reversed(results):
        candidates = sorted(
            LAB_DIR.glob(f"{r['commit']}*/metrics.jsonl"),
            key=lambda p: p.stat().st_mtime_ns,
            reverse=True,
        )
        if candidates:
            path = candidates[0]
            train = []
            val = []
            with open(path) as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("type") == "train":
                        train.append(obj)
                    elif obj.get("type") == "val":
                        val.append(obj)
            return r["commit"], train, val
    return None, [], []

def generate_plots(parsed, run_dir, results):
    if not parsed["train_steps"]:
        return []

    prev_commit, prev_train, prev_val = _load_prev_history(results)
    generated = []

    def _save(fig, name):
        path = run_dir / name
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        generated.append(name)

    # --- 1. Training Loss Curve ---
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = [s["step"] for s in parsed["train_steps"]]
    losses = [s["train_loss"] for s in parsed["train_steps"]]
    ax.plot(steps, losses, label="current", linewidth=1, alpha=0.7)

    if prev_train:
        prev_steps = [s["step"] for s in prev_train]
        prev_losses = [s["train_loss"] for s in prev_train]
        ax.plot(prev_steps, prev_losses, label=f"prev ({prev_commit})",
                linewidth=1, alpha=0.5, linestyle="--")

    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss_curve.png")

    # --- 2. Validation BPB Curve ---
    if parsed["val_steps"]:
        fig, ax = plt.subplots(figsize=(10, 5))
        vsteps = [s["step"] for s in parsed["val_steps"]]
        vbpbs = [s["val_bpb"] for s in parsed["val_steps"]]
        ax.plot(vsteps, vbpbs, "o-", label="current", linewidth=1.5, markersize=3)

        if prev_val:
            pv_steps = [s["step"] for s in prev_val]
            pv_bpbs = [s["val_bpb"] for s in prev_val]
            ax.plot(pv_steps, pv_bpbs, "s--", label=f"prev ({prev_commit})",
                    linewidth=1, markersize=3, alpha=0.6)

        ax.set_xlabel("Step")
        ax.set_ylabel("val_bpb (lower is better)")
        ax.set_title("Validation BPB")
        ax.legend()
        ax.grid(True, alpha=0.3)
        _save(fig, "val_bpb_curve.png")

    # --- 3. Step Timing ---
    if len(parsed["train_steps"]) > 5:
        fig, ax = plt.subplots(figsize=(10, 4))
        steps = [s["step"] for s in parsed["train_steps"]]
        times = [s["step_avg_ms"] for s in parsed["train_steps"]]
        ax.plot(steps, times, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel("Step Avg (ms)")
        ax.set_title("Step Timing")
        ax.grid(True, alpha=0.3)
        _save(fig, "step_timing.png")

    return generated

# ---------------------------------------------------------------------------
# Generate analysis.md
# ---------------------------------------------------------------------------

def generate_analysis(parsed, traj_stats, results, run_dir, plot_files, commit_short):
    summary = parsed["summary"]
    val_bpb = get_val_bpb(parsed) or 0

    lines = []
    lines.append(f"# Run Analysis: {commit_short}")
    lines.append("")
    lines.append(f"**val_bpb: {val_bpb:.6f}**")
    lines.append("")

    # Config dump
    if parsed["config_lines"]:
        lines.append("## Configuration")
        lines.append("```")
        for cl in parsed["config_lines"][:30]:
            lines.append(f"  {cl}")
        lines.append("```")
        lines.append("")

    # Performance
    lines.append("## Performance")
    train_time_s = summary.get("final_train_time_ms", 0) / 1000
    lines.append(f"- Training time: {train_time_s:.1f}s")
    lines.append(f"- Steps completed: {summary.get('final_step', '?')}/{summary.get('total_iterations', '?')}")
    if summary.get("stopped_early"):
        lines.append(f"- Stopped early (wallclock cap)")
    if "peak_memory_mib" in summary:
        lines.append(f"- Peak memory: {summary['peak_memory_mib']} MiB")
    if "artifact_bytes" in summary:
        artifact_mb = summary["artifact_bytes"] / 1_000_000
        lines.append(f"- Artifact size: {summary['artifact_bytes']:,} bytes ({artifact_mb:.2f} MB / 16 MB limit)")
    if "total_submission_bytes" in summary:
        total_mb = summary["total_submission_bytes"] / 1_000_000
        lines.append(f"- Total submission: {summary['total_submission_bytes']:,} bytes ({total_mb:.2f} MB)")
    lines.append("")

    # Final metrics
    lines.append("## Final Metrics")
    if "roundtrip_val_bpb" in summary:
        lines.append(f"- Int8 roundtrip val_bpb: {summary['roundtrip_val_bpb']:.6f}")
    if "final_val_bpb" in summary:
        lines.append(f"- Int8 exact val_bpb: {summary['final_val_bpb']:.8f}")
    if "ttt_val_bpb" in summary:
        lines.append(f"- TTT LoRA val_bpb: {summary['ttt_val_bpb']:.6f}")
    lines.append("")

    # Loss trajectory
    if traj_stats:
        lines.append("## Loss Trajectory")
        lines.append(f"- First loss: {traj_stats.get('first_loss', '?'):.4f}")
        lines.append(f"- Final loss: {traj_stats.get('final_loss', '?'):.4f}")
        lines.append(f"- Min loss: {traj_stats.get('min_loss', '?'):.4f} (step {traj_stats.get('min_loss_step', '?')})")
        if "first_val_bpb" in traj_stats:
            lines.append(f"- First val_bpb: {traj_stats['first_val_bpb']:.4f}")
            lines.append(f"- Best val_bpb during training: {traj_stats['min_val_bpb']:.4f} (step {traj_stats['min_val_bpb_step']})")
        if "end_slope" in traj_stats:
            slope = traj_stats["end_slope"]
            improving = traj_stats.get("still_improving", False)
            lines.append(f"- End-of-run slope: {slope:.6f} ({'still improving' if improving else 'converged/plateaued'})")
        if "avg_step_ms" in traj_stats:
            lines.append(f"- Avg step time: {traj_stats['avg_step_ms']:.1f}ms")
        lines.append("")

    # Plots
    if plot_files:
        lines.append("## Plots")
        for pf in plot_files:
            lines.append(f"- `{pf}`")
        lines.append("")

    # Cross-run comparison
    if results:
        lines.append("## Cross-Run Summary")
        best = find_best_run(results)
        if best:
            best_row, best_bpb = best
            lines.append(f"- All-time best val_bpb: {best_bpb:.6f} (commit {best_row['commit']})")
            if val_bpb > 0:
                delta = val_bpb - best_bpb
                sign = "+" if delta > 0 else ""
                lines.append(f"- This run vs best: {sign}{delta:.6f} ({'worse' if delta > 0 else 'NEW BEST' if delta < 0 else 'same'})")

        recent = results[-5:]
        lines.append(f"- Recent {len(recent)} runs:")
        for r in recent:
            status = r.get("status", "?")
            bpb = r.get("val_bpb", "?")
            desc = r.get("description", "?")
            lines.append(f"  - [{status}] bpb={bpb} | {desc}")
        lines.append("")

        kept = [r for r in results if r.get("status") == "keep"]
        if kept:
            lines.append("## Improvement Lineage (kept runs)")
            for r in kept:
                lines.append(f"  - {r['commit']} bpb={r['val_bpb']} | {r.get('description', '?')}")
            lines.append("")

    # Metrics pointer
    lines.append("## Metrics")
    lines.append(f"Per-step metrics: `.lab/{run_dir.name}/metrics.jsonl`")
    lines.append("Each line is JSON with keys: `type`, `step`, `total`, `train_loss`/`val_loss`/`val_bpb`, `train_time_ms`, `step_avg_ms`")
    lines.append("")

    # Agent section
    lines.append("## Agent Investigation Notes")
    lines.append("")
    lines.append("**YOU MUST write your analysis findings below this line before moving on.**")
    lines.append("Look at the plots above. Load metrics.jsonl and compare with previous runs.")
    lines.append("What patterns do you see? What was surprising? What should you try next?")
    lines.append("")

    analysis_path = run_dir / "analysis.md"
    analysis_path.write_text("\n".join(lines))
    return analysis_path


def find_best_run(results):
    best = None
    for r in results:
        try:
            bpb = float(r["val_bpb"])
            if bpb > 0 and (best is None or bpb < best[1]):
                best = (r, bpb)
        except (ValueError, KeyError):
            continue
    return best

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("analyze.py: Post-run analysis for parameter-golf")

    explicit_log = sys.argv[1] if len(sys.argv) > 1 else None
    if len(sys.argv) > 2:
        print("ERROR: Usage: python analyze.py [log_path]", file=sys.stderr)
        sys.exit(1)

    try:
        log_path = select_log_path(explicit_log)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # Parse log
    parsed = parse_log(log_path)
    if parsed is None:
        print(f"ERROR: Could not parse {log_path}", file=sys.stderr)
        sys.exit(1)

    commit = get_commit_hash(short=True) or "unknown"
    commit_msg = get_commit_message() or "no message"
    val_bpb = get_val_bpb(parsed) or 0
    summary = parsed["summary"]

    print(f"  Log: {log_path}")
    print(f"  Commit: {commit} ({commit_msg})")
    print(f"  val_bpb: {val_bpb:.6f}")
    if "peak_memory_mib" in summary:
        print(f"  Peak memory: {summary['peak_memory_mib']} MiB")
    if "artifact_bytes" in summary:
        print(f"  Artifact: {summary['artifact_bytes']:,} bytes")

    # Trajectory stats
    traj_stats = compute_trajectory_stats(parsed)
    if traj_stats:
        print(f"  Steps: {traj_stats.get('num_train_steps', 0)}, final_loss={traj_stats.get('final_loss', '?'):.4f}")
    else:
        print("  Warning: No step-level metrics found in log")

    # Load existing results (before appending current)
    results = load_results()

    # Archive
    run_dir = archive_run(commit, parsed, log_path=log_path)
    print(f"  Archived to: {run_dir}")

    # Generate plots
    print("  Generating plots...")
    plot_files = generate_plots(parsed, run_dir, results)
    if plot_files:
        print(f"  Generated: {', '.join(plot_files)}")

    # Append to results
    artifact_bytes = summary.get("artifact_bytes", 0)
    append_result(commit, val_bpb, artifact_bytes, "pending", commit_msg)
    print(f"  Appended to results.tsv (status=pending)")

    # Generate analysis
    analysis_path = generate_analysis(parsed, traj_stats, results, run_dir, plot_files, commit)
    print(f"  Analysis: {analysis_path}")

    print("  Done.")


if __name__ == "__main__":
    main()
