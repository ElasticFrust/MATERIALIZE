#!/usr/bin/env python3
"""Renewable training runner with automatic monitoring and restart.

Starts GNN training in renewable mode (fresh data every chunk), then waits
for it to complete before starting CVAE training. Checks process health every
20 minutes and auto-restarts with --resume if the process dies.

Usage:
    python run_renewable_training.py                  # full run (GNN then CVAE)
    python run_renewable_training.py --gnn_only       # GNN only
    python run_renewable_training.py --cvae_only      # CVAE only (needs GNN ckpt)
    python run_renewable_training.py --check_interval 1200  # 20-min default
"""

import argparse
import subprocess
import sys
import time
import os
import signal
from pathlib import Path
from datetime import datetime

WORKDIR = Path(__file__).resolve().parent / "Phase 4"
CHECKPOINTS_DIR = WORKDIR / "checkpoints"
GNN_LOG = Path(__file__).resolve().parent / "gnn_renewable.log"
CVAE_LOG = Path(__file__).resolve().parent / "cvae_renewable.log"
MONITOR_LOG = Path(__file__).resolve().parent / "monitor_renewable.log"

# val/test still loaded from disk for stable evaluation
# Set to None if no pre-saved val set exists yet; training will skip test eval.
DATA_DIR = str(WORKDIR / "processed")


def log(msg, also_print=True):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(MONITOR_LOG, "a") as f:
        f.write(line + "\n")
    if also_print:
        print(line, flush=True)


def gnn_checkpoint_status():
    """Return epoch / chunk_idx from latest GNN checkpoint, or None."""
    ckpt_path = CHECKPOINTS_DIR / "latest_checkpoint.pt"
    if not ckpt_path.exists():
        return None
    try:
        import torch
        ckpt = torch.load(ckpt_path, weights_only=False)
        return {
            "epoch": ckpt.get("epoch"),
            "chunk_idx": ckpt.get("chunk_idx"),
            "best_val_mse": ckpt.get("best_val_mse"),
        }
    except Exception as e:
        return {"error": str(e)}


def build_gnn_cmd(resume=False, **kwargs):
    cmd = [
        sys.executable, "-u", "-m", "gnn.train",
        "--data_dir", DATA_DIR,
        "--output_dir", str(CHECKPOINTS_DIR),
        "--epochs", str(kwargs.get("epochs", 300)),
        "--batch_size", str(kwargs.get("batch_size", 32)),
        "--patience", str(kwargs.get("patience", 30)),
        "--save_every", "1",
        "--renewable",
        "--renewable_n_chunks", str(kwargs.get("renewable_n_chunks", 40)),
        "--renewable_chunk_size", str(kwargs.get("renewable_chunk_size", 2000)),
        "--renewable_n_workers", str(kwargs.get("renewable_n_workers", 4)),
    ]
    if resume:
        cmd.append("--resume")
    return cmd


def build_cvae_cmd(resume=False, **kwargs):
    gnn_ckpt = str(CHECKPOINTS_DIR / "best_model.pt")
    cmd = [
        sys.executable, "-u", "-m", "cvae.train",
        "--data_dir", DATA_DIR,
        "--gnn_checkpoint", gnn_ckpt,
        "--output_dir", str(CHECKPOINTS_DIR),
        "--epochs", str(kwargs.get("epochs", 300)),
        "--batch_size", str(kwargs.get("batch_size", 32)),
        "--patience", str(kwargs.get("patience", 40)),
        "--renewable",
        "--renewable_n_chunks", str(kwargs.get("renewable_n_chunks", 40)),
        "--renewable_chunk_size", str(kwargs.get("renewable_chunk_size", 2000)),
        "--renewable_n_workers", str(kwargs.get("renewable_n_workers", 4)),
    ]
    return cmd


def run_with_monitor(name, cmd, log_path, check_interval, env=None):
    """Run cmd, monitoring every check_interval seconds and auto-restarting.

    On restart the first extra arg '--resume' is injected if not already present.
    Returns when the process exits with code 0, or raises RuntimeError if it
    fails MAX_RESTARTS times in a row without making progress.
    """
    MAX_RESTARTS = 100  # effectively unlimited restarts
    restarts = 0
    run_cmd = list(cmd)

    while restarts <= MAX_RESTARTS:
        log(f"[{name}] Starting (attempt {restarts+1}): {' '.join(run_cmd)}")

        with open(log_path, "a") as logf:
            proc = subprocess.Popen(
                run_cmd,
                cwd=str(WORKDIR),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env or os.environ.copy(),
            )

        log(f"[{name}] PID={proc.pid}")
        start_time = time.time()

        while True:
            try:
                proc.wait(timeout=check_interval)
            except subprocess.TimeoutExpired:
                # Still running — log status
                elapsed = (time.time() - start_time) / 60
                tail = _tail(log_path, 3)
                ckpt = gnn_checkpoint_status() if name == "GNN" else {}
                log(f"[{name}] Alive (PID={proc.pid}, {elapsed:.0f}min). "
                    f"Ckpt: {ckpt}. Last log: {tail}")
                continue

            # Process exited
            rc = proc.returncode
            elapsed = (time.time() - start_time) / 60
            log(f"[{name}] Exited with code {rc} after {elapsed:.0f} min")

            if rc == 0:
                log(f"[{name}] Completed successfully.")
                return

            # Non-zero exit: restart with --resume
            log(f"[{name}] Failed (rc={rc}). Restarting with --resume ...")
            if "--resume" not in run_cmd:
                run_cmd.append("--resume")
            restarts += 1
            time.sleep(5)  # brief pause before restart
            break  # break inner while, loop outer

    raise RuntimeError(f"[{name}] Exceeded {MAX_RESTARTS} restarts. Giving up.")


def _tail(path, n=3):
    """Return last n non-empty lines from a file as a single string."""
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
        non_empty = [l for l in lines if l.strip()]
        return " | ".join(non_empty[-n:])
    except Exception:
        return "(unreadable)"


def main():
    parser = argparse.ArgumentParser(description="Renewable training runner")
    parser.add_argument("--gnn_only", action="store_true")
    parser.add_argument("--cvae_only", action="store_true")
    parser.add_argument("--check_interval", type=int, default=1200,
                        help="Health-check interval in seconds (default 1200 = 20 min)")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--renewable_n_chunks", type=int, default=40)
    parser.add_argument("--renewable_chunk_size", type=int, default=2000)
    parser.add_argument("--renewable_n_workers", type=int, default=4)
    args = parser.parse_args()

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("Renewable training monitor started")
    log(f"  check_interval = {args.check_interval}s ({args.check_interval/60:.0f} min)")
    log(f"  renewable_n_chunks = {args.renewable_n_chunks}")
    log(f"  renewable_chunk_size = {args.renewable_chunk_size}")
    log(f"  renewable_n_workers = {args.renewable_n_workers}")
    log("=" * 60)

    kw = vars(args)

    # ── GNN ──────────────────────────────────────────────────────────────────
    if not args.cvae_only:
        gnn_cmd = build_gnn_cmd(**kw)
        try:
            run_with_monitor("GNN", gnn_cmd, GNN_LOG, args.check_interval)
        except RuntimeError as e:
            log(f"FATAL: {e}")
            sys.exit(1)

    # ── CVAE ─────────────────────────────────────────────────────────────────
    if not args.gnn_only:
        gnn_ckpt = CHECKPOINTS_DIR / "best_model.pt"
        if not gnn_ckpt.exists():
            log("WARNING: GNN best_model.pt not found — CVAE physics loss disabled")

        cvae_cmd = build_cvae_cmd(**kw)
        try:
            run_with_monitor("CVAE", cvae_cmd, CVAE_LOG, args.check_interval)
        except RuntimeError as e:
            log(f"FATAL: {e}")
            sys.exit(1)

    log("All training complete.")


if __name__ == "__main__":
    main()
