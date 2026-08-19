#!/usr/bin/env python
"""Record predicted action chunks vs what the robot actually executed.

Answers "are we following the action chunk?" — chunk-boundary jumps, queue
starvation, stale merges, and command-vs-measured tracking, all on one
wall-clock axis.

Enable by setting an output path before the rollout (no config change, and
a complete no-op when unset):

    LEROBOT_CHUNK_TRACE=/tmp/trace.json just rollout <task> ...

Then plot:

    python -m lerobot.rollout.chunk_trace /tmp/trace.json [--dims 0,1,2] [--out plot.png]

Threading: `record_chunk` is called from the RTC inference thread and
`record_exec` from the main control thread. Each appends to its OWN list
(list.append is atomic under the GIL), so there is no lock and no chance of
perturbing control-loop timing; the two streams are joined offline by
timestamp and by the queue's (chunk_seq, plan_row) provenance.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_tracer: "ChunkTracer | None" = None
_initialized = False


class ChunkTracer:
    def __init__(self, path: str):
        self.path = path
        self.t0 = time.perf_counter()
        self.chunks: list[dict] = []      # appended by the inference thread
        self.execs: list[dict] = []       # appended by the control thread
        self.meta: dict[str, Any] = {}

    # -- inference side -------------------------------------------------
    def record_chunk(self, *, seq: int, birth_t: float, actions, delay_ticks: int,
                     idx_before: int, policy_actions=None) -> None:
        """One predicted chunk. `actions` is the robot-space chunk that was
        merged into the queue (T, action_dim); `policy_actions` the optional
        pre-adapter policy-space chunk (e.g. 9D EE pose)."""
        try:
            rec = {
                "seq": int(seq),
                "birth_t": float(birth_t) - self.t0,
                "merge_t": time.perf_counter() - self.t0,
                "delay_ticks": int(delay_ticks),
                "idx_before": int(idx_before),
                "actions": actions.detach().cpu().tolist(),
            }
            if policy_actions is not None:
                rec["policy_actions"] = policy_actions.detach().cpu().tolist()
            self.chunks.append(rec)
        except Exception:                  # never break a rollout to log it
            logger.exception("chunk_trace: record_chunk failed")

    # -- execution side -------------------------------------------------
    def record_exec(self, *, action: dict, meas: dict | None, info: tuple | None,
                    new_waypoint: bool) -> None:
        """One command actually sent to the robot (post-interpolation).
        `info` is ActionQueue.last_info = (chunk_seq, plan_row, birth_t)."""
        try:
            rec = {
                "t": time.perf_counter() - self.t0,
                "action": action,
                "new_waypoint": bool(new_waypoint),
            }
            if meas:
                rec["meas"] = meas
            if info is not None:
                rec["chunk_seq"] = int(info[0])
                rec["plan_row"] = int(info[1])
            self.execs.append(rec)
        except Exception:
            logger.exception("chunk_trace: record_exec failed")

    def record_starved(self) -> None:
        """A control tick where no action was available (empty queue)."""
        try:
            self.execs.append({"t": time.perf_counter() - self.t0, "starved": True})
        except Exception:
            pass

    def dump(self) -> None:
        try:
            with open(self.path, "w") as f:
                json.dump({"meta": self.meta, "chunks": self.chunks,
                           "exec": self.execs}, f)
            logger.info("chunk_trace: wrote %d chunks / %d exec rows -> %s",
                        len(self.chunks), len(self.execs), self.path)
        except Exception:
            logger.exception("chunk_trace: dump failed")


def get_tracer() -> ChunkTracer | None:
    """The active tracer, or None when LEROBOT_CHUNK_TRACE is unset."""
    global _tracer, _initialized
    if not _initialized:
        _initialized = True
        path = os.environ.get("LEROBOT_CHUNK_TRACE")
        if path:
            _tracer = ChunkTracer(path)
            logger.info("chunk_trace: ENABLED -> %s", path)
    return _tracer


# ---------------------------------------------------------------- plotting


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot(path: str, dims: list[int] | None = None, out: str | None = None,
         key_filter: str | None = None, tmin: float | None = None,
         tmax: float | None = None) -> None:
    import matplotlib
    if out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tr = _load(path)
    chunks, execs = tr["chunks"], tr["exec"]
    if not chunks:
        raise SystemExit(f"{path}: no chunks recorded")
    keys = tr.get("meta", {}).get("action_keys") or []
    fps = float(tr.get("meta", {}).get("fps") or 30.0)

    sent = [e for e in execs if "action" in e]
    if not sent:
        raise SystemExit(f"{path}: no executed actions recorded")
    if not keys:
        keys = list(sent[0]["action"].keys())
    sel = [i for i in range(len(keys))
           if (key_filter is None or key_filter in keys[i])]
    if dims:
        sel = [i for i in dims if i < len(keys)]
    sel = sel[:8]

    te = np.array([e["t"] for e in sent])
    A = np.array([[e["action"].get(k, np.nan) for k in keys] for e in sent])
    M = None
    if any("meas" in e for e in sent):
        M = np.array([[e.get("meas", {}).get(k, np.nan) for k in keys] for e in sent])

    # Gaps: gaps between recorded commands (episode resets, which retreat the
    # arm through an un-instrumented path). Break the plotted line and exclude
    # them from continuity stats — otherwise they read as huge "jumps".
    GAP_S = 0.5
    gap_at = np.where(np.diff(te) > GAP_S)[0]
    Ap = A.copy()
    Ap[gap_at] = np.nan                       # NaN breaks the line segment
    Mp = None if M is None else M.copy()
    if Mp is not None:
        Mp[gap_at] = np.nan

    n = len(sel)
    fig, axes = plt.subplots(n + 1, 1, figsize=(15, 2.1 * (n + 1)), sharex=True)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("tab20")

    for row, d in enumerate(sel):
        ax = axes[row]
        # every predicted chunk, drawn forward from its birth time
        for ci, c in enumerate(chunks):
            arr = np.asarray(c["actions"], dtype=float)
            if arr.ndim != 2 or d >= arr.shape[1]:
                continue
            t = c["birth_t"] + np.arange(arr.shape[0]) / fps
            ax.plot(t, arr[:, d], color=cmap(ci % 20), alpha=0.55, lw=1.0,
                    zorder=1)
            ax.plot(t[0], arr[0, d], ".", color=cmap(ci % 20), ms=4, zorder=2)
        ax.plot(te, Ap[:, d], "k-", lw=1.8, label="executed (sent)", zorder=3)
        if Mp is not None:
            ax.plot(te, Mp[:, d], "--", color="0.45", lw=1.2, label="measured",
                    zorder=2)
        for gi in gap_at:                     # reset / no-command windows
            ax.axvspan(te[gi], te[gi + 1], color="0.85", alpha=0.55, zorder=0)
        for e in execs:                                  # starvation marks
            if e.get("starved"):
                ax.axvline(e["t"], color="red", alpha=0.25, lw=0.6, zorder=0)
        ax.set_ylabel(keys[d] if d < len(keys) else f"dim{d}", fontsize=8)
        ax.grid(alpha=0.25)
        if row == 0:
            ax.legend(loc="upper right", fontsize=8)
            ax.set_title(
                f"{path}   thin colored = predicted chunks ({len(chunks)}), "
                f"black = executed, red = queue starved", fontsize=10)

    # bottom panel: which chunk each executed command came from + queue lag
    ax = axes[-1]
    seqs = np.array([e.get("chunk_seq", np.nan) for e in sent], dtype=float)
    rows = np.array([e.get("plan_row", np.nan) for e in sent], dtype=float)
    ax.plot(te, seqs, "-", color="tab:blue", lw=1.2, label="source chunk seq")
    ax2 = ax.twinx()
    ax2.plot(te, rows, ".", color="tab:orange", ms=2.5, label="plan row consumed")
    ax2.set_ylabel("plan row", color="tab:orange", fontsize=8)
    ax.set_ylabel("chunk seq", color="tab:blue", fontsize=8)
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.25)

    if tmin is not None or tmax is not None:
        axes[0].set_xlim(tmin if tmin is not None else te[0],
                         tmax if tmax is not None else te[-1])
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=120)
        print(f"wrote {out}")
    else:
        plt.show()

    # ---- numeric summary (the part you can act on) ----
    print(f"\nchunks: {len(chunks)}  executed commands: {len(sent)}  "
          f"starved ticks: {sum(1 for e in execs if e.get('starved'))}")
    if len(chunks) > 1:
        births = np.array([c["birth_t"] for c in chunks])
        lat = np.array([c["merge_t"] - c["birth_t"] for c in chunks])
        print(f"replan period: median {np.median(np.diff(births)) * 1000:.0f} ms "
              f"(min {np.diff(births).min() * 1000:.0f}, max {np.diff(births).max() * 1000:.0f})")
        print(f"inference latency: median {np.median(lat) * 1000:.0f} ms  "
              f"max {lat.max() * 1000:.0f} ms  "
              f"({np.median(lat) * fps:.1f} ticks at {fps:.0f} Hz)")
    used = {}
    for e in sent:
        if "chunk_seq" in e:
            used.setdefault(e["chunk_seq"], []).append(e.get("plan_row", 0))
    if used:
        cons = [max(v) - min(v) + 1 for v in used.values()]
        print(f"rows consumed per chunk: median {np.median(cons):.1f} "
              f"(min {min(cons)}, max {max(cons)}) of {len(chunks[0]['actions'])} predicted")
    # chunk-boundary discontinuity: executed jump at each source-chunk change
    gapset = set(gap_at.tolist())
    jumps = []
    for i in range(1, len(sent)):
        if i - 1 in gapset:                   # skip reset gaps, not real jumps
            continue
        if sent[i].get("chunk_seq") != sent[i - 1].get("chunk_seq"):
            jumps.append(np.nanmax(np.abs(A[i] - A[i - 1])))
    if jumps:
        steps = np.nanmax(np.abs(np.diff(A, axis=0)), axis=1)
        steps = np.delete(steps, gap_at)
        print(f"max per-dim jump at chunk switch: median {np.median(jumps):.3f} "
              f"max {np.max(jumps):.3f}  (typical within-chunk step "
              f"{np.nanmedian(steps):.3f}) — a switch jump >> within-chunk step "
              "means discontinuous replans")
    if M is not None:
        err = np.nanmax(np.abs(A - M), axis=1)
        print(f"command-vs-measured error: median {np.nanmedian(err):.3f} "
              f"p95 {np.nanpercentile(err, 95):.3f} max {np.nanmax(err):.3f}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--dims", default=None, help="comma-separated action dims")
    ap.add_argument("--key", default=None, help="substring filter on action keys")
    ap.add_argument("--out", default=None, help="save PNG instead of showing")
    ap.add_argument("--tmin", type=float, default=None, help="zoom start (s)")
    ap.add_argument("--tmax", type=float, default=None, help="zoom end (s)")
    a = ap.parse_args()
    plot(a.trace, [int(x) for x in a.dims.split(",")] if a.dims else None,
         a.out, a.key, a.tmin, a.tmax)


if __name__ == "__main__":
    main()
