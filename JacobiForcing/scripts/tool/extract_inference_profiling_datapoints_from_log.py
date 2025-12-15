import argparse
import json
import math
import os
import re
from pathlib import Path


RUN_ID_RE = re.compile(
    r"ntok(?P<block_size>\d+)_K(?P<K>\d+)_r(?P<r>[0-9.]+)_ng(?P<ngram_size>\d+)"
)

TOKS_PER_SEC_RE = re.compile(r"Avg\s+toks/sec:\s*([0-9.eE+-]+)")
ITERS_PER_TOKEN_RE = re.compile(r"Avg\s+iterations\s*/\s*token:\s*([0-9.eE+-]+)")


def parse_run_id(text: str, fallback_name: str = ""):
    """
    Extract K, r, block_size, ngram_size from either the [RUN] line or the filename.
    """
    m_run = re.search(r"\[RUN\]\s+([^\s]+)", text)
    run_id_str = None
    run_id_str = m_run.group(1).strip()
    
    assert run_id_str is not None, f"file name {text} is not in valid format."


    m = RUN_ID_RE.search(run_id_str)
    if not m:
        return None

    return {
        "block_size": int(m.group("block_size")),
        "K": int(m.group("K")),
        "r": float(m.group("r")),
        "ngram_size": int(m.group("ngram_size")),
    }


def parse_metrics(text: str):
    """
    Extract Avg toks/sec and Avg iterations / token from the EOS-only summary.
    """
    m_tps = TOKS_PER_SEC_RE.search(text)
    m_ipt = ITERS_PER_TOKEN_RE.search(text)

    if not (m_tps and m_ipt):
        return None

    avg_toks_per_sec = float(m_tps.group(1))
    avg_iters_per_token = float(m_ipt.group(1))

    # Compute Avg tokens / iteration; guard against zero
    if avg_iters_per_token > 0:
        avg_tokens_per_iter = 1.0 / avg_iters_per_token
    else:
        avg_tokens_per_iter = math.nan

    return {
        "avg_toks_per_sec": avg_toks_per_sec,
        "avg_iters_per_token": avg_iters_per_token,
        "avg_tokens_per_iter": avg_tokens_per_iter,
    }


def collect_from_logs(log_dir: Path):
    entries = []

    for log_path in sorted(log_dir.rglob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")

        params = parse_run_id(text, fallback_name=log_path.stem)
        if params is None:
            print(f"[WARNING!!!] Could not parse run id from {log_path}")
            continue

        metrics = parse_metrics(text)
        if metrics is None:
            print(f"[WARNING!!!] Could not parse metrics from {log_path}")
            continue

        entry = {
            "log_path": str(log_path),
        }
        entry.update(params)
        entry.update(metrics)
        entries.append(entry)

    return entries


def sanitize_for_json_value(v):
    """
    Replace NaN/inf with None so json is valid.
    """
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        default="/home/lah003/workspace/CLLM2/logs/hparam_sweep_20251125_171121",
        help="Directory containing *.log files",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default=None,
        help="Output JSONL path (default: <log_dir>/summary.jsonl)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        raise SystemExit(f"Log directory does not exist or is not a directory: {log_dir}")

    out_jsonl = (
        Path(args.out_jsonl)
        if args.out_jsonl is not None
        else log_dir / "summary.jsonl"
    )

    entries = collect_from_logs(log_dir)
    if not entries:
        print(f"[INFO] No valid entries found under {log_dir}")
        return

    # ---- Find best runs ----
    def valid_float(e, key):
        v = e.get(key)
        return isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)

    # best avg_toks_per_sec
    valid_tps_entries = [e for e in entries if valid_float(e, "avg_toks_per_sec")]
    best_tps = max(valid_tps_entries, key=lambda e: e["avg_toks_per_sec"]) if valid_tps_entries else None

    # best avg_tokens_per_iter
    valid_tpi_entries = [e for e in entries if valid_float(e, "avg_tokens_per_iter")]
    best_tpi = max(valid_tpi_entries, key=lambda e: e["avg_tokens_per_iter"]) if valid_tpi_entries else None

    if best_tps:
        print("\n=== Best Avg toks/sec ===")
        print(f"value          : {best_tps['avg_toks_per_sec']:.4f}")
        print(f"log_path       : {best_tps['log_path']}")
        print(f"K              : {best_tps['K']}")
        print(f"r              : {best_tps['r']}")
        print(f"block_size     : {best_tps['block_size']}")
        print(f"ngram_size     : {best_tps['ngram_size']}")

    if best_tpi:
        print("\n=== Best Avg tokens / iteration ===")
        print(f"value          : {best_tpi['avg_tokens_per_iter']:.6f}")
        print(f"log_path       : {best_tpi['log_path']}")
        print(f"K              : {best_tpi['K']}")
        print(f"r              : {best_tpi['r']}")
        print(f"block_size     : {best_tpi['block_size']}")
        print(f"ngram_size     : {best_tpi['ngram_size']}")

    # ---- Write JSONL ----
    os.makedirs(out_jsonl.parent, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for item in entries:
            sanitized = {k: sanitize_for_json_value(v) for k, v in item.items()}
            f.write(json.dumps(sanitized) + "\n")

    print(f"\n[DONE] Wrote {len(entries)} entries to {out_jsonl}")


if __name__ == "__main__":
    main()
