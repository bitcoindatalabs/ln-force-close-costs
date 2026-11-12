"""
00_blockstats.py: per-block fee context for the study window (fee-rate percentiles, total fee, weight).

Source, in order of preference:
  1. the chain lake's blocks table (CHAIN_LAKE_DIR/parts/blocks_*.parquet, getblockstats per block), when it covers
     every block of the window;
  2. otherwise getblockstats from our node, one call per block, single-threaded and resumable (about 10 ms a call).

Writes data/work/blockstats.parquet: height, time, txs, total_weight, totalfee, avgfeerate, minfeerate, p10, p25,
p50, p75, p90 (sat/vB, as Bitcoin Core reports them: integer, weighted by vsize).

Usage:  python src/00_blockstats.py [--end HEIGHT]     (default end: the cutoff height + 6 confirmations)
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHAIN_LAKE_DIR, SCAN_START_HEIGHT, WORK_DIR  # noqa: E402

OUT = WORK_DIR / "blockstats.parquet"
STATS = ["height", "time", "txs", "total_weight", "totalfee", "avgfeerate", "minfeerate", "feerate_percentiles"]
PCT = ["p10", "p25", "p50", "p75", "p90"]


def from_lake(start: int, end: int):
    parts = sorted((CHAIN_LAKE_DIR / "parts").glob("blocks_*.parquet"))
    if not parts:
        return None
    cols = ["height", "time", "n_tx", "weight", "bs_totalfee", "bs_avgfeerate", "bs_minfeerate", "bs_feerate_percentiles"]
    frames = []
    for p in parts:
        lo, hi = (int(x) for x in p.stem.split("_")[1:3])
        if hi >= start and lo <= end:
            frames.append(pd.read_parquet(p, columns=cols))
    if not frames:
        return None
    b = pd.concat(frames).query("@start <= height <= @end")
    if b["height"].nunique() != end - start + 1:
        return None
    pct = pd.DataFrame(b.pop("bs_feerate_percentiles").tolist(), columns=PCT, index=b.index)
    b = pd.concat([b, pct], axis=1).rename(columns={"n_tx": "txs", "weight": "total_weight", "bs_totalfee": "totalfee",
                                                    "bs_avgfeerate": "avgfeerate", "bs_minfeerate": "minfeerate"})
    b["source"] = "chain_lake"
    return b.sort_values("height").reset_index(drop=True)


def from_node(start: int, end: int) -> pd.DataFrame:
    from fcc.rpc import Rpc
    rpc = Rpc()
    done = pd.read_parquet(OUT) if OUT.exists() else pd.DataFrame(columns=["height"])
    have = set(done["height"].astype(int))
    todo = [h for h in range(start, end + 1) if h not in have]
    print(f"node: {len(have):,} blocks cached, {len(todo):,} to fetch", flush=True)
    rows, t0 = [], time.time()

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
            done.sort_values("height").to_parquet(OUT, index=False)
            rows = []

    for k, h in enumerate(todo):
        s = rpc.call("getblockstats", [h, STATS])
        p = s.pop("feerate_percentiles")
        rows.append({**s, **dict(zip(PCT, p)), "source": "node"})
        if (k + 1) % 5000 == 0:
            flush()
            print(f"  {k + 1:,}/{len(todo):,}  height {h:,}  {time.time() - t0:,.0f}s", flush=True)
    flush()
    return done.sort_values("height").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", type=int, required=True, help="last height (the snapshot cutoff height)")
    a = ap.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    b = from_lake(SCAN_START_HEIGHT, a.end)
    if b is None:
        print("chain lake does not cover the window; asking the node")
        b = from_node(SCAN_START_HEIGHT, a.end)
    else:
        b.to_parquet(OUT, index=False)
    missing = (a.end - SCAN_START_HEIGHT + 1) - b["height"].nunique()
    print(f"blockstats: {len(b):,} blocks {b['height'].min():,}..{b['height'].max():,}, missing {missing}")
    return 0 if missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
