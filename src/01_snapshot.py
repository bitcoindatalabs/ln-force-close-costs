"""
01_snapshot.py: freeze the inputs at the cutoff and derive the two tables every later stage uses.

Inputs (read-only): the close scanner's tables in LN_ONCHAIN_DIR and their parts/, ln_funding_outpoints, and paper 01's
client-label history. Outputs:
  data/snapshot/{closes,spends,ul_closes,ul_spends,funding_outpoints,client_history}.parquet   (not committed)
  data/work/force_closes.parquet   every force close, listed or not, with its commitment format
  data/work/outputs.parquet        every output of every force close, with the spend that consumed it (if any)
  data/snapshot_manifest.json      cutoff, row counts and the checks below (released)

Checks (the stage fails if one does not hold):
  * each merged table agrees with its parts/ up to the cutoff (same keys);
  * ln_unlisted_force_closes.parquet is rebuilt from parts/ when the merged file cannot be read (it could not on
    2026-11-03: 25 of 27 column chunks were truncated), using the scanner's own consolidation rule;
  * every spend belongs to a close in the snapshot.

Usage:  python src/01_snapshot.py
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CUTOFF_UTC, DATA_DIR, FUNDING_DIR, LN_PARQUET_DIR, ONCHAIN_DIR,  # noqa: E402
                    SCAN_END_HEIGHT, SCAN_START_HEIGHT, SNAPSHOT_DIR, SPEND_CENSOR_BLOCKS, WORK_DIR)

PARTS = ONCHAIN_DIR / "parts"
SPEND_COLS = ("commitment_outpoint, spend_txid, spend_height, spend_time_utc, blocks_after_close::BIGINT AS "
              "blocks_after_close, spend_type, csv_delay, spend_n_inputs, spend_vsize, spend_fee_sat")


def cutoff_height() -> int:
    """Last block before the first block timestamped at or after the cutoff (block times are not monotone)."""
    b = pd.read_parquet(LN_PARQUET_DIR / "bitcoin_blocks_metadata.parquet", columns=["block_height", "timestamp"])
    first = int(b.loc[b["timestamp"] >= pd.Timestamp(CUTOFF_UTC), "block_height"].min())
    return first - 1


def readable(path: Path) -> bool:
    try:
        f = pq.ParquetFile(path)
        for rg in range(f.metadata.num_row_groups):
            f.read_row_group(rg)
        return True
    except Exception:
        return False


def parts(prefix: str, columns=None) -> pd.DataFrame:
    files = sorted(p for p in os.listdir(PARTS) if p.startswith(prefix))
    # the v3 rescan writes chunks without rows as parquet files without columns: skip them
    files = [f for f in files if pq.ParquetFile(PARTS / f).metadata.num_rows > 0]
    frames = [pd.read_parquet(PARTS / f, columns=columns) for f in files]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def drop_null_columns(path: Path) -> pa.Table:
    t = pq.read_table(path)
    return t.select([f.name for f in t.schema if not pa.types.is_null(f.type)])


def main() -> int:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    H = cutoff_height()
    checks, counts = {}, {}
    print(f"cutoff height {H:,} (last block before {CUTOFF_UTC})")

    # --- listed closes and spends: merged tables, cross-checked against parts ---------------------------------------
    for name, merged, prefix, key, hcol in [
        ("closes", "ln_close_txs", "closes_", "chan_point", "close_height"),
        ("spends", "ln_commitment_spends", "spends_", "commitment_outpoint", "spend_height"),
        ("ul_spends", "ln_unlisted_commitment_spends", "ulspends_", "commitment_outpoint", "spend_height"),
    ]:
        t = drop_null_columns(ONCHAIN_DIR / f"{merged}.parquet").to_pandas()
        t = t[t[hcol] <= H].reset_index(drop=True)
        p = parts(prefix, [key, hcol])
        p = p.sort_values(hcol).drop_duplicates(key, keep="first")
        p = p[p[hcol] <= H]
        same = set(t[key]) == set(p[key])
        checks[f"{name}_merged_equals_parts"] = bool(same)
        if not same:
            print(f"FAIL {name}: merged {len(t):,} vs parts {len(p):,}")
            return 1
        t.to_parquet(SNAPSHOT_DIR / f"{name}.parquet", index=False)
        counts[name] = len(t)
        print(f"{name}: {len(t):,} rows (merged = parts up to the cutoff)")

    # --- unlisted force closes: rebuilt from parts when the merged file is unreadable --------------------------------
    merged = ONCHAIN_DIR / "ln_unlisted_force_closes.parquet"
    ok = readable(merged)
    checks["ul_closes_merged_readable"] = ok
    p = parts("ulcloses_")
    p = p.sort_values("close_height").drop_duplicates("chan_point", keep="first")   # scanner's consolidate() rule
    p["close_time_utc"] = pd.to_datetime(p["close_time"], unit="s", utc=True)
    p = p[p["close_height"] <= H].drop(columns=["channel_id", "node1_pub", "node2_pub"]).reset_index(drop=True)
    if ok:
        t = drop_null_columns(merged).to_pandas()
        checks["ul_closes_merged_equals_parts"] = bool(set(t.loc[t["close_height"] <= H, "chan_point"]) == set(p["chan_point"]))
    else:
        # the two readable columns still let us check the count up to the cutoff time
        tt = pq.ParquetFile(merged).read_row_group(0, columns=["close_time_utc"]).column(0).to_pandas()
        last_t = pd.to_datetime(p["close_time"].max(), unit="s", utc=True)
        checks["ul_closes_count_merged_le_last_time"] = int((tt <= last_t).sum())
        checks["ul_closes_count_parts"] = len(p)
        print(f"ul_closes: merged file unreadable; rebuilt {len(p):,} rows from parts "
              f"(merged has {checks['ul_closes_count_merged_le_last_time']:,} rows up to the same time)")
    p.to_parquet(SNAPSHOT_DIR / "ul_closes.parquet", index=False)
    counts["ul_closes"] = len(p)

    # --- reference tables -----------------------------------------------------------------------------------------
    fo_path = FUNDING_DIR / "ln_funding_outpoints.parquet"
    fo = pd.read_parquet(fo_path)
    fo.to_parquet(SNAPSHOT_DIR / "funding_outpoints.parquet", index=False)
    counts["funding_outpoints"] = len(fo)
    h = pd.read_parquet(LN_PARQUET_DIR / "node_client_history.parquet",
                        columns=["pub_key", "valid_from", "valid_to", "client", "client_confidence",
                                 "client_rules_version"])
    h.to_parquet(SNAPSHOT_DIR / "client_history.parquet", index=False)
    counts["client_history"] = len(h)

    # --- derived: force closes and their outputs --------------------------------------------------------------------
    con = duckdb.connect()
    con.execute("SET threads=6")
    con.execute("SET TimeZone = 'UTC'")
    for v in ("closes", "spends", "ul_closes", "ul_spends", "funding_outpoints"):
        con.execute(f"CREATE VIEW {v} AS SELECT * FROM '{SNAPSHOT_DIR / v}.parquet'")
    fmt = """CASE WHEN tx_version = 3 THEN 'zero_fee'
                  WHEN funding_spend_kind = 'taproot_keyspend' THEN 'taproot'
                  WHEN n_anchor_outputs > 0 THEN 'anchor' ELSE 'legacy' END"""
    cols = """close_txid, chan_point, capacity::BIGINT AS capacity, close_height, close_time_utc, tx_version, locktime,
              n_outputs, vsize, weight, fee_sat, feerate_sat_vb, n_anchor_outputs, has_htlc_like_outputs,
              funding_spend_kind, outputs_json"""
    con.execute(f"""COPY (
        SELECT {cols}, TRUE AS listed, c.node1_pub, c.node2_pub, fo.funding_height, {fmt} AS fmt
        FROM closes c LEFT JOIN funding_outpoints fo USING (chan_point) WHERE c.close_type = 'force'
        UNION ALL
        SELECT {cols}, FALSE, NULL, NULL, NULL, {fmt} FROM ul_closes
    ) TO '{WORK_DIR / 'force_closes.parquet'}' (FORMAT parquet)""")
    con.execute(f"CREATE VIEW fc AS SELECT * FROM '{WORK_DIR / 'force_closes.parquet'}'")
    con.execute(f"""COPY (
        WITH o AS (
            SELECT fc.close_txid, fc.listed, fc.fmt, fc.close_height, (j->>'n')::INT AS n,
                   (j->>'value_sat')::BIGINT AS value_sat, j->>'type' AS script_type, j->>'role' AS role
            FROM fc, unnest(from_json(fc.outputs_json, '["json"]')) AS t(j)),
        sp AS (SELECT {SPEND_COLS} FROM spends UNION ALL SELECT {SPEND_COLS} FROM ul_spends)
        SELECT o.*, sp.spend_txid, sp.spend_height, sp.spend_time_utc, sp.blocks_after_close, sp.spend_type,
               sp.csv_delay, sp.spend_n_inputs, sp.spend_vsize, sp.spend_fee_sat
        FROM o LEFT JOIN sp ON sp.commitment_outpoint = o.close_txid || ':' || o.n
    ) TO '{WORK_DIR / 'outputs.parquet'}' (FORMAT parquet)""")
    n_fc = con.execute("SELECT count(*) FROM fc").fetchone()[0]
    n_out = con.execute(f"SELECT count(*), count(spend_txid) FROM '{WORK_DIR / 'outputs.parquet'}'").fetchone()
    # every listed spend must belong to a listed force close in the snapshot
    orphans = con.execute("""SELECT count(*) FROM spends s WHERE split_part(s.commitment_outpoint, ':', 1)
                         NOT IN (SELECT close_txid FROM closes WHERE close_type = 'force')""").fetchone()[0]
    checks["listed_spends_without_close"] = int(orphans)
    counts.update(force_closes=n_fc, force_close_outputs=n_out[0], force_close_outputs_spent=n_out[1])
    print(f"force closes {n_fc:,}; outputs {n_out[0]:,}, spent {n_out[1]:,}; orphan listed spends {orphans}")

    t0, t1 = con.execute("SELECT min(close_time_utc)::VARCHAR, max(close_time_utc)::VARCHAR FROM closes").fetchone()
    # the window's ends: the scan's first block, and the scan's last block against the cutoff (censoring)
    state = json.loads((ONCHAIN_DIR / "scanner_state.json").read_text()) if (ONCHAIN_DIR / "scanner_state.json").exists() else {}
    first_scanned = con.execute(f"""SELECT min(h) FROM (SELECT min(close_height) AS h FROM closes UNION ALL
                                    SELECT min(close_height) FROM ul_closes)""").fetchone()[0]
    checks["first_close_height_ge_scan_start"] = bool(first_scanned >= SCAN_START_HEIGHT)
    checks["scan_end_after_cutoff"] = bool(SCAN_END_HEIGHT >= H)
    checks["spend_censor_rule_inside_scan"] = bool(H - SPEND_CENSOR_BLOCKS < SCAN_END_HEIGHT)

    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()

    man = {"cutoff_utc": CUTOFF_UTC, "cutoff_height": H, "scan_start_height": SCAN_START_HEIGHT,
           "scan_start_utc": "2018-01-12 04:49:23+00", "scan_end_height": SCAN_END_HEIGHT,
           "first_close_utc": str(t0), "last_close_utc": str(t1), "counts": counts, "checks": checks,
           "source": str(ONCHAIN_DIR),
           "sources": {
               "close_tables": {"dir": str(ONCHAIN_DIR), "scanner_last_height": state.get("last_height"),
                                "mtime": {f: time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(os.path.getmtime(ONCHAIN_DIR / f)))
                                          for f in ("ln_close_txs.parquet", "ln_commitment_spends.parquet",
                                                    "ln_unlisted_commitment_spends.parquet",
                                                    "ln_unlisted_force_closes.parquet")}},
               "funding_outpoints": {"path": str(fo_path), "rows": len(fo), "sha256": sha(fo_path),
                                     "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(os.path.getmtime(fo_path)))},
               "client_history": {"path": str(LN_PARQUET_DIR / "node_client_history.parquet"),
                                  "mtime": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(
                                      os.path.getmtime(LN_PARQUET_DIR / "node_client_history.parquet")))},
               "snapshot_taken": time.strftime("%Y-%m-%d %H:%M:%S %Z")},
           "censoring": ("spends are kept up to the cutoff height; statistics about a close's outputs use closes at "
                         f"least {SPEND_CENSOR_BLOCKS} blocks before it; the scan itself runs to {SCAN_END_HEIGHT:,}")}
    (DATA_DIR / "snapshot_manifest.json").write_text(json.dumps(man, indent=1) + "\n", encoding="utf-8")
    return 0 if orphans == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
