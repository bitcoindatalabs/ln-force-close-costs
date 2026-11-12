"""
09_lifecycle.py (P10, RQ4): what a public channel's whole life costs on-chain: its funding transaction, its close and
the spends the close requires, by year and fee regime.

Two steps.

  --extract   (reads the lakes; run outside the cron busy windows, under the lightning_lake lock)
      From FUNDING_DIR/ln_funding_outpoints (every public channel we know, including the lnresearch archive) and the
      chain lake (CHAIN_LAKE_DIR: parts/{txs,outputs,inputs} + linked/{txs,outputs,inputs}_link, linked and validated
      2026-11-06), for every funding transaction: its fee, size and outputs, which of its inputs spend a 2-of-2
      multisig (a splice, or a close that opens a new channel), and how each of its P2WSH/P2TR outputs was spent (a
      2-of-2 witness or a commitment-pattern spender marks an output as a channel we don't list). Also keeps the old
      fee source (ln_funding_txs) for the coverage comparison. Writes three work tables:
        funding_channels.parquet    one row per listed channel (chan_point, funding tx, capacity, in_snapshot)
        funding_txs.parquet         one row per funding transaction (fee, vsize, outputs, old fee if any)
        funding_tx_outputs.parquet  one row per output of a funding transaction, with its spender
  (default)   (reads only the snapshot and the work tables)
      Lifecycle account per public channel of the snapshot list:
        open     the channel's share of its funding transaction's fee (attribution rules below)
        close    cooperative close: the close fee; force close: commitment fee + same-block CPFP child (as 03_costs.py)
        sweeps   force close only: HTLC resolution, to_local / anchor to_remote sweeps, later owner anchor spends
      Third-party anchor sweeps are not a fee the channel pays (reported as value lost, as in 03_costs.py).

Attribution rules for shared transactions (notes.md has the argument):
  Opens, main rule, the equal split ("the funder pays; change is part of the open"; key a): the fee is split equally among the
      channel outputs of the transaction (listed ones, plus outputs later spent by a 2-of-2 witness or a commitment
      transaction: channels outside our list). Every other output is treated as the funder's change.
  Opens, sensitivity: by value (key b: capacity / all outputs' value); by size (key c: the output's own vbytes plus
      an equal share of the transaction's shared bytes, header and inputs, over all outputs); payment batches by size
      (key a2: the equal split, except that a transaction with two or more non-channel outputs is split by size).
  Splices: a transaction that closes a listed channel and funds another is charged once, to the new channel's open;
      the old channel's close costs 0 and its lifecycle is marked spliced.
  A funding transaction that also sweeps force-close outputs: the sweep inputs keep their per-input share (as in
      Table II) and the channels get the remainder, so no fee is counted twice.
  Closing side: spends that batch several closes are split equally per input (as Table II); sensitivity: the whole
      transaction to each close (upper bound). Cooperative closes with extra inputs are split per input as well.

Completed lifecycles: channels of the snapshot list closed in the window (from block 503,816, 12 January 2018) and at
least SPEND_CENSOR_BLOCKS before the
cutoff (so their outputs had time to be spent), as in 03_costs.py.

Writes data/lifecycle.json, paper/tables/table_lifecycle.tex, paper/figures/fig8_lifecycle.pdf (or under --out-dir).

Usage:
  python src/09_lifecycle.py --extract     # chain lake -> data/work/funding_*.parquet
  python src/09_lifecycle.py
"""

import argparse
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import config  # noqa: E402
from config import (ANCHOR_VALUE_SAT, CHAIN_LAKE_DIR, FEE_PERIODS, FUNDING_DIR, PALETTE,  # noqa: E402
                    PUBLIC_FROM, SCAN_START_TS, SNAPSHOT_DIR, SPEND_CENSOR_BLOCKS, coverage)

PUBLIC_YEAR = int(PUBLIC_FROM[:4])
import fcc.style as style  # noqa: E402
from fcc.io import connect, df, manifest  # noqa: E402

PRE_SCAN = SCAN_START_TS               # block 503,816: channels closed before it have no close in the snapshot
PRE_LABEL = "pre-2018"                  # opens before the first fee period
TWO_OF_TWO = """(witness_script_len = 71 AND hex(witness_script) LIKE '5221%52AE'
                 AND substr(hex(witness_script), 71, 2) = '21')"""


# ---------------------------------------------------------------------------------------------------------------------
# Step 1: extract funding transactions from the chain lake
# ---------------------------------------------------------------------------------------------------------------------

def chunk_names() -> dict:
    """chunk start height -> '<start>_<end>' as in the chain lake's file names."""
    out = {}
    for p in (CHAIN_LAKE_DIR / "parts").glob("txs_*.parquet"):
        s, e = p.stem.split("_")[1:]
        out[int(s)] = f"{s}_{e}"
    return out


def extract(work: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    names = chunk_names()
    P, K = CHAIN_LAKE_DIR / "parts", CHAIN_LAKE_DIR / "linked"
    con = duckdb.connect()
    con.execute("SET threads=8")
    con.execute(f"""CREATE TABLE ch AS
        SELECT chan_point, split_part(chan_point, ':', 1) AS txid, split_part(chan_point, ':', 2)::INT AS vout,
               funding_height, capacity_sat, coalesce(in_gossip_archive, FALSE) AS in_gossip_archive,
               chan_point IN (SELECT chan_point FROM '{SNAPSHOT_DIR / 'funding_outpoints.parquet'}') AS in_snapshot
        FROM '{FUNDING_DIR / 'ln_funding_outpoints.parquet'}'""")
    con.execute("""CREATE TABLE tgt AS SELECT unhex(txid) AS txid_b, min(funding_height) AS h,
                          min(funding_height) - min(funding_height) % 1000 AS chunk
                   FROM ch GROUP BY txid""")
    n_ch, n_tx = con.execute("SELECT (SELECT count(*) FROM ch), count(*) FROM tgt").fetchone()
    print(f"listed channels {n_ch:,}; funding transactions {n_tx:,}")
    con.execute("""CREATE TABLE ftx (tx_key BIGINT, height INT, txid VARCHAR, version INT, locktime BIGINT, vsize INT,
                   weight INT, n_in INT, n_out INT, output_value BIGINT, input_value BIGINT, fee BIGINT,
                   block_time BIGINT, n_inputs_2of2 INT, n_inputs_p2wsh INT)""")
    con.execute("""CREATE TABLE fout (tx_key BIGINT, vout INT, value BIGINT, script_type VARCHAR, script_len INT,
                   spend_tx_key BIGINT, spend_vin INT)""")
    t0 = time.time()
    chunks = [c for (c,) in con.execute("SELECT DISTINCT chunk FROM tgt ORDER BY chunk").fetchall()]
    for k, c in enumerate(chunks):
        nm = names[c]
        con.execute(f"""CREATE OR REPLACE TEMP TABLE t AS
            SELECT t.tx_key, t.height, hex(t.txid) AS txid, t.version, t.locktime, t.vsize, t.weight, t.n_in, t.n_out,
                   t.output_value, l.input_value, l.fee, b.time AS block_time
            FROM '{P}/txs_{nm}.parquet' t JOIN (SELECT txid_b FROM tgt WHERE chunk = {c}) g ON g.txid_b = t.txid
            JOIN '{K}/txs_link_{nm}.parquet' l USING (tx_key)
            JOIN (SELECT height, time FROM '{P}/blocks_{nm}.parquet') b USING (height)""")
        con.execute(f"""INSERT INTO fout
            SELECT o.tx_key, o.vout, o.value, o.script_type, o.script_len, k.spend_tx_key, k.spend_vin
            FROM '{P}/outputs_{nm}.parquet' o JOIN '{K}/outputs_link_{nm}.parquet' k USING (tx_key, vout)
            WHERE o.tx_key IN (SELECT tx_key FROM t)""")
        con.execute(f"""INSERT INTO ftx
            SELECT t.*, coalesce(i.n2, 0), coalesce(i.nw, 0) FROM t LEFT JOIN (
                SELECT i.tx_key, count(*) FILTER ({TWO_OF_TWO}) AS n2, count(*) FILTER (k.prev_script_type = 'p2wsh') AS nw
                FROM '{P}/inputs_{nm}.parquet' i JOIN '{K}/inputs_link_{nm}.parquet' k USING (tx_key, vin)
                WHERE i.tx_key IN (SELECT tx_key FROM t) GROUP BY i.tx_key) i USING (tx_key)""")
        if k % 50 == 0:
            print(f"  chunk {c:,} ({k + 1}/{len(chunks)}) {time.time() - t0:.0f}s")
    found = con.execute("SELECT count(*) FROM ftx").fetchone()[0]
    print(f"pass 1: {found:,} of {n_tx:,} funding transactions found, {time.time() - t0:.0f}s")

    # pass 2: who spent the P2WSH / P2TR outputs of funding transactions (listed ones too: a check against the snapshot)
    con.execute("""CREATE TABLE need AS SELECT DISTINCT spend_tx_key, spend_vin,
                          (spend_tx_key >> 15) - (spend_tx_key >> 15) % 1000 AS chunk
                   FROM fout WHERE spend_tx_key IS NOT NULL AND script_type IN ('p2wsh', 'p2tr')""")
    con.execute("""CREATE TABLE spender (spend_tx_key BIGINT, spend_vin INT, spend_txid VARCHAR, spend_locktime BIGINT,
                   spend_sequence BIGINT, spend_2of2 BOOLEAN)""")
    chunks2 = [c for (c,) in con.execute("SELECT DISTINCT chunk FROM need ORDER BY chunk").fetchall()]
    t1 = time.time()
    for k, c in enumerate(chunks2):
        nm = names[c]
        con.execute(f"""INSERT INTO spender
            SELECT n.spend_tx_key, n.spend_vin, hex(t.txid), t.locktime, i.sequence, {TWO_OF_TWO}
            FROM (SELECT spend_tx_key, spend_vin FROM need WHERE chunk = {c}) n
            JOIN '{P}/txs_{nm}.parquet' t ON t.tx_key = n.spend_tx_key
            JOIN '{P}/inputs_{nm}.parquet' i ON i.tx_key = n.spend_tx_key AND i.vin = n.spend_vin""")
        if k % 50 == 0:
            print(f"  spend chunk {c:,} ({k + 1}/{len(chunks2)}) {time.time() - t1:.0f}s")
    print(f"pass 2: {con.execute('SELECT count(*) FROM spender').fetchone()[0]:,} spenders of "
          f"{con.execute('SELECT count(*) FROM need').fetchone()[0]:,}, {time.time() - t1:.0f}s")

    con.execute(f"COPY ch TO '{work / 'funding_channels.parquet'}' (FORMAT parquet)")
    con.execute(f"""COPY (
        SELECT f.* REPLACE (lower(f.txid) AS txid), ft.fee_sat AS old_fee_sat, ft.vsize AS old_vsize,
               ft.n_funding_outputs AS old_n_funding_outputs
        FROM ftx f LEFT JOIN '{FUNDING_DIR / 'ln_funding_txs.parquet'}' ft ON ft.txid = lower(f.txid)
    ) TO '{work / 'funding_txs.parquet'}' (FORMAT parquet)""")
    con.execute(f"""COPY (
        SELECT lower(f.txid) AS txid, o.vout, o.value, o.script_type, o.script_len, o.spend_tx_key, o.spend_vin,
               (o.spend_tx_key >> 15)::INT AS spend_height, lower(s.spend_txid) AS spend_txid,
               s.spend_2of2, (s.spend_locktime >> 24) = 32 AND (s.spend_sequence >> 24) = 128 AS spend_commitment,
               (lower(f.txid) || ':' || o.vout) IN (SELECT chan_point FROM ch) AS listed
        FROM fout o JOIN ftx f USING (tx_key)
        LEFT JOIN spender s USING (spend_tx_key, spend_vin)
    ) TO '{work / 'funding_tx_outputs.parquet'}' (FORMAT parquet)""")
    print(f"wrote {work}/funding_{{channels,txs,tx_outputs}}.parquet in {time.time() - t0:.0f}s")
    return 0 if found == n_tx else 1


# ---------------------------------------------------------------------------------------------------------------------
# Step 2: lifecycle account
# ---------------------------------------------------------------------------------------------------------------------

REQUIRED = ("bump", "htlc", "sweeps", "anchor_cleanup")     # 03_costs.PARTY without the commitment itself
OUT_BYTES = "(8 + CASE WHEN script_len < 253 THEN 1 WHEN script_len < 65536 THEN 3 ELSE 5 END + script_len)"
RULES = {"a": "equal split over the channel outputs, change pays nothing (main rule)",
         "a2": "equal split, but payment batches (2+ non-channel outputs) split by size",
         "b": "by value: every output, change included, pays in proportion to its amount",
         "c": "by size: every output, change included, pays for its own vbytes plus a share of the rest",
         "whole": "whole funding tx to each channel (upper bound)"}


def regime_sql(col: str) -> str:
    """FEE_PERIODS as a SQL CASE; opens before the first period are PRE_LABEL."""
    arms = " ".join(f"WHEN {col} < TIMESTAMPTZ '{end} 00:00:00+00' THEN '{name}'" for name, _, end in FEE_PERIODS)
    return f"CASE WHEN {col} < TIMESTAMPTZ '{FEE_PERIODS[0][1]} 00:00:00+00' THEN '{PRE_LABEL}' {arms} END"


def build(con, work: Path, cutoff: int, category_sql: str):
    """Per-channel open costs (all listed channels) and per-channel lifecycles (closed channels of the snapshot)."""
    con.execute(f"CREATE OR REPLACE VIEW fch AS SELECT * FROM '{work / 'funding_channels.parquet'}'")
    con.execute(f"CREATE OR REPLACE VIEW ftx AS SELECT * FROM '{work / 'funding_txs.parquet'}'")
    con.execute(f"""CREATE OR REPLACE VIEW fto AS SELECT * REPLACE (coalesce(spend_2of2, FALSE) AS spend_2of2,
                           coalesce(spend_commitment, FALSE) AS spend_commitment)
                    FROM '{work / 'funding_tx_outputs.parquet'}'""")

    # closing side: every spend of a force-close output, allocated as in 03_costs.py (per input; same-block bump whole)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE alloc AS
        SELECT close_txid, listed, {category_sql} AS cat, spend_txid,
               spend_fee_sat * CASE WHEN spend_type IN ('anchor_sweep_owner', 'no_witness') AND blocks_after_close = 0
                                    THEN 1.0 ELSE 1.0 / spend_n_inputs END AS fee,
               spend_fee_sat::DOUBLE AS fee_whole
        FROM o WHERE spend_txid IS NOT NULL""")
    req = ", ".join(f"'{k}'" for k in REQUIRED[1:])
    con.execute(f"""CREATE OR REPLACE TEMP TABLE fcost AS
        SELECT fc.close_txid, fc.fee_sat AS commitment, fc.fmt,
               coalesce(sum(a.fee) FILTER (a.cat = 'bump'), 0) AS bump,
               coalesce(sum(a.fee) FILTER (a.cat IN ({req})), 0) AS sweeps,
               coalesce(sum(a.fee_whole) FILTER (a.cat IN ({req})), 0) AS sweeps_whole,
               count(a.cat) FILTER (a.cat = 'third_party_anchor') AS anchors_lost
        FROM fc LEFT JOIN alloc a USING (close_txid) WHERE fc.listed GROUP BY ALL""")
    # the share of a transaction's fee already charged to force closes (when a sweep also funds a channel)
    all_req = ", ".join(f"'{k}'" for k in REQUIRED)
    con.execute(f"""CREATE OR REPLACE TEMP TABLE req AS SELECT spend_txid AS txid, sum(fee) AS req_fee
                    FROM alloc WHERE cat IN ({all_req}) GROUP BY 1""")

    # opening side: one row per funding transaction, then per listed channel
    con.execute(f"""CREATE OR REPLACE TEMP TABLE ftx2 AS
        WITH out AS (
            SELECT txid, count(*) AS n_out_seen, count(*) FILTER (listed) AS n_listed,
                   count(*) FILTER (listed OR (script_type IN ('p2wsh', 'p2tr') AND (spend_2of2 OR spend_commitment)))
                       AS n_chan,
                   sum({OUT_BYTES}) AS out_bytes
            FROM fto GROUP BY txid)
        SELECT f.*, out.n_out_seen, out.n_listed, out.n_chan, f.n_out - out.n_chan AS n_other,
               f.vsize - out.out_bytes AS shared_vb, coalesce(req.req_fee, 0) AS req_fee,
               greatest(f.fee - coalesce(req.req_fee, 0), 0) AS pool,
               f.txid IN (SELECT close_txid FROM c WHERE close_type = 'splice') AS closes_listed_channel
        FROM ftx f JOIN out USING (txid) LEFT JOIN req USING (txid)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE opens AS
        SELECT ch.chan_point, ch.in_snapshot, ch.in_gossip_archive, ch.capacity_sat AS list_capacity_sat,
               o.value AS capacity_sat, ch.funding_height,       -- capacity from the chain: the list has 0/NULL for 40k
               to_timestamp(f.block_time) AS open_time, f.txid, f.fee, f.vsize, f.n_in, f.n_out, f.n_chan, f.n_listed,
               f.n_other, f.pool, f.req_fee, f.old_fee_sat, f.old_vsize, f.closes_listed_channel, f.n_inputs_2of2,
               o.value AS out_value,
               f.pool / f.n_chan AS open_a,
               f.pool * o.value / f.output_value AS open_b,
               f.pool * ({OUT_BYTES.replace('script_len', 'o.script_len')} + f.shared_vb / f.n_out) / f.vsize AS open_c,
               f.fee::DOUBLE AS open_whole,
               f.vsize / f.n_chan AS open_vb
        FROM fch ch LEFT JOIN ftx2 f ON f.txid = ch.txid LEFT JOIN fto o ON o.txid = ch.txid AND o.vout = ch.vout""")
    op = df(con, "SELECT * FROM opens")
    op["open_a2"] = np.where(op["n_other"] >= 2, op["open_c"], op["open_a"])

    lc = df(con, f"""
        SELECT c.chan_point, c.close_type, c.close_txid, c.close_height, c.close_time_utc,
               year(c.close_time_utc) AS y, c.fee_sat AS close_tx_fee, c.n_inputs AS close_n_inputs,
               {regime_sql('c.close_time_utc')} AS close_regime,
               fcost.commitment, fcost.bump, fcost.sweeps AS f_sweeps, fcost.sweeps_whole AS f_sweeps_whole,
               fcost.anchors_lost, fcost.fmt,
               c.close_txid IN (SELECT txid FROM ftx) AS close_tx_funds_listed,
               c.close_height > {cutoff - SPEND_CENSOR_BLOCKS} AS censored
        FROM c LEFT JOIN fcost USING (close_txid)""")
    force = lc["close_type"] == "force"
    spliced = (lc["close_type"] == "splice") & lc["close_tx_funds_listed"]
    lc["close"] = np.select([force, spliced], [lc["commitment"] + lc["bump"], 0.0],
                            lc["close_tx_fee"] / lc["close_n_inputs"])
    lc["close_whole"] = np.select([force, spliced], [lc["commitment"] + lc["bump"], 0.0], lc["close_tx_fee"])
    lc["sweeps"] = np.where(force, lc["f_sweeps"], 0.0)
    lc["sweeps_whole"] = np.where(force, lc["f_sweeps_whole"], 0.0)
    lc["anchors_lost"] = lc["anchors_lost"].fillna(0)
    lc["spliced"] = spliced
    lc = lc.merge(op[op["in_snapshot"]].drop(columns=["in_snapshot"]), on="chan_point", how="left")
    lc["open_regime"] = pd.cut(lc["open_time"], [pd.Timestamp("2000-01-01", tz="UTC")] +
                               [pd.Timestamp(s, tz="UTC") for _, s, _ in FEE_PERIODS] +
                               [pd.Timestamp(FEE_PERIODS[-1][2], tz="UTC")],
                               labels=[PRE_LABEL] + [k for k, _, _ in FEE_PERIODS], right=False).astype(str)
    for r in ("a", "a2", "b", "c", "whole"):
        lc[f"total_{r}"] = lc[f"open_{r}"] + lc["close"] + lc["sweeps"]
    lc["total_close_whole"] = lc["open_a"] + lc["close_whole"] + lc["sweeps_whole"]
    lc["life_days"] = (lc["close_height"] - lc["funding_height"]) / 144
    return op, lc


def checks(con, op: pd.DataFrame, lc: pd.DataFrame, cutoff: int, work: Path) -> dict:
    ft = df(con, "SELECT fee, old_fee_sat, vsize, old_vsize FROM ftx WHERE old_fee_sat IS NOT NULL")
    sp = df(con, f"""SELECT count(*) AS n, count(*) FILTER (o.spend_txid = c.close_txid) AS same,
                            count(*) FILTER (o.spend_txid IS NULL) AS unspent_in_lake,
                            -- the lake spender's shape (commitment fingerprint or not) against our close type
                            count(*) FILTER (o.spend_txid = c.close_txid
                                             AND o.spend_commitment = (c.close_type = 'force')) AS shape_agrees
                     FROM c JOIN fto o ON o.txid || ':' || o.vout = c.chan_point""").iloc[0]
    out = {
        "funding_txs": int(con.execute("SELECT count(*) FROM ftx").fetchone()[0]),
        "listed_channels": int(len(op)), "listed_channels_without_fee": int(op["fee"].isna().sum()),
        "old_source_txs_compared": int(len(ft)),
        "old_source_fee_equal": int((ft["fee"] == ft["old_fee_sat"]).sum()),
        "old_source_vsize_equal": int((ft["vsize"] == ft["old_vsize"]).sum()),
        "list_capacity_missing": int((op["list_capacity_sat"].fillna(0) == 0).sum()),
        "list_capacity_equals_output_value": int((op["out_value"] == op["list_capacity_sat"]).sum()),
        "snapshot_closes": int(con.execute("SELECT count(*) FROM c").fetchone()[0]),
        "snapshot_closes_not_in_list": int(con.execute(
            "SELECT count(*) FROM c WHERE chan_point NOT IN (SELECT chan_point FROM fch)").fetchone()[0]),
        "snapshot_closes_not_in_list_censored": int(con.execute(
            f"SELECT count(*) FROM c WHERE chan_point NOT IN (SELECT chan_point FROM fch) "
            f"AND close_height > {cutoff - SPEND_CENSOR_BLOCKS}").fetchone()[0]),
        "snapshot_closes_compared": int(sp["n"]), "snapshot_close_txid_equals_lake_spender": int(sp["same"]),
        "snapshot_close_type_equals_lake_shape": int(sp["shape_agrees"]),
        "lifecycles_without_open": int(lc["open_a"].isna().sum()),
    }
    cc = config.WORK_DIR / "close_costs.parquet"      # 03_costs.py: the force side must reproduce Table II exactly
    if cc.exists():
        t = pd.read_parquet(cc, columns=["close_txid", "party_fee"])
        m = t.merge(lc.loc[lc["close_type"] == "force", ["close_txid", "close", "sweeps"]], on="close_txid")
        out["table2_closes_compared"] = int(len(m))
        out["table2_party_fee_max_abs_diff"] = float((m["party_fee"] - m["close"] - m["sweeps"]).abs().max())
    return out


def q(x, p):
    return float(np.nanquantile(x, p)) if len(x) else float("nan")


def opens_by_year(op: pd.DataFrame) -> pd.DataFrame:
    op = op.assign(y=op["open_time"].dt.year)
    rows = []
    for (y, snap), g in [((y, True), g) for y, g in op[op["in_snapshot"]].groupby("y")] + \
                        [((y, False), g) for y, g in op.groupby("y")]:
        f = g[g["fee"].notna()]
        rows.append({"y": int(y), "list": "snapshot" if snap else "live", "n": len(g),
                     "cov_chain": len(f) / len(g), "cov_old": float(g["old_fee_sat"].notna().mean()),
                     **{f"med_open_{r}": float(f[f"open_{r}"].median()) for r in RULES},
                     "med_open_vb": float(f["open_vb"].median()),
                     "med_share_cap_a": float((f["open_a"] / f["capacity_sat"]).median()),
                     "agg_share_cap_a": float(f["open_a"].sum() / f["capacity_sat"].sum()),
                     "share_batch": float((f["n_chan"] > 1).mean()), "share_payment_batch": float((f["n_other"] >= 2).mean()),
                     "share_other_channels": float((f["n_chan"] > f["n_listed"]).mean()),
                     "share_spends_2of2": float((f["n_inputs_2of2"] > 0).mean()),
                     "share_also_sweeps": float((f["req_fee"] > 0).mean()),
                     **{f"sum_open_{r}_btc": float(f[f"open_{r}"].sum() / 1e8) for r in ("a", "a2", "b", "c")},
                     "funding_tx_fees_btc": float(f.drop_duplicates("txid")["fee"].sum() / 1e8)})
    return pd.DataFrame(rows).sort_values(["list", "y"]).reset_index(drop=True)


def summarize(g: pd.DataFrame) -> dict:
    fz = g[g["close_type"] == "force"]
    co = g[g["close_type"] != "force"]
    tot = g["total_a"]
    return {
        "n": len(g), "share_force": float(len(fz) / len(g)), "n_spliced": int(g["spliced"].sum()),
        "med_open": float(g["open_a"].median()), "med_close": float(g["close"].median()),
        "med_sweeps_force": float(fz["sweeps"].median()) if len(fz) else None,
        "med_total": float(tot.median()), "q25_total": q(tot, 0.25), "q75_total": q(tot, 0.75), "p90_total": q(tot, 0.9),
        "mean_open": float(g["open_a"].mean()), "mean_close": float(g["close"].mean()),
        "mean_sweeps": float(g["sweeps"].mean()), "mean_total": float(tot.mean()),
        "med_total_coop": float(co["total_a"].median()) if len(co) else None,
        "med_total_force": float(fz["total_a"].median()) if len(fz) else None,
        "med_open_share_of_total_coop": float((co["open_a"] / co["total_a"]).median()) if len(co) else None,
        "med_share_cap": float((tot / g["capacity_sat"]).median()),
        "med_share_cap_coop": float((co["total_a"] / co["capacity_sat"]).median()) if len(co) else None,
        "med_share_cap_force": float((fz["total_a"] / fz["capacity_sat"]).median()) if len(fz) else None,
        "share_over_1pct_cap": float((tot / g["capacity_sat"] > 0.01).mean()),
        "agg_share_cap": float(tot.sum() / g["capacity_sat"].sum()),
        "total_btc": float(tot.sum() / 1e8), "open_btc": float(g["open_a"].sum() / 1e8),
        "close_btc": float(g["close"].sum() / 1e8), "sweeps_btc": float(g["sweeps"].sum() / 1e8),
        "open_share_of_fees": float(g["open_a"].sum() / tot.sum()),
        "anchor_value_lost_btc": float(ANCHOR_VALUE_SAT * g["anchors_lost"].sum() / 1e8),
        "med_life_days": float(g["life_days"].median()),
    }


def sensitivity(lc: pd.DataFrame, op: pd.DataFrame, from_year: int) -> dict:
    out = {"lifecycle_by_close_year": [], "opens_by_year": []}
    variants = {"a": "total_a", "a2": "total_a2", "b": "total_b", "c": "total_c", "open_whole": "total_whole",
                "closing_side_whole": "total_close_whole"}
    for y, g in list(lc.groupby("y")) + [("all", lc[lc["y"] >= from_year])]:
        row = {"y": y if y == "all" else int(y)}
        for k, col in variants.items():
            row[f"med_{k}"] = float(g[col].median())
            row[f"sum_{k}_btc"] = float(g[col].sum() / 1e8)
        out["lifecycle_by_close_year"].append(row)
    s = op[op["in_snapshot"] & op["fee"].notna()].assign(y=lambda d: d["open_time"].dt.year)
    for y, g in s.groupby("y"):
        row = {"y": int(y)}
        for k in RULES:
            row[f"med_open_{k}"] = float(g[f"open_{k}"].median())
            row[f"sum_open_{k}_btc"] = float(g[f"open_{k}"].sum() / 1e8)
        out["opens_by_year"].append(row)
    # where the rules can disagree at all: channels in transactions with more than one output that is not change
    sh = s[(s["n_chan"] > 1) | (s["n_other"] >= 2)]
    out["shared_tx_channels"] = {"n": len(sh), "share": float(len(sh) / len(s)),
                                 "med_open_a": float(sh["open_a"].median()), "med_open_c": float(sh["open_c"].median()),
                                 "med_open_b": float(sh["open_b"].median())}
    return out


def _json_default(o):
    if hasattr(o, "item"):
        o = o.item()
        return None if isinstance(o, float) and math.isnan(o) else o
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(type(o))


def clean(o):
    """NaN -> null, so lifecycle.json is valid JSON."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, float) and math.isnan(o):
        return None
    return o


def table(by: pd.DataFrame, sens: pd.DataFrame):
    n, f1 = style.n, style.f1
    sens = sens.set_index("y")
    rows = []
    for _, r in by.iterrows():
        y = r["y"]
        lab = "All" if y == "all" else str(int(y)) + (r"\textsuperscript{a}" if y == 2026 else "") + \
            (r"\textsuperscript{b}" if y != "all" and coverage(y) == "registry_limited" else "")
        if y == "all":
            rows.append("MIDRULE")
        rng = f"{n(sens.loc[y, 'med_c'])}--{n(sens.loc[y, 'med_a'])}"
        rows.append([lab, n(r["n"]), f1(100 * r["share_force"]), n(r["med_open"]), n(r["med_total_coop"]),
                     n(r["med_total_force"]), n(r["med_total"]), rng, n(r["p90_total"]),
                     f"{100 * r['med_share_cap']:.2f}", f1(100 * r["open_share_of_fees"]), f"{r['total_btc']:.2f}"])
    style.tex_table(
        "table_lifecycle",
        ["Closed", "Channels", "Force \\%", "Open", "Coop.", "Force", "All", "Range", "p90", "\\% cap.",
         "Opens \\%", "BTC"],
        rows, "lrrrrrrrrrrr",
        "What a public channel's life costs on-chain: funding share, close and required spends, by year of close "
        "(medians, sat unless stated).", "tab:lifecycle",
        note=r"\textsuperscript{a}Closes to 4 October 2026 (four weeks for outputs to be spent). Open: the channel's "
             r"share of its funding transaction's fee, split equally over the channel outputs of the transaction; "
             r"other outputs are the funder's change and pay nothing. Range: the median lifecycle if every output, "
             r"change included, pays for its own size, and with the equal split. Coop., Force: "
             r"lifecycles ending in a cooperative or a force close; force closes as in Table~II. \% cap.: median "
             r"lifecycle fees over capacity. Opens \%: opens' share of all lifecycle fees. BTC: all lifecycle fees. "
             r"Closes from 2019: our list of public channels holds few of the channels that closed in 2018.",
        wide=True)


def fig_lifecycle(by: pd.DataFrame, obi: pd.DataFrame):
    plt = style.plt
    fig, ax = plt.subplots(1, 2, figsize=(style.PAGE_W, 2.3), gridspec_kw={"width_ratios": [1.15, 1]})
    y = by["y"].astype(int).to_numpy()
    a = ax[0]
    series = [("med_total_force", "Force-close life", PALETTE["force"], "-"),
              ("med_total_coop", "Cooperative life", PALETTE["mutual"], "-"),
              ("med_open", "Open alone", PALETTE["unlisted"], "--")]
    for col, lab, c, ls in series:
        a.plot(y, by[col], color=c, ls=ls, marker="o", ms=4, mec="white", mew=1, label=lab)
        a.annotate(lab, (y[-1], by[col].iloc[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                   fontsize=6.5, color="#5E6977")
    a.set_yscale("log")
    a.set_ylim(100, 1e5)
    a.set_xlim(y[0] - 0.3, y[-1] + 1.6)
    a.set_title("(a) Lifecycle fees per channel (sat, median)", loc="left")
    a.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    a = ax[1]
    parts = [("open_btc", "Open", PALETTE["unlisted"]), ("close_btc", "Close", PALETTE["force"]),
             ("sweeps_btc", "Required spends", PALETTE["anchor"])]
    tot = by["total_btc"].to_numpy()
    bottom = np.zeros(len(y))
    for col, lab, c in parts:
        v = 100 * by[col].to_numpy() / tot
        a.bar(y, v, bottom=bottom, width=0.62, color=c, label=lab, edgecolor="white", linewidth=1)
        bottom += v
    a.set_ylim(0, 100)
    a.set_title("(b) Where lifecycle fees go (%)", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5, handlelength=1, borderaxespad=0)
    for a in ax:
        a.set_xticks(y)
        a.set_xticklabels([f"'{str(v)[2:]}" for v in y])
        a.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.8)
    style.save(fig, "fig8_lifecycle")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extract", action="store_true", help="read the chain lake and write the funding work tables")
    ap.add_argument("--work-dir", type=Path, default=config.WORK_DIR)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="write lifecycle.json, the table and the figure here instead of data/ and paper/")
    a = ap.parse_args()
    if a.extract:
        return extract(a.work_dir)
    if a.out_dir:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        style.FIGURES_DIR = style.TABLES_DIR = a.out_dir
    costs = importlib.import_module("03_costs")          # one copy of the spend categories (Table II)
    con = connect()
    H = manifest()["cutoff_height"]
    op, lc_all = build(con, a.work_dir, H, costs.CATEGORY)
    lc = lc_all[~lc_all["censored"]].copy()
    ck = checks(con, op, lc, H, a.work_dir)
    print(ck)

    obi = opens_by_year(op)
    cov = {}
    for lst in ("snapshot", "live"):
        g = obi[(obi["list"] == lst) & obi["y"].isin([2022, 2023])]
        cov[lst] = {"n_2022_23": int(g["n"].sum()), "cov_old_2022_23": float((g["cov_old"] * g["n"]).sum() / g["n"].sum()),
                    "cov_chain_2022_23": float((g["cov_chain"] * g["n"]).sum() / g["n"].sum())}
        g = obi[obi["list"] == lst]
        cov[lst].update({"n_all": int(g["n"].sum()), "cov_old_all": float((g["cov_old"] * g["n"]).sum() / g["n"].sum()),
                         "cov_chain_all": float((g["cov_chain"] * g["n"]).sum() / g["n"].sum())})
    # channels that closed in 2018 are mostly missing from our list, so the "all" row, the table and the
    # figure start in 2019; 2018 stays in lifecycle_by_close_year with its coverage flag
    lc_pub = lc[lc["y"] >= PUBLIC_YEAR]
    by_year = [{"y": int(y), "coverage": coverage(y), **summarize(g)} for y, g in lc.groupby("y")] + \
              [{"y": "all", "coverage": f"closes from {PUBLIC_YEAR}", **summarize(lc_pub)}]
    obi["coverage"] = obi["y"].map(coverage)
    by_regime = [{"open_regime": o, "close_regime": c, **summarize(g)}
                 for (o, c), g in lc.groupby(["open_regime", "close_regime"]) if len(g) >= 100]
    by_close_regime = [{"close_regime": c, **summarize(g)} for c, g in lc.groupby("close_regime")]
    by_fmt = [{"y": int(y), "fmt": f, "n": len(g), "med_total": float(g["total_a"].median()),
               "med_open": float(g["open_a"].median())}
              for (y, f), g in lc[lc["close_type"] == "force"].groupby(["y", "fmt"]) if len(g) >= 100]
    sens = sensitivity(lc, op, PUBLIC_YEAR)

    # calendar-year bill of public channels: opens funded in the year, closes (and their spends) in the year
    allfee = df(con, "SELECT year(to_timestamp(time)) AS y, sum(totalfee) / 1e8 AS all_fee_btc FROM bs GROUP BY y")
    s = obi[obi["list"] == "snapshot"][["y", "sum_open_a_btc", "funding_tx_fees_btc"]]
    cl = lc_all.groupby("y").agg(public_close_btc=("close", lambda x: x.sum() / 1e8),
                                 public_sweeps_btc=("sweeps", lambda x: x.sum() / 1e8)).reset_index()
    bill = s.merge(cl, on="y").merge(allfee, on="y")
    bill["open_share_of_public_bill"] = bill["sum_open_a_btc"] / (bill["sum_open_a_btc"] + bill["public_close_btc"]
                                                                  + bill["public_sweeps_btc"])
    bill["public_bill_share_of_all_fees"] = (bill["sum_open_a_btc"] + bill["public_close_btc"]
                                             + bill["public_sweeps_btc"]) / bill["all_fee_btc"]

    pd.set_option("display.width", 250)
    print(obi.round(3).to_string())
    print(pd.DataFrame(by_year).round(4).T.to_string())
    print(pd.DataFrame(by_regime)[["open_regime", "close_regime", "n", "med_total", "med_open", "med_close",
                                   "med_share_cap", "share_force"]].round(4).to_string())
    print(pd.DataFrame(sens["lifecycle_by_close_year"]).round(1).to_string())
    print(pd.DataFrame(sens["opens_by_year"]).round(1).to_string())
    print(sens["shared_tx_channels"])
    print(bill.round(4).to_string())

    shown = [r for r in by_year if r["y"] == "all" or r["y"] >= PUBLIC_YEAR]
    table(pd.DataFrame(shown), pd.DataFrame(sens["lifecycle_by_close_year"]))
    fig_lifecycle(pd.DataFrame(shown[:-1]), obi[obi["list"] == "snapshot"])
    obj = {
        "rules": RULES, "censor_blocks": SPEND_CENSOR_BLOCKS, "cutoff_height": H,
        "checks": ck, "coverage": cov, "opens_by_year": obi.to_dict(orient="records"),
        "lifecycle_by_close_year": by_year, "lifecycle_by_close_regime": by_close_regime,
        "lifecycle_by_regime": by_regime, "force_by_format": by_fmt, "sensitivity": sens,
        "bill_by_year": bill.to_dict(orient="records"),
        "n_lifecycles": int(len(lc_pub)), "n_lifecycles_incl_2018": int(len(lc)),
        "n_closes_censored": int(lc_all["censored"].sum()),
        "snapshot_channels_without_close": dict(zip(("open_at_cutoff", "closed_before_scan"), con.execute(f"""
            SELECT count(*) FILTER (NOT coalesce(fo.mempool_closing_date < {PRE_SCAN}, FALSE)),
                   count(*) FILTER (fo.mempool_closing_date < {PRE_SCAN})
            FROM fo LEFT JOIN c USING (chan_point) WHERE c.chan_point IS NULL""").fetchone())),
    }
    path = (a.out_dir or config.DATA_DIR) / "lifecycle.json"
    obj = clean(json.loads(json.dumps(obj, default=_json_default)))
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
