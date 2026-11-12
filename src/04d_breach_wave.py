"""
04d_breach_wave.py (§V-B): the September–October 2026 breach wave, from the snapshot and the chain lake.

The wave is defined as in 03_costs.py (`penalty_wave_format`): unlisted force closes confirmed 2026-09-28 to
2026-10-06 UTC whose outputs were spent by a revocation (justice) spend. For each revoked commitment this stage finds
the transaction that created the output it spends (paper 03's "funding transaction") and asks whether that
transaction is a splice: whether it spends a 2-of-2 multisig output, i.e. the output of an earlier open of the same
channel. It also follows the funding transaction's change output, to test whether the party that funded and spliced
the channel is the party that penalised the breach, and it repeats the splice test for the force closes of the same
commitment shape that were not penalised (unlisted anchor force closes since 2026-09-01 with the wave's 278-vB,
1,410–1,411-sat commitment). The method is cut down to what the paper reports; a cross-channel clustering (one cluster of the penalizer's coins) is not
repeated here.

Two steps.

  --extract   (reads the chain lake; run outside the cron busy windows, under the lightning_lake lock, ~1 min)
      data/work/breach_closes.parquet     wave + look-alike commitments: tx key, funding key/vout, output spenders
      data/work/breach_funding.parquet    their funding transactions: tx row, fee, inputs (with the outputs they spend)
      data/work/breach_funding_out.parquet  the funding transactions' outputs and their spenders
      data/work/breach_p0_out.parquet     outputs of the open that a splice spends (its change identifies the funder)
      data/work/breach_fwd1.parquet       inputs/outputs of the transactions that spent a funding transaction's change
      data/work/breach_fwd2_inputs.parquet  inputs of the transactions that spent those transactions' outputs
  (default)   (snapshot work tables + the extract; no lake)
      data/breach_wave.json               aggregates only: no transaction, address or key

Terms (one per channel): P0 the open that created the 2-of-2 output a splice spends; F the funding transaction (the
splice, or a plain open); R the revoked commitment; fwd1 the transaction that spent F's change output.

Usage:
  python src/04d_breach_wave.py --extract
  python src/04d_breach_wave.py
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CHAIN_LAKE_DIR, WORK_DIR  # noqa: E402
from fcc.io import connect, df, manifest, write_json  # noqa: E402

WAVE_FROM, WAVE_TO = "2026-09-28", "2026-10-06"       # 03_costs.py penalty_wave_format (BETWEEN, inclusive)
RELATED_FROM = "2026-09-01"
PARTS, LINKED = Path(CHAIN_LAKE_DIR) / "parts", Path(CHAIN_LAKE_DIR) / "linked"
TWO_OF_TWO = "(witness_items = 4 AND n_ecdsa_sigs = 2 AND witness_script_len = 71)"
W = WORK_DIR


# ---- chain lake helpers (tx_key = height << 15 | position) -----------------------------------------------------------
def _files(kind: str, heights) -> list:
    sub = LINKED if kind.endswith("_link") else PARTS
    starts = sorted({int(h) // 1000 * 1000 for h in heights})
    out = []
    for s in starts:
        hits = sorted(sub.glob(f"{kind}_{s}_*.parquet"))
        if len(hits) != 1:
            raise SystemExit(f"chain lake: expected one {kind} file for chunk {s}, found {len(hits)}")
        out.append(str(hits[0]))
    return out


def _by_keys(c, kind: str, keys, cols: str = "*") -> pd.DataFrame:
    keys = sorted({int(k) for k in keys if k is not None and k == k})
    if not keys:
        return pd.DataFrame()
    c.register("_k", pd.DataFrame({"k": keys}))
    out = c.execute(f"SELECT {cols} FROM read_parquet({_files(kind, [k >> 15 for k in keys])}) t "
                    f"SEMI JOIN _k ON t.tx_key = _k.k").df()
    c.unregister("_k")
    return out


def _bundle(c, keys, inputs: bool = True, outputs: bool = True) -> dict:
    """tx rows (+ fee), inputs (+ the outputs they spend) and outputs (+ their spenders) of a set of tx keys."""
    keys = sorted({int(k) for k in keys if k == k})
    t = _by_keys(c, "txs", keys, "tx_key, height, version, locktime, vsize, n_in, n_out, rbf_signal")
    t = t.merge(_by_keys(c, "txs_link", keys, "tx_key, fee"), on="tx_key", how="left")
    out = {"txs": t}
    if inputs:
        i = _by_keys(c, "inputs", keys, "tx_key, vin, prev_vout, sequence, witness_items, n_ecdsa_sigs, "
                                        "witness_script_len, lower(hex(witness_script)) AS witness_script")
        out["inputs"] = i.merge(_by_keys(c, "inputs_link", keys), on=["tx_key", "vin"], how="left")
    if outputs:
        o = _by_keys(c, "outputs", keys, "tx_key, vout, value, script_type")
        out["outputs"] = o.merge(_by_keys(c, "outputs_link", keys), on=["tx_key", "vout"], how="left")
    return out


def _sets(con) -> pd.DataFrame:
    """The wave's revoked commitments and the non-penalised look-alikes, from the snapshot work tables."""
    wave = df(con, f"""
        WITH w AS (SELECT DISTINCT close_txid FROM o JOIN fc USING (close_txid)
                   WHERE o.spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach') AND NOT fc.listed
                     AND fc.close_time_utc BETWEEN TIMESTAMP '{WAVE_FROM}' AND TIMESTAMP '{WAVE_TO}')
        SELECT 'wave' AS kind, fc.close_txid, fc.chan_point, fc.capacity, fc.close_height, fc.close_time_utc,
               max(o.n) FILTER (o.spend_type = 'to_local_revocation_breach') AS pen_vout,
               sum(o.value_sat) FILTER (o.spend_type = 'to_local_revocation_breach') AS penalised_sat,
               max(o.blocks_after_close) FILTER (o.spend_type = 'to_local_revocation_breach') AS pen_blocks,
               max(o.n) FILTER (o.spend_type = 'to_remote_sweep') AS rem_vout,
               max(o.n) FILTER (o.spend_type = 'to_local_delayed_sweep') AS tl_vout,
               max(o.blocks_after_close) FILTER (o.spend_type = 'to_local_delayed_sweep') AS tl_blocks
        FROM w JOIN fc USING (close_txid) JOIN o USING (close_txid) GROUP BY ALL""")
    rel = df(con, f"""
        SELECT 'related' AS kind, fc.close_txid, fc.chan_point, fc.capacity, fc.close_height, fc.close_time_utc,
               NULL::INTEGER AS pen_vout, NULL::BIGINT AS penalised_sat, NULL::BIGINT AS pen_blocks,
               max(o.n) FILTER (o.spend_type = 'to_remote_sweep') AS rem_vout,
               max(o.n) FILTER (o.spend_type = 'to_local_delayed_sweep') AS tl_vout,
               max(o.blocks_after_close) FILTER (o.spend_type = 'to_local_delayed_sweep') AS tl_blocks,
               bool_or(o.spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach')) AS breached
        FROM fc LEFT JOIN o USING (close_txid)
        WHERE NOT fc.listed AND fc.fmt = 'anchor' AND fc.fee_sat IN (1410, 1411) AND fc.vsize = 278
          AND fc.close_time_utc >= TIMESTAMP '{RELATED_FROM}'
        GROUP BY ALL""")
    rel = rel[~rel["close_txid"].isin(wave["close_txid"])].drop(columns="breached")
    return pd.concat([wave, rel], ignore_index=True)


def extract():
    t0 = pd.Timestamp.now()
    sets = _sets(connect())
    nw = int((sets["kind"] == "wave").sum())
    print(f"wave commitments {nw}, look-alikes {int((sets['kind'] == 'related').sum())}")
    c = duckdb.connect()
    c.execute("SET threads=4")

    # R: tx keys, the input (funding outpoint) and the spenders of every output
    want = pd.DataFrame({"txid": [bytes.fromhex(t) for t in sets["close_txid"]],
                         "height": sets["close_height"].astype("int32")})
    c.register("want", want)
    keys = c.execute(f"""SELECT lower(hex(t.txid)) AS close_txid, t.tx_key AS close_key
                         FROM read_parquet({_files('txs', want['height'])}) t
                         JOIN want w ON t.txid = w.txid AND t.height = w.height""").df()
    c.unregister("want")
    if len(keys) != len(sets):
        raise SystemExit(f"chain lake: found {len(keys)} of {len(sets)} commitments")
    sets = sets.merge(keys, on="close_txid")
    ck = sets["close_key"]
    ci = _by_keys(c, "inputs", ck, "tx_key, vin, lower(hex(witness_script)) AS commit_witness_script")
    ci = ci.merge(_by_keys(c, "inputs_link", ck), on=["tx_key", "vin"])
    ci = ci.rename(columns={"tx_key": "close_key", "prev_tx_key": "funding_key"})
    sets = sets.merge(ci[["close_key", "funding_key", "commit_witness_script"]], on="close_key")
    ol = _by_keys(c, "outputs_link", ck)
    for col, v in (("pen_key", "pen_vout"), ("rem_key", "rem_vout"), ("tl_key", "tl_vout")):
        m = ol.rename(columns={"tx_key": "close_key", "vout": v, "spend_tx_key": col})[["close_key", v, col]]
        sets = sets.merge(m.astype({v: "float64"}), on=["close_key", v], how="left")
    sets.to_parquet(W / "breach_closes.parquet", index=False)

    # F: the funding transactions
    f = _bundle(c, sets["funding_key"])
    f["inputs"].to_parquet(W / "breach_funding.parquet", index=False)
    f["txs"].to_parquet(W / "breach_funding_txs.parquet", index=False)
    f["outputs"].to_parquet(W / "breach_funding_out.parquet", index=False)

    # P0: the opens that splices spend (their outputs and spenders: the open's change)
    fi = f["inputs"]
    p0 = fi[(fi["witness_items"] == 4) & (fi["n_ecdsa_sigs"] == 2) & (fi["prev_script_type"] == "p2wsh")]
    pb = _bundle(c, p0["prev_tx_key"], inputs=False)
    pb["outputs"].to_parquet(W / "breach_p0_out.parquet", index=False)
    pb["txs"].to_parquet(W / "breach_p0_txs.parquet", index=False)

    # fwd1: the spends of F's non-channel outputs; fwd2: the inputs of the spends of fwd1's outputs
    fo = f["outputs"].merge(sets[["funding_key"]].drop_duplicates(), left_on="tx_key", right_on="funding_key")
    chan = set(zip(sets["funding_key"], sets["chan_point"].str.split(":").str[1].astype(int)))
    chg = fo[[(k, v) not in chan for k, v in zip(fo["tx_key"], fo["vout"])]]
    b1 = _bundle(c, chg["spend_tx_key"].dropna())
    b1["inputs"].to_parquet(W / "breach_fwd1_inputs.parquet", index=False)
    b1["outputs"].to_parquet(W / "breach_fwd1_outputs.parquet", index=False)
    b1["txs"].to_parquet(W / "breach_fwd1_txs.parquet", index=False)
    b2 = _by_keys(c, "inputs_link", b1["outputs"]["spend_tx_key"].dropna())
    b2.to_parquet(W / "breach_fwd2_inputs.parquet", index=False)
    print(f"funding txs {f['txs']['tx_key'].nunique()}, opens {pb['txs']['tx_key'].nunique()}, "
          f"fwd1 {b1['txs']['tx_key'].nunique()}, fwd2 inputs {len(b2)}; {(pd.Timestamp.now() - t0).seconds} s")


def q(s, ps=(0.1, 0.5, 0.9)) -> dict:
    s = pd.Series(s, dtype="float64").dropna()
    if s.empty:
        return {"n": 0}
    return {f"p{int(p * 100)}": float(s.quantile(p)) for p in ps} | {"min": float(s.min()), "max": float(s.max()),
                                                                       "n": int(len(s))}


def _keys(script: str) -> set:
    """The two 33-byte keys of a 2-of-2 witness script (OP_2 <33> k1 <33> k2 OP_2 OP_CHECKMULTISIG, 71 bytes)."""
    return {script[4:70], script[72:138]} if isinstance(script, str) and len(script) == 142 else set()


def analyze() -> dict:
    x = pd.read_parquet(W / "breach_closes.parquet")
    fi = pd.read_parquet(W / "breach_funding.parquet")
    ft = pd.read_parquet(W / "breach_funding_txs.parquet").set_index("tx_key")
    fo = pd.read_parquet(W / "breach_funding_out.parquet")
    p0o = pd.read_parquet(W / "breach_p0_out.parquet")
    f1i = pd.read_parquet(W / "breach_fwd1_inputs.parquet")
    f1o = pd.read_parquet(W / "breach_fwd1_outputs.parquet")
    f2i = pd.read_parquet(W / "breach_fwd2_inputs.parquet")
    cut = manifest()["cutoff_height"]

    shared = fi[(fi["witness_items"] == 4) & (fi["n_ecdsa_sigs"] == 2) & (fi["prev_script_type"] == "p2wsh")]
    shared = shared.drop_duplicates("tx_key")
    x = x.merge(shared[["tx_key", "prev_tx_key", "witness_script", "sequence"]]
                .rename(columns={"tx_key": "funding_key", "prev_tx_key": "p0_key", "witness_script": "splice_ws",
                                 "sequence": "shared_seq"}), on="funding_key", how="left")
    x["splice"] = x["p0_key"].notna()
    x["fh"] = x["funding_key"].astype("int64") // 32768
    x["p0h"] = x["p0_key"].map(lambda k: int(k) >> 15 if k == k else np.nan)
    x["open_to_splice"] = x["fh"] - x["p0h"]
    x["splice_to_close"] = x["close_height"] - x["fh"]
    x["same_keys"] = [len(_keys(a) & _keys(b)) == 2 for a, b in zip(x["splice_ws"], x["commit_witness_script"])]

    # F's change: its non-channel outputs, and who spent them
    chan = set(zip(x["funding_key"], x["chan_point"].str.split(":").str[1].astype(int)))
    chg = fo[[(k, v) not in chan for k, v in zip(fo["tx_key"], fo["vout"])]]
    chg = chg.sort_values(["tx_key", "vout"]).drop_duplicates("tx_key")      # one change output per funding tx
    x = x.merge(chg[["tx_key", "spend_tx_key"]].rename(columns={"tx_key": "funding_key", "spend_tx_key": "fwd1_key"}),
                on="funding_key", how="left")

    for col in ("pen_key", "rem_key", "tl_key", "fwd1_key", "p0_key"):
        x[col] = pd.to_numeric(x[col], errors="coerce").astype("float64")    # nullable ints -> NaN-aware floats
    # the study window stops at the cutoff: spends confirmed after it are treated as not (yet) made
    for col in ("pen_key", "rem_key", "tl_key", "fwd1_key"):
        x.loc[x[col] // 32768 > cut, col] = np.nan
    f2i = f2i[f2i["tx_key"] // 32768 <= cut]
    f1o = f1o.assign(spend_tx_key=f1o["spend_tx_key"].where(f1o["spend_tx_key"] // 32768 <= cut))

    # P0's change (its non-P2WSH outputs) is the opener's: does the splice, or the spend of the splice's change, use it?
    p0chg = set(zip(p0o.loc[p0o["script_type"] != "p2wsh", "tx_key"].astype("int64"),
                    p0o.loc[p0o["script_type"] != "p2wsh", "vout"].astype("int64")))
    fwi = fi[fi["prev_script_type"] != "p2wsh"].dropna(subset=["prev_tx_key"])
    f_spends_p0 = {int(k) for k, a, b in zip(fwi["tx_key"], fwi["prev_tx_key"], fwi["prev_vout"]) if (int(a), int(b)) in p0chg}
    g = f1i.dropna(subset=["prev_tx_key"])
    fwd1_spends_p0 = {int(k) for k, a, b in zip(g["tx_key"], g["prev_tx_key"], g["prev_vout"]) if (int(a), int(b)) in p0chg}
    x["same_wallet_open_splice"] = x["funding_key"].isin(f_spends_p0) | x["fwd1_key"].isin(fwd1_spends_p0)

    # funder = penalizer: (a) the splice change is spent in the to_remote sweep, which only the penalizer can sign;
    # (b) a P2WPKH output of the change's spend is later spent together with the same channel's penalty output
    x["link_to_remote"] = x["fwd1_key"].notna() & (x["fwd1_key"] == x["rem_key"])
    # the penalty transactions have one output, so "spends the penalty output" = "spends the penalty transaction"
    cosp = {}
    for k, a in zip(f2i["tx_key"], f2i["prev_tx_key"]):
        if a == a:
            cosp.setdefault(int(k), set()).add(int(a))
    small = f1o[(f1o["script_type"] == "p2wpkh") & f1o["spend_tx_key"].notna()]
    spent_with = {}
    for k, s in zip(small["tx_key"].astype("int64"), small["spend_tx_key"].astype("int64")):
        spent_with.setdefault(k, set()).update(cosp.get(s, set()))
    x["link_penalty"] = [(pk == pk) and (fk == fk) and int(pk) in spent_with.get(int(fk), set())
                         for fk, pk in zip(x["fwd1_key"], x["pen_key"])]
    x["funder_is_penalizer"] = x["link_to_remote"] | x["link_penalty"]

    w = x[x["kind"] == "wave"]
    ws = w[w["splice"]]
    other = w[~w["splice"]]
    rel = x[x["kind"] == "related"]
    rs = rel[rel["splice"]]
    in_week = rs["close_time_utc"].between(pd.Timestamp(WAVE_FROM, tz="UTC"), pd.Timestamp(WAVE_TO, tz="UTC"))
    out = {
        "source": "src/04d_breach_wave.py: snapshot work tables + chain lake",
        "cutoff_height": cut,
        "first": str(w["close_time_utc"].min()), "last": str(w["close_time_utc"].max()),
        "penalised_commitments": int(len(w)),
        "btc_all": float(w["penalised_sat"].sum() / 1e8),
        "channels": int(len(ws)),                                   # the wave proper: splice-then-breach channels
        "btc_without_outlier": float(ws["penalised_sat"].sum() / 1e8),
        "splices": {
            "n": int(len(ws)),
            "same_funding_keys_before_and_after": int(ws["same_keys"].sum()),
            "shared_input_sequence_0": int((ws["shared_seq"] == 0).sum()),
            "blocks_open_to_splice": q(ws["open_to_splice"]),
            "blocks_splice_to_revoked_close": q(ws["splice_to_close"]),
            "splice_to_close_le_4_blocks": int((ws["splice_to_close"] <= 4).sum()),
            "open_and_splice_same_wallet": int(ws["same_wallet_open_splice"].sum()),
            "funder_is_penalizer_to_remote_sweep": int(ws["link_to_remote"].sum()),
            "funder_is_penalizer_change_with_penalty": int(ws["link_penalty"].sum()),
            "funder_is_penalizer_any": int(ws["funder_is_penalizer"].sum()),
            "penalty_blocks_after_close": q(ws["pen_blocks"]),
            "capacity_sat": q(ws["capacity"]),
        },
        "other_penalised": {
            "n": int(len(other)),
            "splice": False,
            "channel_age_days": [float((r.close_height - r.fh) / 144) for r in other.itertuples()],
            "btc": float(other["penalised_sat"].sum() / 1e8),
        },
        "look_alikes": {
            "definition": f"unlisted anchor force closes since {RELATED_FROM} with the wave's commitment shape "
                          "(278 vB, fee 1,410-1,411 sat), not penalised",
            "n": int(len(rel)),
            "splices": int(len(rs)),
            "splices_in_wave_week": int(in_week.sum()),
            "splice_blocks_splice_to_close": q(rs["splice_to_close"]),
            "splice_to_local_swept_after_blocks": q(rs["tl_blocks"]),
            "funder_on_counterparty_side_to_remote_sweep": int(rs["link_to_remote"].sum()),
            "funder_on_broadcaster_side_to_local_sweep": int((rs["fwd1_key"].notna() & (rs["fwd1_key"] == rs["tl_key"])).sum()),
        },
    }
    wk = int(in_week.sum())
    out["share_of_pattern_closes_revoked_in_wave_week"] = len(ws) / (len(ws) + wk) if len(ws) + wk else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extract", action="store_true", help="read the chain lake (under the lightning_lake lock)")
    a = ap.parse_args(argv)
    if a.extract:
        extract()
        return 0
    res = analyze()
    p = write_json("breach_wave.json", res)
    print(json.dumps({k: res[k] for k in ("penalised_commitments", "channels", "btc_without_outlier")}),
          json.dumps(res["splices"])[:600], json.dumps(res["look_alikes"])[:400], sep="\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
