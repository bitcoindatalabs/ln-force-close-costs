"""
08_htlc_deadlines.py (P9, agenda idea I2): how close to their deadline HTLCs were resolved on-chain, and what happened
when one payment was resolved on-chain in more than one channel.

Every HTLC output of a force close (data/work/outputs.parquet) is joined to the chain lake (bitcoin_chain_extractor.py,
CHAIN_LAKE_DIR/parts): the spending transaction's nLockTime from `txs`, and the revealed witness script of the input
that spent the HTLC from `inputs`. Two facts come out of that (BOLT 3):

  * cltv_expiry. An HTLC-timeout transaction's nLockTime is the HTLC's expiry (its pre-signed signature commits to it,
    so batched timeouts share one expiry). A received-HTLC script carries `<cltv_expiry> OP_CHECKLOCKTIMEVERIFY`, so
    HTLC-success transactions and the remote's timeout claims give the expiry too. Offered HTLCs claimed by the remote
    with the preimage carry no expiry.
  * the payment hash. Both script kinds contain RIPEMD160(payment_hash). Before PTLCs every hop of a payment, and every
    part of a multi-part payment, uses the same hash, so the same hash in two force closes is one payment.

Measured:
  1. Deadlines: blocks from expiry to confirmation of HTLC-timeout transactions (by year, by the block median fee rate
     at expiry, pre-anchor vs anchor); margin left when HTLC-success transactions confirmed. The yardstick is the
     cltv_expiry_delta nodes advertise in gossip (time_lock_delta: 10th percentile 40 blocks, median 40 in 2023 and 80
     in 2025-26), the margin a forwarding node has between its outgoing HTLC's expiry and its incoming one's.
  2. Cascades: force closes whose HTLC shares a payment hash with an HTLC of another force close.
  3. Split resolutions: one payment resolved with the preimage in one channel and by timeout in another. Legs are
     paired by amount (within 1% or 1,000 sat); the upstream leg is the one with the larger amount (it includes the
     forwarding fee), else the later expiry. Upstream timed out while downstream was paid: the node between them paid
     out and was not paid. Upstream paid while downstream timed out: the downstream node revealed the preimage and did
     not claim its HTLC in time. Either way one party lost the HTLC amount. Pairs in which a channel used a
     to_self_delay of at most STAGED_MAX_TO_SELF_DELAY blocks are a staged test, not organic losses (seven
     pairs in 2026, fresh 10k-20k-sat channels, chained HTLC-timeout change), and are
     reported apart. The paper's figures start in 2021, the window in which every pair was checked by hand.
  4. The yardstick itself: the cltv_expiry_delta (time_lock_delta) that public channels advertise, from the daily
     gossip snapshots of the Lightning lake (gossip_daily/policies, from May 2023), by year. Read under the
     lightning_lake lock, like the chain-lake join.

Only aggregates are released (no txid, hash, node or address). Spend statistics use closes at least
SPEND_CENSOR_BLOCKS before the cutoff; HTLC outputs still unspent at the cutoff are not in these numbers.

Writes data/work/htlc_spends.parquet (local), data/htlc.json, paper/figures/fig7_htlc_deadlines.{pdf,png}.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (CHAIN_LAKE_DIR, CUTOFF_UTC, LN_PARQUET_DIR, PALETTE, SCAN_START_HEIGHT, SPEND_CENSOR_BLOCKS,  # noqa: E402
                    WORK_DIR)
from fcc.io import connect, df, manifest, write_json  # noqa: E402
from fcc.style import COL_W, plt, save  # noqa: E402

LAKE_PARTS = CHAIN_LAKE_DIR / "parts"
MARGINS = (40, 80, 144)                 # blocks: gossip time_lock_delta p10 (and 2023 median), 2025-26 median, high
FEE_BUCKETS = [(0, 5, "<5"), (5, 20, "5-20"), (20, 50, "20-50"), (50, 1e9, "50+")]   # sat/vB, block median at expiry
PREIMAGE = ("htlc_success_2nd_stage", "htlc_remote_preimage_claim")
TIMEOUT = ("htlc_timeout_2nd_stage", "htlc_remote_timeout_claim")
STAGED_MAX_TO_SELF_DELAY = 6            # blocks: the staged 2026 test pairs; no other pair has one below 40
SPLIT_FROM_YEAR = 2021                  # every pair from 2021 was checked against the chain by hand
GOSSIP_POLICIES = LN_PARQUET_DIR / "gossip_daily" / "policies"

RX_EXPIRY = re.compile(rb"\x75\x03(...)\xb1\x75\xac", re.S)     # OP_DROP <3-byte cltv_expiry> OP_CLTV OP_DROP OP_CHECKSIG
RX_PAYHASH = re.compile(rb"\xa9\x14(.{20})\x88", re.S)         # OP_HASH160 <RIPEMD160(payment_hash)> OP_EQUALVERIFY
# early implementations (2018-19) also wrote HTLCs with a cltv_expiry of 0 or a few hundred blocks (pushed as OP_0 or
# 1-2 bytes): not a block height, so no deadline
RX_EXPIRY_SMALL = re.compile(rb"\x75(?:\x00|\x01(.)|\x02(..))\xb1\x75\xac", re.S)
ZERO_EXPIRY_SPENDS = {}


def extract(con) -> pd.DataFrame:
    """HTLC spends joined to the chain lake: locktime of the spending tx and the revealed HTLC script."""
    if not any(LAKE_PARTS.glob("txs_*.parquet")):
        sys.exit(f"chain lake not found at {LAKE_PARTS}")
    con.execute("""
        CREATE TEMP TABLE hs AS
        SELECT o.close_txid, o.listed, o.fmt, o.close_height, o.n, o.value_sat, o.spend_txid, o.spend_height,
               o.spend_type, o.spend_n_inputs, o.spend_vsize, o.spend_fee_sat, unhex(o.spend_txid) AS spend_txid_b,
               -- inputs.prev_txid8: first 8 bytes of the display-order txid as a signed big-endian int64
               (('0x' || substr(o.close_txid, 1, 16))::UBIGINT::HUGEINT
                 - CASE WHEN ('0x' || substr(o.close_txid, 1, 16))::UBIGINT >= 9223372036854775808::UBIGINT
                        THEN 18446744073709551616::HUGEINT ELSE 0 END)::BIGINT AS close_txid8
        FROM o WHERE o.spend_type LIKE 'htlc%'""")
    lo, hi = con.execute("SELECT min(spend_height), max(spend_height) FROM hs").fetchone()
    con.execute(f"""
        CREATE TEMP TABLE ht AS
        SELECT tx_key, height, txid, locktime FROM read_parquet('{LAKE_PARTS}/txs_*.parquet')
        WHERE height BETWEEN {lo} AND {hi} AND txid IN (SELECT spend_txid_b FROM hs)""")
    con.execute(f"""
        CREATE TEMP TABLE hi AS
        SELECT tx_key, prev_txid8, prev_vout, witness_script FROM read_parquet('{LAKE_PARTS}/inputs_*.parquet')
        WHERE height BETWEEN {lo} AND {hi} AND tx_key IN (SELECT tx_key FROM ht)""")
    h = df(con, """
        SELECT hs.* EXCLUDE (spend_txid_b, close_txid8), ht.locktime, hi.witness_script
        FROM hs
        LEFT JOIN ht ON ht.txid = hs.spend_txid_b AND ht.height = hs.spend_height
        LEFT JOIN hi ON hi.tx_key = ht.tx_key AND hi.prev_txid8 = hs.close_txid8 AND hi.prev_vout = hs.n""")
    missing = int(h["witness_script"].isna().sum())
    print(f"  HTLC spends {len(h):,}; without a lake match {missing}")
    if missing:
        sys.exit("some HTLC spends have no chain-lake match; is the lake complete for the window?")

    def parse(ws):
        m = RX_EXPIRY.search(ws)
        p = RX_PAYHASH.search(ws)
        if m:
            e = int.from_bytes(m.group(1), "little")
        else:
            k = RX_EXPIRY_SMALL.search(ws)
            e = int.from_bytes(k.group(1) or k.group(2) or b"", "little") if k else np.nan
        return (e, p.group(1).hex() if p else None)

    parsed = [parse(ws) for ws in h["witness_script"]]
    h["script_expiry"] = [p[0] for p in parsed]
    h["payment_h160"] = [p[1] for p in parsed]
    h["expiry"] = np.where(h["spend_type"] == "htlc_timeout_2nd_stage", h["locktime"], h["script_expiry"])
    h["conf_minus_expiry"] = h["spend_height"] - h["expiry"]
    h["close_minus_expiry"] = h["close_height"] - h["expiry"]
    h["path"] = np.select([h["spend_type"].isin(PREIMAGE), h["spend_type"].isin(TIMEOUT)], ["preimage", "timeout"],
                          "breach")
    # HTLCs whose cltv_expiry is not a block height of the window (0 or a few hundred: a received script with
    # "OP_0 OP_CLTV" or a 1-2 byte push, or an HTLC-timeout tx with a small nLockTime): spends of closes from 2018 to
    # April 2019, an early-implementation artefact. They have no deadline, so they leave the HTLC analysis; the count
    # is reported.
    zero = (h["spend_type"].isin(("htlc_success_2nd_stage", "htlc_remote_timeout_claim"))
            & (h["script_expiry"] < SCAN_START_HEIGHT)) | \
           ((h["spend_type"] == "htlc_timeout_2nd_stage") & (h["locktime"] < SCAN_START_HEIGHT))
    ZERO_EXPIRY_SPENDS.update({"spends": int(zero.sum()), "closes": int(h.loc[zero, "close_txid"].nunique()),
                               "max_close_height": int(h.loc[zero, "close_height"].max()) if zero.any() else None})
    h = h[~zero].reset_index(drop=True)
    # sanity: received-HTLC spends must carry an expiry in their script, HTLC-timeout txs a height locktime
    bad = h[(h["spend_type"].isin(("htlc_success_2nd_stage", "htlc_remote_timeout_claim")) & h["script_expiry"].isna())
            | ((h["spend_type"] == "htlc_timeout_2nd_stage") & ~h["locktime"].between(1, 499_999_999))]
    if len(bad):
        bad.to_parquet(WORK_DIR / "htlc_spends_failing_checks.parquet")
        sys.exit(f"{len(bad)} HTLC spends fail the script checks")
    h = h.drop(columns=["witness_script"])
    if h["payment_h160"].isna().any():
        sys.exit("HTLC script without a payment hash")
    h.to_parquet(WORK_DIR / "htlc_spends.parquet")
    return h


def bucket(x):
    if pd.isna(x):                      # expiry outside the block-stats window
        return None
    for lo, hi, name in FEE_BUCKETS:
        if lo <= x < hi:
            return name
    return None


def deadlines(con, h: pd.DataFrame, H: int) -> dict:
    con.register("hh", h[h["close_height"] <= H - SPEND_CENSOR_BLOCKS])
    t = df(con, """
        SELECT hh.*, year(fc.close_time_utc) AS y, bs.p50 AS fee_at_expiry
        FROM hh JOIN fc USING (close_txid) LEFT JOIN bs ON bs.height = hh.expiry::BIGINT""")
    t["fee_bucket"] = t["fee_at_expiry"].map(bucket)

    def late(g):
        x = g["conf_minus_expiry"]
        out = {"n": int(len(g)), "median_blocks": float(x.median()), "p90_blocks": float(x.quantile(0.9))}
        for m in MARGINS:
            out[f"share_gt{m}"] = float((x > m).mean())
        return out

    to = t[t["spend_type"] == "htlc_timeout_2nd_stage"]
    ready = to[to["close_minus_expiry"] <= 0]           # commitment already confirmed at expiry: the tx's own delay
    res = {
        "timeout_by_year": {int(y): late(g) for y, g in to.groupby("y")},
        "timeout_commitment_confirmed_before_expiry_share": float((to["close_minus_expiry"] <= 0).mean()),
        "timeout_ready_by_fee": {b: late(g) for b, g in ready.groupby("fee_bucket")},
        "timeout_ready_by_fee_format": {f"{b}|{f}": late(g) for (b, f), g in
                                        ready[ready["fmt"].isin(("anchor", "legacy"))].groupby(["fee_bucket", "fmt"])},
        "timeout_ready_by_year_format": {f"{y}|{f}": late(g) for (y, f), g in
                                         ready[ready["fmt"].isin(("anchor", "legacy"))].groupby(["y", "fmt"])},
        "timeout_late_close_by_fee": {b: late(g) for b, g in to[to["close_minus_expiry"] > 0].groupby("fee_bucket")},
    }
    su = t[t["spend_type"] == "htlc_success_2nd_stage"]
    res["success_by_year"] = {int(y): {"n": int(len(g)), "median_margin_blocks": float((-g["conf_minus_expiry"]).median()),
                                       "at_or_after_expiry": int((g["conf_minus_expiry"] >= 0).sum()),
                                       "share_at_or_after_expiry": float((g["conf_minus_expiry"] >= 0).mean())}
                              for y, g in su.groupby("y")}
    rt = t[t["spend_type"] == "htlc_remote_timeout_claim"]
    res["remote_timeout_by_year"] = {int(y): {"n": int(len(g)), "median_blocks": float(g["conf_minus_expiry"].median()),
                                              "p90_blocks": float(g["conf_minus_expiry"].quantile(0.9))}
                                     for y, g in rt.groupby("y")}
    res["counts"] = t["spend_type"].value_counts().to_dict()
    return res


def cascades(con, h: pd.DataFrame) -> dict:
    hh = h[h["path"] != "breach"]
    per_hash = hh.groupby("payment_h160")["close_txid"].nunique()
    multi = set(per_hash[per_hash > 1].index)
    closes = hh.groupby("close_txid").agg(close_height=("close_height", "first"))
    in_multi = set(hh.loc[hh["payment_h160"].isin(multi), "close_txid"])
    closes["shared"] = closes.index.isin(in_multi)
    con.register("cl", closes.reset_index())
    y = df(con, "SELECT year(fc.close_time_utc) AS y, count(*) AS htlc_closes, sum(cl.shared::INT) AS shared "
                "FROM cl JOIN fc USING (close_txid) GROUP BY y ORDER BY y")
    span = hh[hh["payment_h160"].isin(multi)].groupby("payment_h160")["close_height"].agg(lambda s: s.max() - s.min())
    sizes = per_hash[per_hash > 1].value_counts().sort_index()
    paths = hh[hh["payment_h160"].isin(multi)].groupby("payment_h160")["path"].agg(set)
    all_timeout = int(sum(p == {"timeout"} for p in paths))
    return {"by_year": y.to_dict(orient="records"),
            "hashes_all_legs_timed_out": all_timeout, "hashes_all_legs_timed_out_share": all_timeout / len(multi),
            "hashes_all_legs_preimage": int(sum(p == {"preimage"} for p in paths)),
            "hashes_in_2plus_closes": int(len(multi)), "closes_involved": int(len(in_multi)),
            "htlc_closes": int(len(closes)), "max_closes_per_hash": int(per_hash.max()),
            "hashes_by_close_count": {int(k): int(v) for k, v in sizes.items()},
            "median_span_blocks": float(span.median()), "p90_span_blocks": float(span.quantile(0.9))}


def split_resolutions(con, h: pd.DataFrame) -> dict:
    hh = h[h["path"] != "breach"]
    both = hh.groupby("payment_h160").filter(lambda g: g["close_txid"].nunique() > 1 and g["path"].nunique() == 2)
    fc = df(con, "SELECT close_txid, node1_pub, node2_pub, year(close_time_utc) AS y, capacity FROM fc")
    tsd = df(con, "SELECT close_txid, min(csv_delay) AS to_self_delay FROM o WHERE csv_delay IS NOT NULL GROUP BY 1")
    both = both.merge(fc, on="close_txid", how="left").merge(tsd, on="close_txid", how="left")
    legs = []
    for _, g in both.groupby("payment_h160"):
        pre, tmo = g[g["path"] == "preimage"], g[g["path"] == "timeout"]
        for _, p in pre.iterrows():
            c = tmo[tmo["close_txid"] != p["close_txid"]]
            if c.empty:
                continue
            t = c.iloc[(c["value_sat"] - p["value_sat"]).abs().argsort().iloc[0]]
            if abs(t["value_sat"] - p["value_sat"]) > max(0.01 * p["value_sat"], 1000):
                continue
            if t["value_sat"] != p["value_sat"]:
                up = "timeout" if t["value_sat"] > p["value_sat"] else "preimage"
            elif pd.notna(p["expiry"]) and pd.notna(t["expiry"]) and p["expiry"] != t["expiry"]:
                up = "timeout" if t["expiry"] > p["expiry"] else "preimage"
            else:
                up = "unknown"
            listed = bool(p["listed"] and t["listed"])
            shared = listed and bool({p["node1_pub"], p["node2_pub"]} & {t["node1_pub"], t["node2_pub"]})
            # pairing filters (from a hand check of all 65 pairs of the 2021-2026 snapshot):
            #  1. a forward needs the upstream (larger) leg to expire after the downstream leg: when both expiries are
            #     known, the upstream expiry must be later (pairs with an unknown expiry gap are kept);
            #  2. a retry, not two hops: the timeout leg confirmed before the other channel was even closed (the
            #     payer re-sent the same payment hash over another channel after the first attempt timed out).
            #  "Both legs offered by the same node" (the other retry sign) needs the broadcaster's identity, which the
            #  chain does not give; on the hand-checked set rule 2 removes every retry pair.
            up_leg, dn_leg = (t, p) if up == "timeout" else (p, t)
            gap = (up_leg["expiry"] - dn_leg["expiry"]) if up != "unknown" and pd.notna(p["expiry"]) and pd.notna(t["expiry"]) \
                else np.nan
            legs.append({"h": p["payment_h160"], "y": int(min(p["y"], t["y"])), "sat": int(min(p["value_sat"], t["value_sat"])),
                         "upstream": up, "both_listed": listed, "shared_node": shared,
                         "preimage_first": bool(p["spend_height"] < t["spend_height"]),
                         "gap_blocks": int(t["spend_height"] - p["spend_height"]),
                         "expiry_gap": gap,
                         "fails_expiry_gap": bool(pd.notna(gap) and gap <= 0),
                         "retry": bool(t["spend_height"] < p["close_height"]),
                         "min_to_self_delay": float(np.nanmin([p["to_self_delay"], t["to_self_delay"], np.inf])),
                         "max_capacity": int(max(p["capacity"], t["capacity"])),
                         # upstream timed out, then the downstream preimage claim: blocks between the two
                         "preimage_after_timeout_blocks": int(p["spend_height"] - t["spend_height"])})
    L = pd.DataFrame(legs)
    who = {"timeout": "forwarder_lost", "preimage": "downstream_lost", "unknown": "unclear"}
    L["loser"] = L["upstream"].map(who)
    L["staged"] = L["min_to_self_delay"] <= STAGED_MAX_TO_SELF_DELAY
    paired = ~L["fails_expiry_gap"] & ~L["retry"]
    keep = paired & ~L["staged"]
    out = summarize_pairs(L[keep])                                   # 2018-2026, organic
    out["before_filters"] = summarize_pairs(L)
    # the paper's window: 2021-2026, where every pair was checked against the chain by hand
    w = L["y"] >= SPLIT_FROM_YEAR
    out["from_2021"] = {"before_filters": summarize_pairs(L[w]),
                        "after_filters_incl_staged": summarize_pairs(L[paired & w]),
                        "after_filters": summarize_pairs(L[keep & w])}
    st = L[paired & L["staged"]]
    out["staged"] = {"pairs": int(len(st)), "payments": int(st["h"].nunique()), "btc": float(st["sat"].sum() / 1e8),
                     "years": sorted(int(y) for y in st["y"].unique()),
                     "max_capacity_sat": q_(st["max_capacity"]),
                     "preimage_after_timeout_blocks": q_(st["preimage_after_timeout_blocks"]),
                     "max_to_self_delay": STAGED_MAX_TO_SELF_DELAY,
                     "rule": f"a channel of the pair used a to_self_delay of at most {STAGED_MAX_TO_SELF_DELAY} blocks"}
    out["filters"] = {"pairs_before": int(len(L)), "fails_expiry_gap": int(L["fails_expiry_gap"].sum()),
                      "retry": int(L["retry"].sum()), "both": int((L["fails_expiry_gap"] & L["retry"]).sum()),
                      "staged": int((paired & L["staged"]).sum()),
                      "pairs_after": int(keep.sum()),
                      "rules": ["upstream expiry after downstream expiry when both are known",
                                "timeout leg not confirmed before the other channel's close (a retry)",
                                out["staged"]["rule"] + " (a staged test, reported apart)"]}
    return out


def q_(s) -> dict:
    s = pd.Series(s, dtype="float64").dropna()
    return {"min": float(s.min()), "median": float(s.median()), "max": float(s.max())} if len(s) else {}


def advertised_deltas() -> dict:
    """time_lock_delta (cltv_expiry_delta) of every channel direction in each daily gossip snapshot, by year."""
    import duckdb
    files = sorted(GOSSIP_POLICIES.glob("*.parquet"))
    if not files:
        sys.exit(f"gossip policies not found at {GOSSIP_POLICIES}")
    c = duckdb.connect()
    c.execute("SET threads=6")
    cut = CUTOFF_UTC[:10].replace("-", "")            # snapshots before the cutoff day only
    src = f"read_parquet('{GOSSIP_POLICIES}/*.parquet', filename=true, union_by_name=true)"
    flt = f"time_lock_delta IS NOT NULL AND split_part(filename, 'date=', 2) < '{cut}'"
    q = f"""SELECT substr(split_part(filename, 'date=', 2), 1, 4)::INT AS y, count(*) AS policies,
                   quantile_cont(time_lock_delta, 0.1) AS p10, median(time_lock_delta) AS median,
                   mode(time_lock_delta) AS mode
            FROM {src} WHERE {flt} GROUP BY ALL ORDER BY y"""
    y = c.execute(q).df()
    tot = c.execute(f"""SELECT count(*), quantile_cont(time_lock_delta, 0.1), median(time_lock_delta),
                               min(split_part(filename, 'date=', 2)), max(split_part(filename, 'date=', 2))
                        FROM {src} WHERE {flt}""").fetchone()
    return {"source": "Lightning lake gossip_daily/policies (one row per channel direction per daily snapshot)",
            "first_snapshot": tot[3][:8], "last_snapshot": tot[4][:8], "policies": int(tot[0]),
            "p10": float(tot[1]), "median": float(tot[2]),
            "by_year": {int(r.y): {"policies": int(r.policies), "p10": float(r.p10), "median": float(r.median),
                                   "mode": int(r.mode)} for r in y.itertuples()}}


def summarize_pairs(L: pd.DataFrame) -> dict:
    return {
        "payments": int(L["h"].nunique()), "legs": int(len(L)), "btc": float(L["sat"].sum() / 1e8),
        "max_leg_btc": float(L["sat"].max() / 1e8), "median_leg_sat": float(L["sat"].median()),
        "both_listed_legs": int(L["both_listed"].sum()), "both_listed_sharing_a_node": int(L["shared_node"].sum()),
        "by_loser": {k: {"legs": int(len(g)), "payments": int(g["h"].nunique()), "btc": float(g["sat"].sum() / 1e8),
                         "preimage_on_chain_first": float(g["preimage_first"].mean()),
                         "median_gap_blocks": float(g["gap_blocks"].median())} for k, g in L.groupby("loser")},
        "by_year": {int(y): {"legs": int(len(g)), "payments": int(g["h"].nunique()), "btc": float(g["sat"].sum() / 1e8)}
                    for y, g in L.groupby("y")},
    }


def figure(d: dict, c: dict):
    fig, (a, b) = plt.subplots(1, 2, figsize=(2 * COL_W, 2.2), gridspec_kw={"width_ratios": [1.2, 1]})
    names = [x[2] for x in FEE_BUCKETS]
    xs = np.arange(len(names))
    for i, (fmt, lab) in enumerate((("legacy", "Pre-anchor"), ("anchor", "Anchor"))):
        v = [100 * d["timeout_ready_by_fee_format"].get(f"{nm}|{fmt}", {}).get("share_gt40", np.nan) for nm in names]
        bars = a.bar(xs + (i - 0.5) * 0.38, v, 0.36, color=PALETTE[fmt], label=lab)
        for r, val in zip(bars, v):
            a.text(r.get_x() + r.get_width() / 2, val + 0.8, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5,
                   color=PALETTE["ink_mark"])
    a.set_xticks(xs, [f"{nm}" for nm in names])
    a.set_xlabel("Block median fee rate at expiry (sat/vB)")
    a.set_ylabel("% confirmed > 40 blocks after expiry")
    a.set_title("(a) HTLC-timeout transactions", loc="left")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper left")
    yy = [r["y"] for r in c["by_year"]]
    share = [100 * r["shared"] / r["htlc_closes"] for r in c["by_year"]]
    bars = b.bar(yy, share, 0.6, color=PALETTE["force"])
    for r, val in zip(bars, share):
        b.text(r.get_x() + r.get_width() / 2, val + 0.6, f"{val:.0f}", ha="center", va="bottom", fontsize=6.5,
               color=PALETTE["ink_mark"])
    b.set_xticks(yy, [str(y) for y in yy])
    b.grid(axis="x", visible=False)
    b.set_ylabel("% of force closes with HTLCs")
    b.set_title("(b) Payment also on-chain in another close", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7_htlc_deadlines")


def main() -> int:
    con = connect(threads=10)
    H = manifest()["cutoff_height"]
    h = extract(con)
    d = deadlines(con, h, H)
    c = cascades(con, h)
    s = split_resolutions(con, h)
    g = advertised_deltas()
    if g["p10"] != MARGINS[0]:
        print(f"WARNING: the tight margin ({MARGINS[0]}) is no longer the p10 of advertised deltas ({g['p10']})")
    figure(d, c)
    print({k: d[k] for k in ("timeout_commitment_confirmed_before_expiry_share",)})
    print({k: v for k, v in c.items() if k != "by_year"})
    print({k: v for k, v in s.items() if k not in ("by_year",)})
    write_json("htlc.json", {"margins_blocks": list(MARGINS), "fee_buckets": [x[2] for x in FEE_BUCKETS],
                             "excluded_zero_cltv_expiry": ZERO_EXPIRY_SPENDS,
                             "advertised_cltv_expiry_delta": g,
                             "deadlines": d, "cascades": c, "split_resolutions": s})
    return 0


if __name__ == "__main__":
    sys.exit(main())
