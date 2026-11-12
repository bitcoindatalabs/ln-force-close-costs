"""
02b_pre_anchor_2026.py (§IV, census): who makes 2026's pre-anchor public force closes? 

Pre-anchor (legacy) commitments were about a quarter of 2026's public force closes. This stage measures how
concentrated those closes are on a few nodes, how old the channels were, what the share is without the most active
nodes, and, for those nodes, their client label (paper 01's, from the snapshot) and whether their node_announcement
advertised anchor support (feature bits 20/21 option_anchor_outputs or 22/23 option_anchors_zero_fee_htlc_tx) at the
channel's funding date.

A close counts toward a node if the node is either endpoint (node1_pub or node2_pub). Gossip orders the two by key
(BOLT 7), so the order says nothing about who opened or who broadcast.

Two steps.

  --extract   (reads the Lightning lake once; under the lightning_lake lock, outside the busy windows; seconds)
      node_client_history.parquet rows (validity interval, feature_sig) for the endpoints of 2026's public force
      closes -> data/work/pre_anchor_endpoint_features.parquet
  (default)   snapshot work tables + that copy -> data/pre_anchor_2026.json (aggregates only: no node is named)

Usage:
  python src/02b_pre_anchor_2026.py --extract
  python src/02b_pre_anchor_2026.py
"""

import argparse
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import LN_PARQUET_DIR, SNAPSHOT_DIR, WORK_DIR  # noqa: E402
from fcc.io import connect, df, manifest, write_json  # noqa: E402

YEAR = 2026
TOP = 5
ANCHOR_BITS = {20, 21, 22, 23}
FEATURES = WORK_DIR / "pre_anchor_endpoint_features.parquet"


def closes(con, year: int) -> pd.DataFrame:
    d = df(con, f"""SELECT close_txid, capacity, close_height, funding_height, fmt, node1_pub, node2_pub,
                           epoch(close_time_utc)::BIGINT AS close_t
                    FROM fc WHERE listed AND year(close_time_utc) = {year}""")
    d["age_days"] = (d["close_height"] - d["funding_height"]) / 144.0
    return d


def extract():
    con = connect()
    d = closes(con, YEAR)
    pubs = pd.DataFrame({"pub_key": sorted(set(d["node1_pub"]) | set(d["node2_pub"]))})
    c = duckdb.connect()
    c.register("pubs", pubs)
    src = LN_PARQUET_DIR / "node_client_history.parquet"
    h = c.execute(f"""SELECT h.pub_key, h.valid_from, h.valid_to, h.feature_sig
                      FROM '{src}' h SEMI JOIN pubs USING (pub_key)""").df()
    h.to_parquet(FEATURES, index=False)
    print(f"{len(h):,} history rows for {h['pub_key'].nunique():,} of {len(pubs):,} endpoints -> {FEATURES}")


def _intervals(path_or_df) -> dict:
    h = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_parquet(path_or_df)
    h = h.copy()
    h["vf"] = pd.to_datetime(h["valid_from"], format="%Y%m%d", utc=True)
    h["vt"] = pd.to_datetime(h["valid_to"], format="%Y%m%d", utc=True) + pd.Timedelta(days=1)
    return {k: g for k, g in h.groupby("pub_key")}


def _at(hist: dict, pub: str, t: pd.Timestamp):
    """The history row covering t, else the nearest one (history starts 2023-05-08)."""
    g = hist.get(pub)
    if g is None:
        return None
    cov = g[(g["vf"] <= t) & (g["vt"] > t)]
    if len(cov):
        return cov.iloc[0]
    dist = pd.concat([(g["vf"] - t).abs(), (g["vt"] - t).abs()], axis=1).min(axis=1)
    return g.loc[dist.idxmin()]


def _label(row) -> str:
    return "Unknown" if row is None else row["label"]


def _anchor_status(row) -> str:
    if row is None:
        return "missing"
    sig = row["feature_sig"] or ""
    if not str(sig).strip():
        return "missing"
    bits = {int(b) for b in str(sig).split(",") if b.strip().isdigit()}
    return "anchor" if bits & ANCHOR_BITS else "no_anchor"


def analyze() -> dict:
    con = connect()
    d = closes(con, YEAR)
    bs = df(con, "SELECT height, time FROM bs")
    d = d.merge(bs.rename(columns={"height": "funding_height", "time": "funding_t"}), on="funding_height", how="left")
    d["t"] = pd.to_datetime(d["funding_t"].fillna(d["close_t"]), unit="s", utc=True)
    leg = d[d["fmt"] == "legacy"]
    nodes = pd.concat([leg["node1_pub"], leg["node2_pub"]]).value_counts()
    top = list(nodes.head(TOP).index)
    in_top = d["node1_pub"].isin(top) | d["node2_pub"].isin(top)
    leg_top = leg[in_top.loc[leg.index]]
    leg_rest = leg[~in_top.loc[leg.index]]
    rest = d[~in_top]

    labels = _intervals(pd.read_parquet(SNAPSHOT_DIR / "client_history.parquet").rename(columns={"client": "label"}))
    feats = _intervals(FEATURES) if FEATURES.exists() else {}
    top_nodes = []
    for pk in top:
        rows = leg[(leg["node1_pub"] == pk) | (leg["node2_pub"] == pk)]
        lab = pd.Series([_label(_at(labels, pk, t)) for t in rows["t"]]).mode().iloc[0]
        st = pd.Series([_anchor_status(_at(feats, pk, t)) for t in rows["t"]]).value_counts().to_dict() if feats else {}
        top_nodes.append({"closes": int(len(rows)), "client": lab, "anchor_feature_status": st,
                          "median_age_days": float(rows["age_days"].median())})
    status_of = lambda s: max(s, key=s.get) if s else None   # noqa: E731
    ldk_no_anchor = sum(1 for n in top_nodes if n["client"] == "LDK" and status_of(n["anchor_feature_status"]) == "no_anchor")

    # endpoints of every pre-anchor close: one side without anchor bits (the format needs both sides)
    one_lacks = None
    if feats:
        s1 = [_anchor_status(_at(feats, a, t)) for a, t in zip(leg["node1_pub"], leg["t"])]
        s2 = [_anchor_status(_at(feats, b, t)) for b, t in zip(leg["node2_pub"], leg["t"])]
        one_lacks = sum({x, y} == {"anchor", "no_anchor"} for x, y in zip(s1, s2)) / len(leg)

    return {
        "source": "src/02b_pre_anchor_2026.py: snapshot work tables (force_closes), client_history, and a copy of "
                  "the Lightning lake's node_client_history feature bits",
        "cutoff_height": manifest()["cutoff_height"], "year": YEAR,
        "public_force_closes": int(len(d)), "pre_anchor": int(len(leg)),
        "pre_anchor_share": len(leg) / len(d),
        "pre_anchor_nodes": int(len(nodes)),
        "top_n": TOP,
        "top_nodes_share_of_pre_anchor": len(leg_top) / len(leg),
        "top_nodes_pre_anchor_median_age_days": float(leg_top["age_days"].median()),
        "top_nodes": top_nodes,
        "top_nodes_ldk_without_anchor_bits": ldk_no_anchor,
        "without_top_nodes": {"public_force_closes": int(len(rest)), "pre_anchor": int(len(leg_rest)),
                              "pre_anchor_share": len(leg_rest) / len(rest),
                              "pre_anchor_median_age_days": float(leg_rest["age_days"].median()),
                              "pre_anchor_median_age_years": float(leg_rest["age_days"].median() / 365.25)},
        "pre_anchor_one_endpoint_without_anchor_bits_share": one_lacks,
        "anchor_baseline_top_nodes_share": (lambda a: float(
            (a["node1_pub"].isin(set(pd.concat([a["node1_pub"], a["node2_pub"]]).value_counts().head(TOP).index))
             | a["node2_pub"].isin(set(pd.concat([a["node1_pub"], a["node2_pub"]]).value_counts().head(TOP).index)))
            .mean()))(d[d["fmt"] == "anchor"]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extract", action="store_true", help="copy feature bits from the Lightning lake (locked)")
    a = ap.parse_args(argv)
    if a.extract:
        extract()
        return 0
    res = analyze()
    p = write_json("pre_anchor_2026.json", res)
    print(json.dumps(res, indent=1, default=str))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
