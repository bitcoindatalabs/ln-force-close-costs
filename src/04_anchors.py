"""
04_anchors.py (RQ3): what anchor outputs are used for, and where their value goes.

  * fate of every P2WSH anchor of every force close (public or not): owner bump (spent by its owner in the commitment's
    block), owner cleanup (later), third party (anyone, from 16 blocks after the close), unspent;
  * delay of third-party sweeps by the close's year;
  * owner fee bumps: share of anchor force closes with a same-block owner child, by year and by pending HTLCs, and
    whether the child raised the package fee rate;
  * third-party sweep transactions chain-wide (every anchor-script spend with an empty signature, including anchors of
    commitments we never saw): anchors per transaction, fee rate, and the split of the anchors' 330 sat between miners
    (fees) and sweepers (what is left), by sweep year.
Simple taproot channels' anchors (P2TR, 330 sat) use a different script; they are counted, not classified.

Writes data/anchors.json, data/work/third_party_sweeps.parquet, paper/figures/fig4_anchors.pdf,
paper/tables/table_anchors.tex.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ANCHOR_CSV_BLOCKS, ANCHOR_SWEEP_INPUT_VB, ANCHOR_VALUE_SAT, PALETTE,  # noqa: E402
                    SPEND_CENSOR_BLOCKS, WORK_DIR)
from fcc.io import connect, df, manifest, write_json  # noqa: E402
from fcc.style import PAGE_W, f1, n, plt, save, tex_table  # noqa: E402


def fate(con, H: int) -> pd.DataFrame:
    return df(con, f"""
        SELECT year(fc.close_time_utc) AS y, o.listed,
               count(*) AS anchors,
               count(*) FILTER (o.spend_type = 'anchor_sweep_owner' AND o.blocks_after_close = 0) AS owner_bump,
               count(*) FILTER (o.spend_type = 'anchor_sweep_owner' AND o.blocks_after_close > 0) AS owner_cleanup,
               count(*) FILTER (o.spend_type = 'anchor_sweep_anyone' AND o.blocks_after_close <= {ANCHOR_CSV_BLOCKS + 1}) AS third_fast,
               count(*) FILTER (o.spend_type = 'anchor_sweep_anyone' AND o.blocks_after_close > {ANCHOR_CSV_BLOCKS + 1}) AS third_late,
               count(*) FILTER (o.spend_txid IS NULL) AS unspent,
               count(*) FILTER (o.spend_txid IS NOT NULL AND o.spend_type NOT IN ('anchor_sweep_owner', 'anchor_sweep_anyone')) AS other,
               median(o.blocks_after_close) FILTER (o.spend_type = 'anchor_sweep_anyone') AS third_med_delay,
               quantile_cont(o.blocks_after_close, 0.9) FILTER (o.spend_type = 'anchor_sweep_anyone') AS third_p90_delay,
               avg((o.blocks_after_close = {ANCHOR_CSV_BLOCKS})::INT) FILTER (o.spend_type = 'anchor_sweep_anyone') AS third_at_16
        FROM o JOIN fc USING (close_txid)
        WHERE o.role = 'anchor' AND o.script_type = 'witness_v0_scripthash'
          AND fc.close_height <= {H - SPEND_CENSOR_BLOCKS}
        GROUP BY ALL ORDER BY y, listed""")


def bumps(con, H: int) -> pd.DataFrame:
    """Anchor force closes (public, version 2) with an owner child in the same block, and whether it raised the rate."""
    return df(con, f"""
        WITH b AS (
            SELECT close_txid, max(spend_fee_sat::DOUBLE / spend_vsize) AS child_fr
            FROM o WHERE listed AND spend_type = 'anchor_sweep_owner' AND blocks_after_close = 0 GROUP BY close_txid)
        SELECT year(fc.close_time_utc) AS y, fc.has_htlc_like_outputs AS htlc, count(*) AS closes,
               count(b.close_txid) AS bumped, count(*) FILTER (b.child_fr > fc.feerate_sat_vb) AS raised
        FROM fc LEFT JOIN b USING (close_txid)
        WHERE fc.listed AND fc.fmt = 'anchor' AND fc.close_height <= {H - SPEND_CENSOR_BLOCKS}
        GROUP BY ALL ORDER BY y, htlc""")


def sweeps(con) -> pd.DataFrame:
    """One row per third-party sweep transaction, chain-wide."""
    con.execute("""CREATE OR REPLACE TEMP VIEW anyone AS
        SELECT spend_txid, spend_height, spend_time_utc, spend_fee_sat, spend_vsize, spend_n_inputs FROM o
        WHERE spend_type = 'anchor_sweep_anyone'
        UNION ALL
        SELECT spend_txid, spend_height, spend_time_utc, spend_fee_sat, spend_vsize, spend_n_inputs FROM us
        WHERE output_role = 'anchor_unarmed' AND spend_type = 'anchor_sweep_anyone'""")
    return df(con, """SELECT spend_txid, any_value(spend_height) AS height, any_value(spend_time_utc) AS t,
                             any_value(spend_fee_sat) AS fee, any_value(spend_vsize) AS vsize,
                             any_value(spend_n_inputs) AS n_inputs, count(*) AS anchors
                      FROM anyone GROUP BY spend_txid""")


def sweep_economics(tx: pd.DataFrame) -> pd.DataFrame:
    tx = tx.copy()
    tx["y"] = tx["t"].dt.year
    tx["value"] = ANCHOR_VALUE_SAT * tx["anchors"]
    tx["miners"] = np.minimum(tx["fee"], tx["value"])
    tx["sweepers"] = tx["value"] - tx["miners"]
    tx["fr"] = tx["fee"] / tx["vsize"]
    be = ANCHOR_VALUE_SAT / ANCHOR_SWEEP_INPUT_VB
    g = tx.groupby("y")
    return pd.DataFrame({
        "txs": g.size(), "anchors": g["anchors"].sum(), "per_tx": g["anchors"].mean(),
        "share_only_anchors": g.apply(lambda x: (x["n_inputs"] == x["anchors"]).mean(), include_groups=False),
        "med_fr": g["fr"].median(), "p90_fr": g["fr"].quantile(0.9),
        "share_above_breakeven": g["fr"].apply(lambda x: (x > be).mean()),
        "value_btc": g["value"].sum() / 1e8, "miners_btc": g["miners"].sum() / 1e8,
        "sweepers_btc": g["sweepers"].sum() / 1e8,
    }).reset_index().assign(miners_share=lambda d: d["miners_btc"] / d["value_btc"])


def fig_anchors(fa: pd.DataFrame, ec: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 2.55))
    a = ax[0]
    f = fa.groupby("y")[["owner_bump", "owner_cleanup", "third_fast", "third_late", "unspent"]].sum()
    tot = f.sum(axis=1)
    parts = [("owner_bump", "Owner, fee bump", "#1a9e6e"), ("owner_cleanup", "Owner, cleanup", PALETTE["owner"]),
             ("third_fast", "Third party, at block 16–17", PALETTE["third_party"]),
             ("third_late", "Third party, later", "#F0B49C")]
    bottom = np.zeros(len(f))
    for k, lab, col in parts:
        v = 100 * f[k] / tot
        a.bar(f.index, v, bottom=bottom, width=0.62, color=col, label=lab, edgecolor="white", linewidth=1)
        bottom += v.to_numpy()
    a.set_ylim(0, 100)
    a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2, fontsize=6.3, handlelength=1, borderaxespad=0,
             columnspacing=0.8)
    a.set_xticks(f.index)
    a.set_xticklabels([f"'{str(v)[2:]}" for v in f.index])
    a.set_title("(a) Who spent each anchor (%)", loc="left")
    a = ax[1]
    d = fa[fa["third_med_delay"].notna()].groupby("y")[["third_med_delay", "third_p90_delay"]].max()
    a.plot(d.index, d["third_med_delay"], color=PALETTE["third_party"], marker="o", ms=4, mec="white", mew=1,
           label="Median")
    a.plot(d.index, d["third_p90_delay"], color=PALETTE["third_party"], lw=0.8, alpha=0.6, label="90th percentile")
    a.axhline(ANCHOR_CSV_BLOCKS, color=PALETTE["ink_mark"], lw=0.7)
    a.annotate("timelock: 16 blocks", (2022.55, ANCHOR_CSV_BLOCKS), xytext=(0, 3), textcoords="offset points",
               fontsize=6.5, color="#5E6977")
    a.set_yscale("log")
    a.set_yticks([18, 144, 1008, 4320, 12960])
    a.set_yticklabels(["3 hours", "1 day", "1 week", "1 month", "3 months"])
    a.minorticks_off()
    a.set_ylim(10, 40000)
    a.set_title("(b) Wait for a third-party sweep", loc="left")
    a.annotate("90th pct.", (2024, d.loc[2024, "third_p90_delay"]), xytext=(0, 4), textcoords="offset points",
               ha="center", fontsize=6.5, color="#5E6977")
    a.annotate("median", (2024, d.loc[2024, "third_med_delay"]), xytext=(4, -9), textcoords="offset points",
               ha="left", fontsize=6.5, color="#5E6977")
    a.set_xticks(d.index)
    a.set_xticklabels([f"'{str(v)[2:]}" for v in d.index])
    a = ax[2]
    e = ec[ec["y"] >= 2022]
    a.bar(e["y"], 100 * e["miners_share"], width=0.62, color=PALETTE["miners"], label="To miners (fees)",
          edgecolor="white", linewidth=1)
    a.bar(e["y"], 100 * (1 - e["miners_share"]), bottom=100 * e["miners_share"], width=0.62,
          color=PALETTE["third_party"], label="Kept by sweepers", edgecolor="white", linewidth=1)
    for _, r in e.iterrows():
        a.annotate(f"{100 * r['miners_share']:.0f}", (r["y"], 100 * r["miners_share"] / 2), ha="center",
                   va="center", fontsize=6.8, color="white")
    a.set_ylim(0, 100)
    a.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=2, fontsize=6.3, handlelength=1, borderaxespad=0,
             columnspacing=0.8)
    a.set_title("(c) Where swept anchor value went (%)", loc="left")
    a.set_xticks(e["y"])
    a.set_xticklabels([f"'{str(v)[2:]}" for v in e["y"]])
    for a in ax:
        a.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig4_anchors")


def main() -> int:
    con = connect()
    H = manifest()["cutoff_height"]
    fa = fate(con, H)
    bu = bumps(con, H)
    tx = sweeps(con)
    tx.to_parquet(WORK_DIR / "third_party_sweeps.parquet", index=False)
    ec = sweep_economics(tx)
    taproot = con.execute("""SELECT count(*), count(spend_txid) FROM o
                             WHERE role = 'anchor' AND script_type = 'witness_v1_taproot'""").fetchone()
    taproot_y = df(con, f"""SELECT year(fc.close_time_utc) AS y, count(*) AS anchors, count(o.spend_txid) AS spent,
                                   count(*) FILTER (o.spend_txid IS NULL AND fc.close_height <= {H - SPEND_CENSOR_BLOCKS}) AS unspent_old,
                                   sum(o.value_sat) / 1e8 AS btc
                            FROM o JOIN fc USING (close_txid)
                            WHERE o.role = 'anchor' AND o.script_type = 'witness_v1_taproot' GROUP BY y ORDER BY y""")
    print(fa.to_string())
    print(bu.to_string())
    print(ec.round(3).to_string())
    fig_anchors(fa, ec)

    f = fa.groupby("y").sum(numeric_only=True)
    b = bu.groupby("y")[["closes", "bumped", "raised"]].sum()
    d = fa.groupby("y")[["third_med_delay"]].max()
    e = ec.set_index("y")
    rows = []
    for y in f.index:
        r, tot = f.loc[y], f.loc[y, "anchors"]
        ee = e.loc[y] if y in e.index else None
        rows.append([str(y) + (r"\textsuperscript{a}" if y == 2026 else ""), n(tot),
                     f1(100 * (r["owner_bump"] + r["owner_cleanup"]) / tot),
                     f1(100 * (r["third_fast"] + r["third_late"]) / tot), f1(100 * r["unspent"] / tot),
                     f1(100 * b.loc[y, "bumped"] / b.loc[y, "closes"]),
                     n(d.loc[y, "third_med_delay"]) if pd.notna(d.loc[y, "third_med_delay"]) else "--",
                     n(ee["txs"]) if ee is not None else "--", f1(ee["per_tx"]) if ee is not None else "--",
                     f"{ee['med_fr']:.2f}" if ee is not None else "--",
                     f1(100 * ee["miners_share"]) if ee is not None else "--"])
    tex_table("table_anchors", ["Year", "Anchors", "Owner \\%", "Third \\%", "Unspent \\%", "Bumped \\%",
                                "Delay", "Sweep txs", "Per tx", "sat/vB", "Miners \\%"],
              rows, "lrrrrrrrrrr",
              "What anchor outputs are used for. Left: anchors by the year of their force close. Right: third-party "
              "sweep transactions by the year they confirmed.", "tab:anchors",
              note=r"\textsuperscript{a}Closes to 4 October 2026. Anchors: P2WSH anchors of all force closes, public or "
                   r"not. Bumped: share of public anchor force closes with an owner child in the commitment's block. "
                   r"Delay: median blocks from the close to a third-party sweep. Sweep txs: transactions spending "
                   r"anchors with an empty signature (chain-wide). sat/vB: their median fee rate. Miners: share of the "
                   r"anchors' value paid as fees. Break-even fee rate: " + f"{ANCHOR_VALUE_SAT / ANCHOR_SWEEP_INPUT_VB:.2f}"
                   + r" sat/vB.", wide=True)

    write_json("anchors.json", {
        "fate": fa.to_dict(orient="records"), "bumps": bu.to_dict(orient="records"),
        "sweep_economics": ec.to_dict(orient="records"),
        "taproot_anchors": {"outputs": taproot[0], "spent": taproot[1]},
        "taproot_anchors_by_year": taproot_y.to_dict(orient="records"),
        "breakeven_sat_vb": ANCHOR_VALUE_SAT / ANCHOR_SWEEP_INPUT_VB,
        "totals": {"sweep_txs": int(len(tx)), "anchors_swept": int(tx["anchors"].sum()),
                   "value_btc": float(ec["value_btc"].sum()), "miners_btc": float(ec["miners_btc"].sum()),
                   "sweepers_btc": float(ec["sweepers_btc"].sum())},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
