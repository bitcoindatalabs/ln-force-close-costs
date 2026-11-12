"""
03_costs.py (RQ2): what a force close costs, in three currencies: fees (sat), time (days until funds return) and
block space (vB), against a cooperative close of the same year.

Cost account per public force close (closes at least SPEND_CENSOR_BLOCKS before the cutoff, so their outputs had time
to be spent). Only transactions the force close requires are counted:
  commitment       the commitment transaction's fee
  bump             an owner's anchor spend confirmed in the same block as the commitment (a CPFP child): full fee
  htlc             second-stage HTLC transactions and claims of HTLC outputs on the counterparty's commitment
  sweeps           the broadcaster's timelocked to_local output; the to_remote output of anchor channels (1-block CSV)
  anchor_cleanup   an owner's anchor spend after the commitment confirmed
Spends that batch several inputs are split equally per input. Not counted: spends of a pre-anchor channel's to_remote
output (a plain wallet output, like a cooperative close's outputs), third-party anchor sweeps (paid from the anchor's
330 sat, reported as value lost), and penalty transactions (reported separately).

Writes data/work/close_costs.parquet, data/costs.json, paper/figures/fig3_costs.pdf, paper/tables/table_costs.tex.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANCHOR_VALUE_SAT, PALETTE, PUBLIC_FROM, SPEND_CENSOR_BLOCKS, WORK_DIR, coverage  # noqa: E402

PUBLIC_YEAR = int(PUBLIC_FROM[:4])     # per-close statistics of public channels are shown from 2019
from fcc.io import connect, df, manifest, write_json  # noqa: E402
from fcc.style import PAGE_W, f1, n, plt, save, tex_table  # noqa: E402

CATEGORY = """CASE
    WHEN spend_type = 'anchor_sweep_owner' AND blocks_after_close = 0 THEN 'bump'
    WHEN spend_type = 'anchor_sweep_owner' THEN 'anchor_cleanup'
    WHEN spend_type = 'anchor_sweep_anyone' THEN 'third_party_anchor'
    WHEN spend_type IN ('htlc_timeout_2nd_stage', 'htlc_success_2nd_stage', 'htlc_remote_timeout_claim',
                        'htlc_remote_preimage_claim') THEN 'htlc'
    WHEN spend_type IN ('to_local_delayed_sweep', 'to_remote_sweep') THEN 'sweeps'
    WHEN spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach') THEN 'penalty'
    WHEN spend_type = 'to_remote_p2wpkh_spend' THEN 'wallet'
    WHEN spend_type = 'no_witness' THEN 'bump'          -- pay-to-anchor child of a zero-fee commitment
    ELSE 'other' END"""
PARTY = ["commitment", "bump", "htlc", "sweeps", "anchor_cleanup"]


def close_costs(con, cutoff: int) -> pd.DataFrame:
    con.execute(f"""CREATE OR REPLACE TEMP TABLE alloc AS
        SELECT close_txid, {CATEGORY} AS cat, spend_type, value_sat, blocks_after_close, csv_delay,
               spend_fee_sat * CASE WHEN spend_type IN ('anchor_sweep_owner', 'no_witness') AND blocks_after_close = 0
                                    THEN 1.0 ELSE 1.0 / spend_n_inputs END AS fee,
               spend_vsize * CASE WHEN spend_type IN ('anchor_sweep_owner', 'no_witness') AND blocks_after_close = 0
                                  THEN 1.0 ELSE 1.0 / spend_n_inputs END AS vb
        FROM o WHERE listed AND spend_txid IS NOT NULL""")
    piv = df(con, """SELECT close_txid,
            sum(fee) FILTER (cat = 'bump') AS bump, sum(fee) FILTER (cat = 'htlc') AS htlc,
            sum(fee) FILTER (cat = 'sweeps') AS sweeps, sum(fee) FILTER (cat = 'anchor_cleanup') AS anchor_cleanup,
            sum(fee) FILTER (cat = 'third_party_anchor') AS third_party_fee,
            count(*) FILTER (cat = 'third_party_anchor') AS anchors_lost,
            count(*) FILTER (cat = 'penalty') AS penalty_inputs,
            sum(vb) FILTER (cat NOT IN ('wallet', 'penalty')) AS spend_vb,
            max(blocks_after_close) FILTER (spend_type = 'to_local_delayed_sweep') AS to_local_wait,
            max(csv_delay) FILTER (spend_type = 'to_local_delayed_sweep') AS to_self_delay,
            sum(value_sat) FILTER (spend_type = 'to_local_delayed_sweep') AS to_local_sat,
            count(*) FILTER (cat = 'htlc') AS htlc_outputs
        FROM alloc GROUP BY close_txid""")
    fc = df(con, f"""SELECT close_txid, close_height, year(close_time_utc) AS y, fmt, capacity, fee_sat AS commitment,
                            vsize AS commitment_vb, has_htlc_like_outputs, n_anchor_outputs
                     FROM fc WHERE listed AND close_height <= {cutoff - SPEND_CENSOR_BLOCKS}""")
    t = fc.merge(piv, on="close_txid", how="left")
    for k in ["bump", "htlc", "sweeps", "anchor_cleanup", "third_party_fee", "anchors_lost", "penalty_inputs",
              "spend_vb", "htlc_outputs"]:
        t[k] = t[k].fillna(0)
    t["party_fee"] = t[PARTY].sum(axis=1)
    t["total_vb"] = t["commitment_vb"] + t["spend_vb"]
    t["anchor_value_lost"] = ANCHOR_VALUE_SAT * t["anchors_lost"]
    t["cost_share_of_capacity"] = t["party_fee"] / t["capacity"]
    return t


def mutual(con, cutoff: int) -> pd.DataFrame:
    return df(con, f"""SELECT year(close_time_utc) AS y, count(*) AS n, median(fee_sat) AS fee_med, avg(fee_sat) AS fee_mean,
                              quantile_cont(fee_sat, 0.25) AS fee_q25, quantile_cont(fee_sat, 0.75) AS fee_q75,
                              median(vsize) AS vb_med, sum(fee_sat) AS fee_total
                       FROM c WHERE close_type = 'mutual' AND close_height <= {cutoff - SPEND_CENSOR_BLOCKS}
                       GROUP BY y ORDER BY y""")


def summarize(t: pd.DataFrame, mu: pd.DataFrame) -> pd.DataFrame:
    g = t.groupby("y")
    s = pd.DataFrame({
        "n": g.size(),
        "med_party_fee": g["party_fee"].median(), "p90_party_fee": g["party_fee"].quantile(0.9),
        "q25_party_fee": g["party_fee"].quantile(0.25), "q75_party_fee": g["party_fee"].quantile(0.75),
        "mean_party_fee": g["party_fee"].mean(), "total_party_btc": g["party_fee"].sum() / 1e8,
        "med_commitment": g["commitment"].median(),
        "med_vb": g["total_vb"].median(),
        "share_over_1pct": g["cost_share_of_capacity"].apply(lambda x: (x > 0.01).mean()),
        "med_to_local_days": g["to_local_wait"].median() / 144, "p90_to_local_days": g["to_local_wait"].quantile(0.9) / 144,
        "share_htlc": g["htlc_outputs"].apply(lambda x: (x > 0).mean()),
        "btc_days_locked": g.apply(lambda x: (x["to_local_sat"].fillna(0) * x["to_local_wait"].fillna(0) / 144).sum() / 1e8,
                                   include_groups=False),
        "anchor_value_lost_btc": g["anchor_value_lost"].sum() / 1e8,
    })
    for k in ["commitment", "bump", "htlc", "sweeps", "anchor_cleanup"]:
        s[f"sum_{k}"] = g[k].sum()
    s = s.reset_index().merge(mu.rename(columns={"n": "mutual_n"}), on="y")
    s["ratio_med"] = s["med_party_fee"] / s["fee_med"]
    s["ratio_vb"] = s["med_vb"] / s["vb_med"]
    s["coverage"] = s["y"].map(coverage)
    return s


def by_format(t: pd.DataFrame) -> pd.DataFrame:
    return t[t["fmt"].isin(["legacy", "anchor"])].groupby(["y", "fmt"]).agg(
        n=("party_fee", "size"), med_party_fee=("party_fee", "median"), med_commitment=("commitment", "median"),
        med_vb=("total_vb", "median"), share_bumped=("bump", lambda x: (x > 0).mean())).reset_index()


def fig_costs(s: pd.DataFrame, t: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 2.3), gridspec_kw={"width_ratios": [1, 1.1, 1]})
    y = s["y"].to_numpy()
    a = ax[0]
    g = t.groupby("y")["party_fee"]
    a.fill_between(y, g.quantile(0.25), g.quantile(0.75), color=PALETTE["force"], alpha=0.12, lw=0)
    a.plot(y, s["med_party_fee"], color=PALETTE["force"], marker="o", ms=4, mec="white", mew=1, label="Force close")
    a.plot(y, s["fee_med"], color=PALETTE["mutual"], marker="o", ms=4, mec="white", mew=1, label="Cooperative close")
    a.set_yscale("log")
    a.set_title("(a) Fees per close (sat, median)", loc="left")
    a.legend(loc="upper right", fontsize=6.8, handlelength=1.2)
    a.set_ylim(100, 1e5)
    a = ax[1]
    comp = ["commitment", "sweeps", "htlc", "bump", "anchor_cleanup"]
    names = {"commitment": "Commitment", "sweeps": "Sweeps", "htlc": "HTLC resolution", "bump": "Fee bump",
             "anchor_cleanup": "Anchor cleanup"}
    cols = {"commitment": "#D4603A", "sweeps": "#2a78d6", "htlc": "#4a3aa7", "bump": "#1a9e6e",
            "anchor_cleanup": "#C4CBD3"}
    tot = s[[f"sum_{k}" for k in comp]].sum(axis=1)
    bottom = np.zeros(len(y))
    for k in comp:
        v = 100 * s[f"sum_{k}"] / tot
        a.bar(y, v, bottom=bottom, width=0.62, color=cols[k], label=names[k], edgecolor="white", linewidth=1)
        bottom += v.to_numpy()
    a.set_ylim(0, 100)
    a.set_title("(b) Where force-close fees go (%)", loc="left")
    a.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5, handlelength=1, borderaxespad=0)
    a = ax[2]
    a.fill_between(y, s["med_to_local_days"], s["p90_to_local_days"], color=PALETTE["owner"], alpha=0.12, lw=0)
    a.plot(y, s["med_to_local_days"], color=PALETTE["owner"], marker="o", ms=4, mec="white", mew=1)
    a.plot(y, s["p90_to_local_days"], color=PALETTE["owner"], lw=0.8, alpha=0.6)
    a.annotate("90th pct.", (y[-1], s["p90_to_local_days"].iloc[-1]), xytext=(2, 2), textcoords="offset points",
               fontsize=6.5, color="#5E6977")
    a.annotate("median", (y[-1], s["med_to_local_days"].iloc[-1]), xytext=(2, -8), textcoords="offset points",
               fontsize=6.5, color="#5E6977")
    a.set_ylim(0, None)
    a.set_title("(c) Days until the closer's funds return", loc="left")
    for a in ax:
        a.set_xticks(y)
        a.set_xticklabels([f"'{str(v)[2:]}" for v in y])
        a.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig3_costs")


def main() -> int:
    con = connect()
    H = manifest()["cutoff_height"]
    t = close_costs(con, H)
    t.to_parquet(WORK_DIR / "close_costs.parquet", index=False)
    mu = mutual(con, H)
    s = summarize(t, mu)
    bf = by_format(t)
    print(s.round(2).to_string())
    print(bf.round(2).to_string())
    pen = df(con, """SELECT year(spend_time_utc) AS y, count(DISTINCT spend_txid) AS txs, count(*) AS inputs,
                            sum(value_sat) / 1e8 AS btc, bool_or(listed) AS any_listed
                     FROM o WHERE spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach')
                     GROUP BY y ORDER BY y""")
    pen_m = df(con, """SELECT strftime(date_trunc('month', spend_time_utc), '%Y-%m') AS m, listed,
                              count(DISTINCT spend_txid) AS txs, sum(value_sat) / 1e8 AS btc
                       FROM o WHERE spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach')
                       GROUP BY ALL ORDER BY ALL""")
    wave = df(con, """SELECT strftime(fc.close_time_utc, '%Y-%m-%d') AS d, count(DISTINCT o.spend_txid) AS txs,
                             count(DISTINCT o.close_txid) AS closes, sum(o.value_sat) / 1e8 AS btc,
                             max(o.blocks_after_close) AS max_blocks_to_penalty
                      FROM o JOIN fc USING (close_txid)
                      WHERE o.spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach') AND NOT o.listed
                        AND fc.close_time_utc >= TIMESTAMP '2026-09-20' GROUP BY d ORDER BY d""")
    wave_fmt = df(con, """SELECT fc.fmt, count(DISTINCT fc.close_txid) AS closes, count(DISTINCT split_part(fc.chan_point, ':', 1)) AS funding_txs
                          FROM o JOIN fc USING (close_txid)
                          WHERE o.spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach') AND NOT o.listed
                            AND fc.close_time_utc BETWEEN TIMESTAMP '2026-09-28' AND TIMESTAMP '2026-10-06' GROUP BY fc.fmt""")
    csv = df(con, f"""SELECT csv_delay AS d, count(*) AS k FROM o JOIN fc USING (close_txid)
                      WHERE o.listed AND o.spend_type = 'to_local_delayed_sweep'
                        AND year(fc.close_time_utc) >= {PUBLIC_YEAR}
                      GROUP BY d ORDER BY k DESC""")
    s_all, t_all = s, t                               # every year, 2018 included (released in costs.json)
    s, t = s[s["y"] >= PUBLIC_YEAR].reset_index(drop=True), t[t["y"] >= PUBLIC_YEAR]
    fig_costs(s, t)

    rows = []
    for _, r in s.iterrows():
        rows.append([str(int(r["y"])) + (r"\textsuperscript{a}" if r["y"] == 2026 else ""), n(r["n"]),
                     n(r["med_commitment"]), n(r["med_party_fee"]), n(r["p90_party_fee"]), n(r["fee_med"]),
                     f1(r["ratio_med"]), n(r["med_vb"]), n(r["vb_med"]), f1(100 * r["share_over_1pct"]),
                     f1(r["med_to_local_days"])])
    tex_table("table_costs", ["Year", "Force closes", "Commit.", "Total", "p90", "Coop.", "Ratio", "vB", "Coop.\\ vB",
                              ">1\\% cap.", "Days"],
              rows, "lrrrrrrrrrr",
              "What a public force close costs, against a cooperative close, 2019--2026 (medians, sat unless stated).",
              "tab:costs",
              note=r"\textsuperscript{a}Closes to 4 October 2026, so that outputs had four weeks to be spent. "
                   r"Total: every fee the force close requires (commitment, fee bump, HTLC resolution, sweeps, "
                   r"anchor cleanup); batched spends split per input. Ratio: median total over the median cooperative "
                   r"close fee. vB: block space, the same way. $>$1\% cap.: share of force closes whose fees exceed 1\% "
                   r"of channel capacity. Days: median wait before the broadcaster's own balance can be swept.",
              wide=True)

    write_json("costs.json", {
        "by_year": s_all.to_dict(orient="records"),
        "public_from_year": PUBLIC_YEAR,
        "by_format": bf.to_dict(orient="records"),
        "penalties": pen.to_dict(orient="records"),
        "penalties_monthly": pen_m.to_dict(orient="records"),
        "penalty_wave_daily": wave.to_dict(orient="records"),
        "penalty_wave_format": wave_fmt.to_dict(orient="records"),
        "to_self_delay": csv.head(15).to_dict(orient="records"),
        "n_closes": int(len(t)), "n_closes_incl_2018": int(len(t_all)), "censor_blocks": SPEND_CENSOR_BLOCKS,
        "all_years": {
            "med_party_fee": float(t["party_fee"].median()), "total_party_btc": float(t["party_fee"].sum() / 1e8),
            "med_to_local_days": float(t["to_local_wait"].median() / 144),
            "share_to_self_delay_144": float(csv.loc[csv["d"] == 144, "k"].sum() / csv["k"].sum()),
            "share_to_self_delay_ge_2015": float(csv.loc[csv["d"] >= 2015, "k"].sum() / csv["k"].sum()),
            "anchor_value_lost_btc": float(t["anchor_value_lost"].sum() / 1e8),
        },
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
