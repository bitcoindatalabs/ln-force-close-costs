"""
05_fee_regimes.py (RQ4): how the fee market shapes force closes and their costs.

Needs data/work/blockstats.parquet (00_blockstats.py): per-block fee-rate percentiles from our node.

  A. Do fee spikes cause force closes? Weekly force closes of public channels (mass-close events excluded) per 1,000
     open channels against the weekly median block fee rate: an elasticity with year fixed effects (so it compares weeks
     within the same fee regime, not 2023 with 2025) and Newey–West errors; the same for pre-anchor and anchor closes
     separately; and an event study around fee spikes (weeks whose median fee rate was at least SPIKE_X times the
     previous 8 weeks' and at least SPIKE_MIN sat/vB).
  B. Fee guessing: a pre-anchor commitment's fee rate is fixed when it is signed; an anchor commitment's is low and
     bumped. Commitment fee rate against the median (p50) and the 10th percentile (p10) of its own block.
  C. Do third-party sweepers wait for cheap blocks? Fee rate of the sweep's block against the block 16 after the close.
  D. Lightning's closing bill: fees and block space of every close and every required spend (public or not), as a
     share of all transaction fees and all block weight, by year. Channel opens are not included.

Writes data/fees.json, paper/figures/fig5_fee_spikes.pdf, fig6_fee_guessing.pdf, paper/tables/table_fee_share.tex.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANCHOR_CSV_BLOCKS, CUTOFF_UTC, PALETTE, PUBLIC_FROM, SCAN_START_TS, WORK_DIR, coverage  # noqa: E402
from fcc.io import connect, df, manifest, write_json  # noqa: E402
from fcc.style import COL_W, PAGE_W, f1, n, plt, save, tex_table  # noqa: E402

SPIKE_X, SPIKE_MIN, SPIKE_LOOKBACK = 2.0, 10.0, 8
PRE_SCAN = SCAN_START_TS     # block 503,816: channels closed before it have no close in the snapshot
ANCHOR_FROM_YEAR = 2021      # anchor channels date from 2020-21: the anchor elasticity uses weeks from 2021


def weekly(con, start: str = PUBLIC_FROM) -> pd.DataFrame:
    """One row per calendar week (Monday 00:00 UTC) that lies wholly inside the window: from the first Monday on or
    after both the first block we hold and `start` (public series from 2019) to the last week that ends by the
    cutoff. The chain has every block, so every such week is complete whatever its block count (June 2021 had two
    weeks of fewer than 700 blocks, the China mining ban). Weeks without any force close are kept (counts 0), and
    the frame has no gaps, so a lag of k rows is a lag of k calendar weeks."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW ev AS SELECT * FROM '{WORK_DIR / 'mass_event_flags.parquet'}'")
    w = df(con, """
        WITH fee AS (SELECT date_trunc('week', to_timestamp(time)) AS wk, median(p50) AS p50, median(p10) AS p10,
                            min(height) AS h0, count(*) AS blocks, min(time) AS t0
                     FROM bs GROUP BY wk),
        f AS (SELECT date_trunc('week', fc.close_time_utc) AS wk,
                     count(*) FILTER (fc.listed AND NOT ev.in_event) AS organic,
                     count(*) FILTER (fc.listed) AS public,
                     count(*) FILTER (fc.listed AND NOT ev.in_event AND fc.fmt = 'legacy') AS legacy,
                     count(*) FILTER (fc.listed AND NOT ev.in_event AND fc.fmt = 'anchor') AS anchor,
                     count(*) FILTER (NOT fc.listed) AS unlisted,
                     count(*) AS all_force
              FROM fc LEFT JOIN ev USING (close_txid) GROUP BY wk)
        SELECT fee.*, f.* EXCLUDE (wk) FROM fee LEFT JOIN f USING (wk) ORDER BY wk""")
    cnt = ["organic", "public", "legacy", "anchor", "unlisted", "all_force"]
    w[cnt] = w[cnt].fillna(0).astype(int)
    first_block = pd.Timestamp(int(w["t0"].min()), unit="s", tz="UTC")
    lo = max(first_block, pd.Timestamp(start, tz="UTC"))
    lo = lo.normalize() + pd.Timedelta(days=(7 - lo.weekday()) % 7)
    if lo < max(first_block, pd.Timestamp(start, tz="UTC")):
        lo += pd.Timedelta(days=7)
    hi = pd.Timestamp(CUTOFF_UTC) - pd.Timedelta(days=7)            # the week must end by the cutoff
    w = w[(w["wk"] >= lo) & (w["wk"] <= hi)].drop(columns=["t0"]).reset_index(drop=True)
    if not (w["wk"].diff().dropna() == pd.Timedelta(days=7)).all():
        raise SystemExit("the weekly frame has gaps: blockstats does not cover the window")
    opens = [con.execute(f"""
        SELECT count(*) FROM fo LEFT JOIN c USING (chan_point)
        WHERE fo.funding_height < {h} AND coalesce(c.close_height, 1e9) >= {h}
          AND NOT (c.close_height IS NULL AND fo.mempool_closing_date IS NOT NULL
                   AND fo.mempool_closing_date < {PRE_SCAN})""").fetchone()[0] for h in w["h0"]]
    w["open"] = opens
    w["rate"] = 1000 * w["organic"] / w["open"]
    w["y"] = w["wk"].dt.year
    return w


def elasticity(w: pd.DataFrame, ycol: str, offset: bool = False, fee: str = "p50") -> dict:
    d = w[(w[ycol] > 0) & (w[fee] > 0)].copy()
    yv = np.log(d[ycol]) - (np.log(d["open"]) if offset else 0)
    X = pd.get_dummies(d["y"].astype(str), prefix="y", drop_first=True, dtype=float)
    X.insert(0, "trend", (d["wk"] - d["wk"].min()).dt.days / 365.25)     # adoption drifts within a year too
    X.insert(0, "log_fee", np.log(d[fee].astype(float)))
    X = sm.add_constant(X)
    r = sm.OLS(yv.astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    lo, hi = r.conf_int().loc["log_fee"]
    return {"b": float(r.params["log_fee"]), "lo": float(lo), "hi": float(hi), "p": float(r.pvalues["log_fee"]),
            "weeks": int(len(d))}


def robustness(w: pd.DataFrame) -> dict:
    """The organic-rate elasticity under other choices: lagged / lead fees (closes broadcast in a spike may confirm
    later), the 10th percentile, all public closes (mass events kept), and months instead of weeks."""
    w = w.copy()
    fee = w.set_index("wk")["p50"]                  # lags by calendar week (the frame has no gaps)
    for k in (1, 2, 3, 4):
        w[f"p50_lag{k}"] = (w["wk"] - pd.Timedelta(weeks=k)).map(fee)
    w["p50_lead1"] = (w["wk"] + pd.Timedelta(weeks=1)).map(fee)
    out = {"baseline": elasticity(w, "organic", offset=True),
           "fee_lag1": elasticity(w.dropna(subset=["p50_lag1"]), "organic", offset=True, fee="p50_lag1"),
           "fee_lag2": elasticity(w.dropna(subset=["p50_lag2"]), "organic", offset=True, fee="p50_lag2"),
           "fee_lag3": elasticity(w.dropna(subset=["p50_lag3"]), "organic", offset=True, fee="p50_lag3"),
           "fee_lag4": elasticity(w.dropna(subset=["p50_lag4"]), "organic", offset=True, fee="p50_lag4"),
           "fee_lead1": elasticity(w.dropna(subset=["p50_lead1"]), "organic", offset=True, fee="p50_lead1"),
           "fee_p10": elasticity(w, "organic", offset=True, fee="p10"),
           "with_mass_events": elasticity(w, "public", offset=True)}
    m = w.assign(wk=w["wk"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()).groupby("wk").agg(
        organic=("organic", "sum"), open=("open", "mean"), p50=("p50", "median"), y=("y", "first")).reset_index()
    out["monthly"] = elasticity(m, "organic", offset=True)
    return out


def spikes(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    base = w["p50"].shift(1).rolling(SPIKE_LOOKBACK, min_periods=SPIKE_LOOKBACK).median()
    w["spike"] = (w["p50"] >= SPIKE_X * base) & (w["p50"] >= SPIKE_MIN)
    first = w["spike"] & ~w["spike"].shift(1).rolling(SPIKE_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    rows = []
    for i in np.flatnonzero(first.to_numpy()):
        if i < 4 or i + 1 >= len(w):
            continue
        pre, post = w.iloc[i - 4:i], w.iloc[i:i + 2]
        rows.append({"week": w.loc[i, "wk"].strftime("%Y-%m-%d"), "p50": float(w.loc[i, "p50"]),
                     "p50_before": float(pre["p50"].median()),
                     **{f"{k}_before": float(pre[k].mean()) for k in SPIKE_SERIES},
                     **{f"{k}_after": float(post[k].mean()) for k in SPIKE_SERIES}})
    return pd.DataFrame(rows)


SPIKE_SERIES = ("organic", "legacy", "anchor", "unlisted", "all_force")


def spike_ratio(sp: pd.DataFrame) -> dict:
    return {k: (float(sp[f"{k}_after"].sum() / sp[f"{k}_before"].sum()) if len(sp) and sp[f"{k}_before"].sum() > 0
                else None) for k in SPIKE_SERIES}


def frame_summary(w: pd.DataFrame) -> dict:
    """The headline tests on another week frame (robustness: the 2021 start of the first draft, or 2018 with its
    coverage-limited public counts)."""
    sp = spikes(w)
    wa = w[w["y"] >= ANCHOR_FROM_YEAR]
    rb = robustness(w)
    return {"weeks": int(len(w)), "first_week": w["wk"].min().strftime("%Y-%m-%d"),
            "last_week": w["wk"].max().strftime("%Y-%m-%d"),
            "elasticity": {"organic_rate": elasticity(w, "organic", offset=True),
                           "legacy": elasticity(w, "legacy", offset=True),
                           "anchor": elasticity(wa, "anchor", offset=True), "unlisted": elasticity(w, "unlisted"),
                           "all_force": elasticity(w, "all_force")},
            "robustness": {k: rb[k] for k in ("fee_lag1", "fee_lag2", "fee_lag3", "fee_lag4", "monthly")},
            "spikes": sp["week"].tolist() if len(sp) else [], "spike_ratio": spike_ratio(sp)}


def fee_guessing(con) -> pd.DataFrame:
    return df(con, """
        SELECT year(fc.close_time_utc) AS y, fc.fmt, fc.listed, fc.feerate_sat_vb AS fr, fc.fee_sat, fc.vsize,
               bs.p10, bs.p50, bs.minfeerate
        FROM fc JOIN bs ON bs.height = fc.close_height WHERE fc.fmt IN ('legacy', 'anchor')""")


def guess_summary(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["ratio"] = g["fr"] / g["p50"].clip(lower=1)
    g["over_sat"] = np.maximum(0, g["fee_sat"] - g["p10"] * g["vsize"])
    out = g.groupby(["y", "fmt"]).agg(
        n=("fr", "size"), med_ratio=("ratio", "median"), q25_ratio=("ratio", lambda x: x.quantile(0.25)),
        q75_ratio=("ratio", lambda x: x.quantile(0.75)), share_over_2x=("ratio", lambda x: (x > 2).mean()),
        share_below_p10=("fr", lambda x: np.nan),
        over_btc=("over_sat", lambda x: x.sum() / 1e8), med_fr=("fr", "median"), med_p50=("p50", "median")).reset_index()
    below = g.assign(b=g["fr"] < g["p10"]).groupby(["y", "fmt"])["b"].mean().reset_index()
    out["share_below_p10"] = out.merge(below, on=["y", "fmt"])["b"]
    return out


def sweeper_timing(con) -> pd.DataFrame:
    return df(con, f"""
        SELECT year(fc.close_time_utc) AS y, o.blocks_after_close AS d,
               b16.p50 AS p50_at_16, bsw.p50 AS p50_at_sweep, o.spend_fee_sat::DOUBLE / o.spend_vsize AS sweep_fr
        FROM o JOIN fc USING (close_txid)
        JOIN bs b16 ON b16.height = fc.close_height + {ANCHOR_CSV_BLOCKS}
        JOIN bs bsw ON bsw.height = o.spend_height
        WHERE o.spend_type = 'anchor_sweep_anyone'""")


def closing_bill(con) -> pd.DataFrame:
    third = WORK_DIR / "third_party_sweeps.parquet"
    return df(con, f"""
        WITH parts AS (
            SELECT year(close_time_utc) AS y, fee_sat AS fee, vsize AS vb, 'mutual' AS k FROM c WHERE close_type <> 'force'
            UNION ALL SELECT year(close_time_utc), fee_sat, vsize, 'commitment' FROM fc
            UNION ALL
            SELECT year(spend_time_utc),
                   spend_fee_sat * CASE WHEN spend_type IN ('anchor_sweep_owner', 'no_witness') AND blocks_after_close = 0
                                        THEN 1.0 ELSE 1.0 / spend_n_inputs END,
                   spend_vsize * CASE WHEN spend_type IN ('anchor_sweep_owner', 'no_witness') AND blocks_after_close = 0
                                      THEN 1.0 ELSE 1.0 / spend_n_inputs END, 'spends'
            FROM o WHERE spend_txid IS NOT NULL AND spend_type NOT IN ('to_remote_p2wpkh_spend', 'anchor_sweep_anyone')
            UNION ALL SELECT year(t), fee, vsize, 'third_party' FROM '{third}'),
        ln AS (SELECT y, sum(fee) / 1e8 AS ln_fee_btc, sum(vb) AS ln_vb,
                      sum(fee) FILTER (k = 'mutual') / 1e8 AS mutual_btc,
                      sum(fee) FILTER (k <> 'mutual') / 1e8 AS force_btc
               FROM parts GROUP BY y),
        allb AS (SELECT year(to_timestamp(time)) AS y, sum(totalfee) / 1e8 AS all_fee_btc,
                        sum(total_weight) / 4.0 AS all_vb FROM bs GROUP BY y)
        SELECT ln.*, allb.all_fee_btc, allb.all_vb, ln.ln_fee_btc / allb.all_fee_btc AS fee_share,
               ln.ln_vb / allb.all_vb AS vb_share
        FROM ln JOIN allb USING (y) ORDER BY y""")


def fig_spikes(w: pd.DataFrame, sp: pd.DataFrame):
    fig = plt.figure(figsize=(PAGE_W, 2.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.6, 1], height_ratios=[1, 1.15], hspace=0.38, wspace=0.28)
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[1, 0], sharex=a0)
    a0.plot(w["wk"], w["p50"], color=PALETTE["ink_mark"], lw=1.0)
    a0.set_yscale("log")
    a0.set_yticks([1, 3, 10, 30, 100])
    a0.set_yticklabels(["1", "3", "10", "30", "100"])
    a0.minorticks_off()
    a0.set_title("(a) Median block fee rate, weekly (sat/vB)", loc="left")
    a1.plot(w["wk"], w["rate"], color=PALETTE["force"], lw=1.0)
    a1.set_ylim(0, None)
    a1.set_title("(b) Public force closes per 1,000 open channels, weekly", loc="left")
    for _, r in sp.iterrows():
        t = pd.Timestamp(r["week"])
        for a in (a0, a1):
            a.axvline(t, color=PALETTE["muted"], lw=0.6, alpha=0.6, zorder=0)
    plt.setp(a0.get_xticklabels(), visible=False)
    a1.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    a1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    for a in (a0, a1):
        a.grid(axis="x", visible=False)
    a2 = fig.add_subplot(gs[:, 1])
    # unlisted closes carry no node keys, so mass-close events cannot be removed from them: not shown
    keys = [("organic", "All\npublic", PALETTE["force"]), ("legacy", "Pre-\nanchor", PALETTE["legacy"]),
            ("anchor", "Anchor", PALETTE["anchor"])]
    vals = [sp[f"{k}_after"].sum() / sp[f"{k}_before"].sum() for k, _, _ in keys]
    a2.bar(range(3), vals, width=0.55, color=[c for _, _, c in keys], edgecolor="white", linewidth=1)
    a2.axhline(1, color=PALETTE["ink_mark"], lw=0.7)
    for i, v in enumerate(vals):
        a2.annotate(f"{v:.2f}×", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
                    color="#242D38")
    a2.set_xticks(range(3))
    a2.set_xticklabels([lab for _, lab, _ in keys], fontsize=7)
    a2.set_ylim(0, max(vals) * 1.25)
    a2.set_title(f"(c) Spike weeks ÷ the 4 weeks\nbefore ({len(sp)} spikes)", loc="left")
    a2.grid(axis="x", visible=False)
    save(fig, "fig5_fee_spikes")


def fig_guessing(gs: pd.DataFrame):
    """Commitment fee rate ÷ median fee rate of its block, by year: median and middle half, per format."""
    fig, a = plt.subplots(figsize=(COL_W, 2.25))
    for fmt, lab, col in (("legacy", "Pre-anchor", PALETTE["legacy"]), ("anchor", "Anchor", PALETTE["anchor"])):
        d = gs[gs["fmt"] == fmt].sort_values("y")
        a.fill_between(d["y"], d["q25_ratio"], d["q75_ratio"], color=col, alpha=0.15, lw=0)
        a.plot(d["y"], d["med_ratio"], color=col, marker="o", ms=4, mec="white", mew=1, label=lab)
    a.axhline(1, color=PALETTE["ink_mark"], lw=0.7)
    a.set_yscale("log", base=2)
    a.set_yticks([0.5, 1, 2, 4])
    a.set_yticklabels(["½×", "1×", "2×", "4×"])
    a.minorticks_off()
    a.set_ylim(0.4, 5)
    a.set_ylabel("Fee rate ÷ block median")
    ys = sorted(gs["y"].unique())
    a.set_xticks(ys)
    a.set_xticklabels([f"'{str(v)[2:]}" for v in ys])
    a.grid(axis="x", visible=False)
    a.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    save(fig, "fig6_fee_guessing")


def main() -> int:
    con = connect()
    H = manifest()["cutoff_height"]
    have = con.execute("SELECT count(*), min(height), max(height) FROM bs").fetchone()
    print(f"blockstats: {have[0]:,} blocks {have[1]:,}..{have[2]:,}")
    w = weekly(con)
    el = {"organic_rate": elasticity(w, "organic", offset=True), "legacy": elasticity(w, "legacy", offset=True),
          "anchor": elasticity(w[w["y"] >= ANCHOR_FROM_YEAR], "anchor", offset=True),
          "unlisted": elasticity(w, "unlisted"), "all_force": elasticity(w, "all_force")}
    rob = robustness(w)
    frames = {"from_2021": frame_summary(weekly(con, start="2021-01-01")),
              "from_2018": frame_summary(weekly(con, start="2018-01-01"))}
    print({k: (v["weeks"], v["elasticity"]["organic_rate"]["b"], v["spikes"], v["spike_ratio"]) for k, v in frames.items()})
    sp = spikes(w)
    g = fee_guessing(con)
    gs = guess_summary(g)
    st = sweeper_timing(con)
    st_s = st.assign(cheaper=st["p50_at_sweep"] < st["p50_at_16"], waited=st["d"] > ANCHOR_CSV_BLOCKS + 1).groupby("y").agg(
        n=("d", "size"), share_waited=("waited", "mean"), med_p50_at_16=("p50_at_16", "median"),
        med_p50_at_sweep=("p50_at_sweep", "median"), med_sweep_fr=("sweep_fr", "median"),
        share_sweep_block_cheaper=("cheaper", "mean")).reset_index()
    bill = closing_bill(con)
    print(pd.DataFrame(el).T.round(3))
    print(pd.DataFrame(rob).T.round(3))
    print(sp.round(2).to_string())
    print(gs.round(3).to_string())
    print(st_s.round(2).to_string())
    print(bill.round(4).to_string())

    fig_spikes(w, sp)
    fig_guessing(gs)
    bill["coverage"] = bill["y"].map(coverage)
    rows = []
    for _, r in bill.iterrows():
        lab = str(int(r["y"])) + (r"\textsuperscript{a}" if r["y"] == 2026 else "")
        if r["coverage"] == "registry_limited":      # 2018 shows force closes only
            rows.append([lab + r"\textsuperscript{b}", "--", f"{r['force_btc']:.2f}", "--", "--", "--"])
            continue
        rows.append([lab, f"{r['mutual_btc']:.2f}", f"{r['force_btc']:.2f}", f"{r['ln_fee_btc']:.2f}",
                     f"{100 * r['fee_share']:.2f}", f"{100 * r['vb_share']:.2f}"])
    tex_table("table_fee_share", ["Year", "Coop.\\ BTC", "Force BTC", "Total BTC", "\\% of fees", "\\% of block space"],
              rows, "lrrrrr", "Lightning's closing bill: fees paid by channel closes and the transactions they require, "
              "public or not, against all Bitcoin transaction fees and block space.", "tab:fee_share",
              note=r"\textsuperscript{a}January--October. Force: commitments, required spends (split per input) and "
                   r"third-party anchor sweeps; cooperative closes of channels outside our public list cannot be seen "
                   r"and are not included, nor are channel opens. \textsuperscript{b}From 12 January; cooperative "
                   r"closes of 2018 are mostly outside our list, so only force closes are shown.")
    write_json("fees.json", {
        "weekly": w.assign(wk=w["wk"].dt.strftime("%Y-%m-%d")).drop(columns=["h0"]).to_dict(orient="records"),
        "elasticity": el, "robustness": rob, "spikes": sp.to_dict(orient="records"),
        "spike_rule": {"x": SPIKE_X, "min_sat_vb": SPIKE_MIN, "lookback_weeks": SPIKE_LOOKBACK},
        "spike_ratio": spike_ratio(sp),
        "frame": {"first_week": w["wk"].min().strftime("%Y-%m-%d"), "last_week": w["wk"].max().strftime("%Y-%m-%d"),
                  "weeks": int(len(w)), "public_from": PUBLIC_FROM, "anchor_from_year": ANCHOR_FROM_YEAR,
                  "rule": "calendar weeks wholly inside the window, zero-close weeks kept, lags by calendar week"},
        "frames": frames,
        "fee_guessing": gs.to_dict(orient="records"), "sweeper_timing": st_s.to_dict(orient="records"),
        "closing_bill": bill.to_dict(orient="records"),
        "blockstats": {"blocks": have[0], "first": have[1], "last": have[2], "cutoff": H},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
