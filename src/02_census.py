"""
02_census.py (RQ1): how many channels force close, how that changed 2018–2026, and who is involved.

  * yearly census of public closes (mutual / force / splice) and of force closes outside the public list;
  * monthly force-close rate per 1,000 open public channels (open = funding output not yet spent), with and
    without mass-close events (one node party to >= MASS_MIN force closes on one UTC day);
  * cumulative incidence of a force close by channel age, per funding-year cohort (administrative censoring only,
    so the share at age a uses channels funded at least a before the cutoff);
  * commitment formats of force closes (legacy, anchor, simple taproot, zero-fee);
  * force share of closes by client pair, from paper 01's labels (closes from 2023-05-08, labels valid that day).

Writes data/census.json, paper/figures/fig1_census.pdf, fig2_cohorts.pdf, paper/tables/table_census.tex,
table_client_pairs.tex.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PALETTE, PUBLIC_FROM, SCAN_START_HEIGHT, SCAN_START_TS, WORK_DIR, coverage  # noqa: E402
from fcc.io import connect, df, manifest, write_json  # noqa: E402
from fcc.style import COL_W, PAGE_W, f1, n, plt, save, tex_table  # noqa: E402

MASS_MIN = 20                 # force closes by one node on one UTC day that make a mass-close event
LABELS_FROM = "2023-05-08"    # first day of paper 01's client labels
CLIENTS = ["LND", "CLN", "Eclair", "LDK"]
PRE_SCAN = SCAN_START_TS      # block 503,816 (2018-01-12): channels closed before it have no close in the snapshot


def yearly(con) -> pd.DataFrame:
    pub = df(con, """
        SELECT year(close_time_utc) AS y, count(*) AS closes,
               count(*) FILTER (close_type = 'mutual') AS mutual, count(*) FILTER (close_type = 'force') AS force,
               count(*) FILTER (close_type = 'splice') AS splice,
               sum(capacity) FILTER (close_type = 'force') / 1e8 AS force_btc,
               sum(capacity) FILTER (close_type = 'mutual') / 1e8 AS mutual_btc
        FROM c GROUP BY y ORDER BY y""")
    ul = df(con, """SELECT year(close_time_utc) AS y, count(*) AS ul_force, sum(capacity) / 1e8 AS ul_btc
                    FROM u GROUP BY y ORDER BY y""")
    fmt = df(con, """SELECT year(close_time_utc) AS y, listed, fmt, count(*) AS k FROM fc GROUP BY ALL""")
    t = pub.merge(ul, on="y", how="left")
    t["force_share"] = t["force"] / (t["force"] + t["mutual"])
    t["all_force"] = t["force"] + t["ul_force"]          # complete in every year (listed + unlisted)
    t["all_force_btc"] = t["force_btc"] + t["ul_btc"]
    for listed, tag in ((True, "pub"), (False, "ul")):
        p = fmt[fmt["listed"] == listed].pivot_table(index="y", columns="fmt", values="k", fill_value=0)
        p = p.div(p.sum(axis=1), axis=0).add_prefix(f"{tag}_fmt_").reset_index()
        t = t.merge(p, on="y", how="left")
    age = df(con, """SELECT year(close_time_utc) AS y, close_type,
                            median((close_height - fo.funding_height) / 144.0) AS med_age_days
                     FROM c JOIN fo USING (chan_point) WHERE close_type IN ('force', 'mutual') GROUP BY ALL""")
    age = age.pivot(index="y", columns="close_type", values="med_age_days").add_prefix("med_age_").reset_index()
    t = t.merge(age, on="y", how="left").fillna(0)
    t["coverage"] = t["y"].map(coverage)
    return t


def event_flags(con) -> pd.DataFrame:
    """Per listed force close: is it part of a mass-close event (one node, one UTC day, >= MASS_MIN force closes)?"""
    return df(con, f"""
        WITH x AS (SELECT close_time_utc::DATE AS d, node1_pub AS node FROM c WHERE close_type = 'force'
                   UNION ALL SELECT close_time_utc::DATE, node2_pub FROM c WHERE close_type = 'force'),
        ev AS (SELECT d, node FROM x GROUP BY ALL HAVING count(*) >= {MASS_MIN})
        SELECT c.close_txid, EXISTS (SELECT 1 FROM ev WHERE ev.d = c.close_time_utc::DATE
                                     AND ev.node IN (c.node1_pub, c.node2_pub)) AS in_event
        FROM c WHERE close_type = 'force'""")


def monthly(con) -> pd.DataFrame:
    """Closes per month and the open public population at each month start (heights from the closes' own times)."""
    m = df(con, f"""
        WITH ev AS (   -- mass-close events: node x UTC day with >= {MASS_MIN} listed force closes
            SELECT d, node FROM (
                SELECT close_time_utc::DATE AS d, node1_pub AS node FROM c WHERE close_type = 'force'
                UNION ALL SELECT close_time_utc::DATE, node2_pub FROM c WHERE close_type = 'force')
            GROUP BY ALL HAVING count(*) >= {MASS_MIN}),
        f AS (SELECT c.*, EXISTS (SELECT 1 FROM ev WHERE ev.d = c.close_time_utc::DATE
                                  AND ev.node IN (c.node1_pub, c.node2_pub)) AS in_event
              FROM c WHERE close_type = 'force')
        SELECT date_trunc('month', close_time_utc) AS month, count(*) AS force,
               count(*) FILTER (NOT in_event) AS force_organic, min(close_height) AS h0
        FROM f GROUP BY month ORDER BY month""")
    mut = df(con, """SELECT date_trunc('month', close_time_utc) AS month, count(*) AS mutual FROM c
                     WHERE close_type = 'mutual' GROUP BY month""")
    ul = df(con, """SELECT date_trunc('month', close_time_utc) AS month, count(*) AS ul_force FROM u GROUP BY month""")
    # every month with a force close of any kind (2018 has months without a listed close), first block from fc
    allm = df(con, """SELECT date_trunc('month', close_time_utc) AS month, min(close_height) AS h_all FROM fc
                      GROUP BY month""")
    m = allm.merge(m, on="month", how="left").merge(mut, on="month", how="left").merge(ul, on="month", how="left")
    for k in ("force", "force_organic", "mutual", "ul_force"):
        m[k] = m[k].fillna(0).astype(int)
    m["h0"] = m["h0"].fillna(m["h_all"]).astype(int)
    m = m.drop(columns=["h_all"]).sort_values("month").reset_index(drop=True)
    # open listed channels at the first close height of each month (funding output unspent at that height)
    opens = []
    for h in m["h0"]:
        opens.append(con.execute(f"""
            SELECT count(*) FROM fo LEFT JOIN c USING (chan_point)
            WHERE fo.funding_height < {h} AND coalesce(c.close_height, 1e9) >= {h}
              AND NOT (c.close_height IS NULL AND fo.mempool_closing_date IS NOT NULL
                       AND fo.mempool_closing_date < {PRE_SCAN})""").fetchone()[0])
    m["open"] = opens
    m["rate"] = 1000 * m["force"] / m["open"]
    m["rate_organic"] = 1000 * m["force_organic"] / m["open"]
    first = pd.Timestamp(PUBLIC_FROM, tz="UTC")
    m["coverage"] = np.select([m["month"] < first, m["month"] == first], ["registry_limited", "ramping"], "full")
    return m


def cohorts(con, cutoff: int) -> dict:
    """Share of channels funded in each year that ended in a force (or mutual) close by age a (days)."""
    ch = df(con, f"""SELECT fo.funding_height AS fh, c.close_height AS ch, c.close_type AS ct
                    FROM fo LEFT JOIN c USING (chan_point) WHERE fo.funding_height >= {SCAN_START_HEIGHT}""")
    starts = year_start_heights(con)
    ch["cohort"] = np.searchsorted(np.array(list(starts.values())), ch["fh"].to_numpy(), side="right") - 1
    ch["cohort"] = ch["cohort"].map(dict(enumerate(starts.keys())))
    ages = np.arange(0, 731, 7)
    out = {}
    for y, g in ch.groupby("cohort"):
        if y > 2025:
            continue
        age_close = (g["ch"] - g["fh"]) / 144.0
        span = (cutoff - g["fh"]) / 144.0
        force, mutual, nrisk = [], [], []
        for a in ages:
            at = span >= a                               # channels observed for at least a days
            k = int(at.sum())
            nrisk.append(k)
            force.append(float(((g["ct"] == "force") & (age_close <= a) & at).sum() / k) if k else None)
            mutual.append(float(((g["ct"] == "mutual") & (age_close <= a) & at).sum() / k) if k else None)
        out[str(y)] = {"n": int(len(g)), "age_days": ages.tolist(), "force": force, "mutual": mutual,
                       "n_observed": nrisk, "coverage": coverage(y)}
    return out


def year_start_heights(con) -> dict:
    """First block of each year of the window, from the per-block fee table (every block from the scan start); the
    window's first year starts at the scan's first block."""
    r = df(con, """SELECT year(to_timestamp(time)) AS y, min(height) AS h FROM bs GROUP BY y ORDER BY y""")
    return dict(zip(r["y"].astype(int), r["h"].astype(int)))


def client_pairs(con) -> pd.DataFrame:
    return df(con, f"""
        WITH lab AS (
            SELECT pub_key, strptime(valid_from, '%Y%m%d')::DATE AS d0,
                   coalesce(strptime(valid_to, '%Y%m%d')::DATE, DATE '2100-01-01') AS d1,
                   CASE WHEN client_confidence IN ('high', 'medium', 'operator') THEN client ELSE 'Unknown' END AS cl
            FROM h),
        x AS (
            SELECT c.close_type, c.capacity, coalesce(l1.cl, 'Unknown') AS a, coalesce(l2.cl, 'Unknown') AS b
            FROM c
            LEFT JOIN lab l1 ON l1.pub_key = c.node1_pub AND c.close_time_utc::DATE BETWEEN l1.d0 AND l1.d1
            LEFT JOIN lab l2 ON l2.pub_key = c.node2_pub AND c.close_time_utc::DATE BETWEEN l2.d0 AND l2.d1
            WHERE c.close_time_utc >= DATE '{LABELS_FROM}' AND c.close_type IN ('force', 'mutual'))
        SELECT least(a, b) || '–' || greatest(a, b) AS pair, count(*) AS closes,
               count(*) FILTER (close_type = 'force') AS force,
               count(*) FILTER (close_type = 'force') / count(*) AS force_share,
               (a = 'Unknown' OR b = 'Unknown') AS has_unknown
        FROM x GROUP BY ALL ORDER BY closes DESC""")


def pair_top_node(con) -> dict:
    """For each client pair: the share of its closes that involve its single most active node (is a pair's outlier one
    operator?). Aggregate only; no node is named."""
    x = df(con, f"""
        WITH lab AS (
            SELECT pub_key, strptime(valid_from, '%Y%m%d')::DATE AS d0,
                   coalesce(strptime(valid_to, '%Y%m%d')::DATE, DATE '2100-01-01') AS d1,
                   CASE WHEN client_confidence IN ('high', 'medium', 'operator') THEN client ELSE 'Unknown' END AS cl
            FROM h)
        SELECT c.close_type, c.node1_pub, c.node2_pub,
               least(coalesce(l1.cl, 'Unknown'), coalesce(l2.cl, 'Unknown')) || '–' ||
               greatest(coalesce(l1.cl, 'Unknown'), coalesce(l2.cl, 'Unknown')) AS pair
        FROM c
        LEFT JOIN lab l1 ON l1.pub_key = c.node1_pub AND c.close_time_utc::DATE BETWEEN l1.d0 AND l1.d1
        LEFT JOIN lab l2 ON l2.pub_key = c.node2_pub AND c.close_time_utc::DATE BETWEEN l2.d0 AND l2.d1
        WHERE c.close_time_utc >= DATE '{LABELS_FROM}' AND c.close_type IN ('force', 'mutual')""")
    out = {}
    for pair, d in x.groupby("pair"):
        if len(d) < 1000:
            continue
        top = pd.concat([d["node1_pub"], d["node2_pub"]]).value_counts().index[0]
        m = (d["node1_pub"] == top) | (d["node2_pub"] == top)
        out[pair] = {"closes": int(len(d)), "top_node_share": float(m.mean()),
                     "force_share_without_top_node": float((d.loc[~m, "close_type"] == "force").mean()) if (~m).any() else None}
    return out


def validation(con) -> pd.DataFrame:
    """Our close type against mempool.space's close reason (1 mutual, 2 force, 3 force with penalty), where it has one."""
    return df(con, """SELECT c.close_type, fo.mempool_closing_reason::INT AS reason, count(*) AS k
                      FROM c JOIN fo USING (chan_point) WHERE fo.mempool_closing_reason IS NOT NULL
                      GROUP BY ALL ORDER BY ALL""")


def validation_coverage(con) -> pd.DataFrame:
    """Share of public closes, by year, for which mempool.space gives a close reason (Data, Validation)."""
    return df(con, """SELECT year(c.close_time_utc) AS y, count(*) AS closes,
                             count(fo.mempool_closing_reason) AS with_reason
                      FROM c LEFT JOIN fo USING (chan_point) GROUP BY y ORDER BY y""")


def mass_events(con) -> pd.DataFrame:
    return df(con, f"""
        WITH x AS (SELECT close_time_utc::DATE AS d, node1_pub AS node FROM c WHERE close_type = 'force'
                   UNION ALL SELECT close_time_utc::DATE, node2_pub FROM c WHERE close_type = 'force'),
        ev AS (SELECT d, node, count(*) AS k FROM x GROUP BY ALL HAVING count(*) >= {MASS_MIN})
        SELECT year(d) AS y, count(*) AS events, sum(k) AS closes_in_events, max(k) AS largest FROM ev GROUP BY y
        ORDER BY y""")


def fig_census(y: pd.DataFrame, m: pd.DataFrame):
    """the public / unlisted split, cooperative closes, force share and rates from 2019; all force closes
    (listed + unlisted, complete in every year) from the scan start."""
    fig, ax = plt.subplots(1, 3, figsize=(PAGE_W, 2.25), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    m = m.assign(all_force=m["force"] + m["ul_force"].fillna(0))
    q = m.assign(q=m["month"].dt.tz_localize(None).dt.to_period("Q")).groupby("q")[
        ["force", "mutual", "ul_force", "all_force"]].agg(["sum", "size"])
    q = q[q[("force", "size")] == 3].xs("sum", axis=1, level=1)       # complete quarters only
    x = q.index.to_timestamp() + pd.Timedelta(days=45)
    pub = x >= pd.Timestamp(PUBLIC_FROM)
    a = ax[0]
    a.plot(x, q["all_force"] / 1e3, color=PALETTE["ink_mark"], lw=1.0, ls="--", label="All force (listed or not)")
    a.plot(x[pub], q["mutual"][pub] / 1e3, color=PALETTE["mutual"], label="Public, cooperative")
    a.plot(x[pub], q["force"][pub] / 1e3, color=PALETTE["force"], label="Public, force")
    a.plot(x[pub], q["ul_force"][pub] / 1e3, color=PALETTE["unlisted"], label="Unlisted, force")
    for ser, col in (("mutual", "mutual"), ("force", "force"), ("ul_force", "unlisted")):
        a.annotate(f"{q[ser].iloc[-1] / 1e3:.1f}k", (x[-1], q[ser].iloc[-1] / 1e3), xytext=(3, 0),
                   textcoords="offset points", va="center", fontsize=7, color="#5E6977")
    a.set_title("(a) Closes per quarter (thousands)", loc="left")
    a.legend(loc="upper left", fontsize=6.8, handlelength=1.2)
    a.set_ylim(0, None)
    a = ax[1]
    y = y[y["y"] >= pd.Timestamp(PUBLIC_FROM).year]
    a.plot(y["y"], 100 * y["force_share"], color=PALETTE["force"], marker="o", markersize=4,
           markeredgecolor="white", markeredgewidth=1)
    for _, r in y.iterrows():
        a.annotate(f"{100 * r['force_share']:.0f}%", (r["y"], 100 * r["force_share"]), xytext=(0, 5),
                   textcoords="offset points", ha="center", fontsize=7, color="#5E6977")
    a.set_ylim(0, 55)
    a.set_title("(b) Force share of public closes", loc="left")
    a.set_xticks(y["y"])
    a.set_xticklabels([f"'{str(v)[2:]}" for v in y["y"]])
    a.grid(axis="x", visible=False)
    a = ax[2]
    m = m[m["month"] >= pd.Timestamp(PUBLIC_FROM, tz="UTC")]
    a.plot(m["month"], m["rate"], color=PALETTE["mutual"], lw=1.0, label="All")
    a.plot(m["month"], m["rate_organic"], color=PALETTE["force"], label="Without mass-close events")
    a.set_ylim(0, None)
    a.set_title("(c) Monthly force-close rate", loc="left")
    a.set_ylabel("Per 1,000 open public channels", fontsize=7.5)
    a.legend(loc="upper right", fontsize=6.8, handlelength=1.2)
    for a in (ax[0], ax[2]):
        a.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
        a.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("'%y"))
        a.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig1_census")


def fig_cohorts(co: dict):
    fig, a = plt.subplots(figsize=(COL_W, 2.3))
    shades = ["#F6D9CC", "#F0BBA4", "#E9A487", "#DE825F", "#D4603A", "#A95137", "#6E3322"]
    co = {k: v for k, v in co.items() if int(k) >= pd.Timestamp(PUBLIC_FROM).year}     # cohorts from 2019
    for (yr, d), col in zip(sorted(co.items()), shades):
        ages = np.array(d["age_days"])
        f = np.array([np.nan if v is None else v for v in d["force"]])
        nobs = np.array(d["n_observed"])
        keep = nobs >= 0.5 * d["n"]
        a.plot(ages[keep], 100 * f[keep], color=col, lw=1.6)
        a.annotate(yr, (ages[keep][-1], 100 * f[keep][-1]), xytext=(3, 0), textcoords="offset points",
                   va="center", fontsize=7, color="#5E6977")
    a.set_xlabel("Channel age (days)")
    a.set_ylabel("Force-closed by that age (%)")
    a.set_xticks([0, 90, 180, 365, 540, 730])
    a.set_xlim(0, 790)
    a.set_ylim(0, None)
    a.grid(axis="x", visible=False)
    a.set_title("Public channels, by year funded", loc="left")
    fig.tight_layout()
    save(fig, "fig2_cohorts")


def main() -> int:
    con = connect()
    man = manifest()
    y = yearly(con)
    m = monthly(con)
    co = cohorts(con, man["cutoff_height"])
    pairs = client_pairs(con)
    ev = mass_events(con)
    event_flags(con).to_parquet(WORK_DIR / "mass_event_flags.parquet", index=False)
    print(y.round(3).to_string())
    print(ev.to_string())

    # 2024-09-01: the single-operator unlisted event named in the agenda (blocks 859,401–859,418)
    sep = con.execute("""SELECT count(*), sum(capacity) / 1e8, count(*) FILTER (fmt = 'legacy'),
                                min(close_height), max(close_height)
                         FROM fc WHERE NOT listed AND close_height BETWEEN 859401 AND 859418""").fetchone()
    sep_t = dict(con.execute(f"SELECT height, time FROM bs WHERE height IN ({sep[3]}, {sep[4]})").fetchall())
    sep_before = con.execute(f"""SELECT count(*) FROM fc WHERE NOT listed
                                 AND close_height BETWEEN {sep[3] - 144} AND {sep[3] - 1}""").fetchone()[0]

    fig_census(y, m)
    fig_cohorts(co)

    rows = []
    dash = "--"
    for _, r in y.iterrows():
        lab = str(int(r["y"])) + (r"\textsuperscript{a}" if r["y"] == 2026 else "") + \
            (r"\textsuperscript{b}" if r["coverage"] == "registry_limited" else "") + \
            (r"\textsuperscript{c}" if r["coverage"] == "ramping_january" else "")
        if r["coverage"] == "registry_limited":     # 2018 shows force-close totals only
            rows.append([lab, n(r["all_force"])] + [dash] * 9)
            continue
        rows.append([lab, n(r["all_force"]), n(r["mutual"]),
                     n(r["force"]), f1(100 * r["force_share"]), n(r["force_btc"]), n(r["med_age_force"]),
                     n(r["ul_force"]), n(r["ul_btc"]), f1(100 * r["pub_fmt_legacy"]),
                     f1(100 * r["ul_fmt_taproot"])])
    tex_table("table_census", ["Year", "All force", "Coop.", "Force", "Force \\%", "Force BTC", "Age (d)", "Unlisted",
                               "Unl. BTC", "Pub. legacy \\%", "Unl. taproot \\%"],
              rows, "lrrrrrrrrrr",
              "Lightning channel closes on the chain, 2018--2026. All force: every force close, public or not. "
              "Public: channels in our list of every public channel. Unlisted: force closes of channels outside it, "
              "found by the commitment pattern.",
              "tab:census",
              note=r"\textsuperscript{a}January--October. \textsuperscript{b}From 12 January (block 503,816). Our list "
                   r"of public channels holds few of the channels that closed in 2018, so only the force-close total is "
                   r"shown. \textsuperscript{c}The list's coverage is still ramping up in January 2019. Age: median "
                   r"age at force close. Pub.\ legacy: share of public force closes with a pre-anchor commitment. "
                   r"Unl.\ taproot: share of unlisted force closes of simple taproot channels. Splices ("
                   + n(y["splice"].sum()) + ") are not shown.",
              wide=True)

    named = pairs[~pairs["has_unknown"] & (pairs["closes"] >= 300)].head(10)
    tex_table("table_client_pairs", ["Client pair", "Closes", "Force", "Force \\%"],
              [[r["pair"].replace("–", "--"), n(r["closes"]), n(r["force"]), f1(100 * r["force_share"])] for _, r in named.iterrows()],
              "lrrr", "Force share of public closes by the client pair at the two ends (closes from 8 May 2023).",
              "tab:client_pairs",
              note="Clients from paper 01's labels on the day of the close (high or medium confidence). The chain "
                   "does not show which side broadcast a force close, so a close is attributed to the pair.")

    va = validation(con)
    agree = va[((va["close_type"] == "mutual") & (va["reason"] == 1)) | ((va["close_type"] == "force") & va["reason"].isin([2, 3]))
               | ((va["close_type"] == "splice") & (va["reason"] == 1))]["k"].sum()
    print(va.to_string(), "agreement", agree / va["k"].sum())
    vc = validation_coverage(con)
    vc["coverage"] = vc["with_reason"] / vc["closes"]
    write_json("census.json", {
        "validation_mempool": {"rows": va.to_dict(orient="records"), "compared": int(va["k"].sum()),
                               "agree": int(agree),
                               "coverage_by_year": vc.to_dict(orient="records")},
        "yearly": y.to_dict(orient="records"),
        "monthly": m.assign(month=m["month"].dt.strftime("%Y-%m")).drop(columns=["h0"]).to_dict(orient="records"),
        "cohorts": co,
        "client_pairs": pairs.to_dict(orient="records"),
        "client_pair_top_node": pair_top_node(con),
        "mass_events": ev.to_dict(orient="records"),
        "mass_min": MASS_MIN,
        "unlisted_2024_09_01": {"closes": sep[0], "btc": sep[1], "legacy": sep[2], "first_height": sep[3],
                                "last_height": sep[4], "blocks": sep[4] - sep[3] + 1,
                                "hours_first_to_last_block": (sep_t[sep[4]] - sep_t[sep[3]]) / 3600,
                                "unlisted_force_closes_in_the_144_blocks_before": sep_before},
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
