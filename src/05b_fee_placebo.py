"""
05b_fee_placebo.py (RQ4, placebo): could ordinary weeks give the fee-spike results by chance?

Tests both parts of the §6.4 reading ("no wave of force closes when fees spike; at most a small delayed rise"):
  1. the event study: organic public force closes in weeks [0, +1] of a fee spike ÷ weeks [−4, −1] (05: 0.92), and
     here also weeks [+2, +3] ÷ [−4, −1], the window of the delayed rise;
  2. the lag profile of the elasticity (organic rate on the weekly median fee rate, year fixed effects + trend,
     Newey–West, exactly as 05): lead 1, same week, lags 1–4 (05: lags 2 and 3 about 0.07, p = 0.014).

Every placebo keeps the force-close series as observed and breaks only its timing against fees:
  A. Random weeks: as many placebo spike weeks as there are real spikes, each drawn from the same calendar year as a
     real spike, among ordinary weeks (not a spike week; window [−4, +3] clear of every real spike's; placebo windows
     clear of each other). Also without the year match. RANDOM_DRAWS draws. Event ratios only.
  B. Block shuffle: each year's weekly fee series is cut into blocks of BLOCK_WEEKS weeks (random phase) and the blocks
     are shuffled within the year. Keeps each year's fee level and distribution, spikes as episodes and the
     autocorrelation inside blocks. Lags are taken after the shuffle; the spike rule runs on the shuffled series.
     BLOCK_DRAWS draws; SENSITIVITY_BLOCKS as a check.
  C. Circular shift: the log fee rate's deviations from its year means and trend are rotated by every shift of at least
     MIN_SHIFT weeks and added back. Keeps the whole autocorrelation and every spike, moved in time. Exhaustive.
The delayed rise is also tested family-wise: the largest lag 1–4 coefficient against the largest in each placebo draw
(this replaces the Bonferroni correction and allows for the lags being correlated).

Two week frames. "panel" is 05's weekly table as it stands, so the real numbers are the published ones. "calendar"
lags by calendar week and takes the fee rate of weeks missing from the panel from blockstats. Since the 2018-2026
rerun, 05 itself keeps every calendar week of its frame (no block-count rule, zero-close weeks kept) and lags
by calendar week, so the two frames coincide; the calendar frame stays as a check.

Reads data/fees.json (05_fee_regimes.py) and data/work/blockstats.parquet only, and stops if they disagree.
Writes data/fees_placebo.json.
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RANDOM_SEED, SRC_DIR, WORK_DIR  # noqa: E402
from fcc.io import read_json, write_json  # noqa: E402


def _stage05():
    """05_fee_regimes.py, for its estimator and spike rule, so the placebo follows any change made there."""
    spec = importlib.util.spec_from_file_location("fee_regimes", SRC_DIR / "05_fee_regimes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FR = _stage05()

RANDOM_DRAWS = 10_000
BLOCK_DRAWS = 5_000
BLOCK_WEEKS = 8              # the spike rule's lookback; the fee rate's within-year autocorrelation is about 0.4 at 8 weeks
SENSITIVITY_BLOCKS = (4, 13)
SENSITIVITY_DRAWS = 2_000    # also the block draws for the calendar frame
MIN_SHIFT = 13               # keeps every placebo timing (shift − 1 … shift + 4) at least 8 weeks from the real ones
GAP = 8                      # event windows [−4, +3] do not overlap when weeks are at least 8 apart
TIMINGS = {"fee_lead1": -1, "baseline": 0, "fee_lag1": 1, "fee_lag2": 2, "fee_lag3": 3, "fee_lag4": 4}
DELAYED = ("fee_lag1", "fee_lag2", "fee_lag3", "fee_lag4")
BEFORE = (-4, -1)
WINDOWS = {"w0_1": (0, 1), "w2_3": (2, 3)}
SERIES = ("organic", "legacy", "anchor")
ALPHA = 0.05


# --- inputs ----------------------------------------------------------------------------------------------------------

def load() -> tuple[dict, pd.DataFrame, pd.DataFrame, list]:
    fj = read_json("fees.json")
    panel = pd.DataFrame(fj["weekly"])
    panel["key"] = panel["wk"]
    panel["wk"] = pd.to_datetime(panel["wk"], utc=True)
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    fee = con.execute(f"""
        SELECT strftime(date_trunc('week', to_timestamp(time))::DATE, '%Y-%m-%d') AS key, median(p50) AS p50,
               count(*) AS blocks
        FROM '{WORK_DIR / 'blockstats.parquet'}' WHERE height <= {fj['blockstats']['cutoff']}
        GROUP BY key ORDER BY key""").df()
    m = panel.merge(fee, on="key", how="left", suffixes=("", "_bs"))
    bad = m["p50_bs"].isna() | (m["p50"] != m["p50_bs"]) | (m["blocks"] != m["blocks_bs"])
    if bad.any():
        raise SystemExit(f"fees.json and blockstats.parquet disagree on {int(bad.sum())} weeks: rerun 05_fee_regimes.py")
    weeks = pd.date_range(panel["wk"].min(), panel["wk"].max(), freq="7D")
    cal = pd.DataFrame({"wk": weeks, "key": weeks.strftime("%Y-%m-%d")})
    cal = cal.merge(panel.drop(columns=["wk", "p50", "blocks", "y"]), on="key", how="left").merge(fee, on="key")
    cal["y"] = cal["wk"].dt.year
    gaps = cal.loc[~cal["key"].isin(panel["key"]), ["key", "p50", "blocks"]].to_dict(orient="records")
    for f in (panel, cal):
        if (f["p50"] <= 0).any():
            raise SystemExit("a weekly median fee rate is 0: the log-fee placebos need a positive series")
    return fj, panel.reset_index(drop=True), cal.reset_index(drop=True), gaps


# --- statistics shared by the real data and the placebos -------------------------------------------------------------

def lag_profile(f: pd.DataFrame, p50: np.ndarray, full: bool = False) -> dict:
    """05's elasticity of the organic force-close rate for each fee timing (lags by row of the frame, as 05)."""
    s = pd.Series(p50, index=f.index)
    out = {}
    for name, k in TIMINGS.items():
        r = FR.elasticity(f.assign(fee_x=s.shift(k)).dropna(subset=["fee_x"]), "organic", offset=True, fee="fee_x")
        out[name] = r if full else (r["b"], r["p"])
    return out


def spike_weeks(p50: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """05's spike rule (spikes()): every spike week, and the first week of each cluster."""
    s = pd.Series(p50)
    base = s.shift(1).rolling(FR.SPIKE_LOOKBACK, min_periods=FR.SPIKE_LOOKBACK).median()
    spike = (s >= FR.SPIKE_X * base) & (s >= FR.SPIKE_MIN)
    first = spike & ~spike.shift(1).rolling(FR.SPIKE_LOOKBACK, min_periods=1).max().fillna(0).astype(bool)
    return spike.to_numpy(), np.flatnonzero(first.to_numpy())


def window_means(f: pd.DataFrame) -> dict:
    """Mean weekly count in [−4, −1] and in each window after every week; NaN where a window leaves the frame or holds
    a dropped week."""
    n, out = len(f), {}
    for s in SERIES:
        x = f[s].to_numpy(float)
        out[s] = {}
        for name, (a, b) in {"before": BEFORE, **WINDOWS}.items():
            m = np.full(n, np.nan)
            for i in range(max(0, -a), n - max(0, b)):
                m[i] = x[i + a:i + b + 1].mean()
            out[s][name] = m
    return out


def ratios(wm: dict, idx: np.ndarray) -> dict:
    """Pooled event ratio, as 05: summed window means after ÷ summed window means before, over the spikes whose windows
    are complete. Returns {series_window: ratio}."""
    out = {}
    for s in SERIES:
        for w in WINDOWS:
            b, a = wm[s]["before"][idx], wm[s][w][idx]
            ok = np.isfinite(a) & np.isfinite(b)
            out[f"{s}_{w}"] = float(a[ok].sum() / b[ok].sum()) if ok.any() and b[ok].sum() > 0 else math.nan
    return out


def usable(wm: dict, idx: np.ndarray) -> np.ndarray:
    """Spikes 05 can use: complete [−4, −1] and [0, +1] windows."""
    return idx[np.isfinite(wm["organic"]["before"][idx]) & np.isfinite(wm["organic"]["w0_1"][idx])]


def place(real: float, null) -> dict:
    """Where the real value falls in the placebo distribution. p_upper: share of draws at least as large (with the
    real alignment counted as one draw); p_lower: at least as small."""
    x = np.asarray(null, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"real": real, "draws": 0}
    q = np.quantile(x, [0.025, 0.05, 0.5, 0.95, 0.975])
    return {"real": real, "draws": n, "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if n > 1 else 0.0,
            "q025": float(q[0]), "q05": float(q[1]), "q50": float(q[2]), "q95": float(q[3]), "q975": float(q[4]),
            "percentile": float(100 * (np.sum(x < real) + 0.5 * np.sum(x == real)) / n),
            "p_upper": float((1 + np.sum(x >= real)) / (1 + n)), "p_lower": float((1 + np.sum(x <= real)) / (1 + n))}


# --- placebo fee series ----------------------------------------------------------------------------------------------

def block_shuffle(p50: np.ndarray, years: np.ndarray, block: int, rng) -> np.ndarray:
    out = np.empty_like(p50)
    for y in np.unique(years):
        pos = np.flatnonzero(years == y)
        cut = int(rng.integers(0, block))       # random phase: the first block has `cut` weeks (none when 0)
        blocks = np.split(pos, [e for e in range(cut, len(pos), block) if 0 < e < len(pos)])
        out[pos] = p50[np.concatenate([blocks[j] for j in rng.permutation(len(blocks))])]
    return out


def deviations(f: pd.DataFrame, p50: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """log fee = year means + trend (fitted) + deviation."""
    X = pd.get_dummies(f["y"].astype(str), dtype=float)
    X["trend"] = (f["wk"] - f["wk"].min()).dt.days / 365.25
    X = X.to_numpy(float)
    lf = np.log(p50)
    fit = X @ np.linalg.lstsq(X, lf, rcond=None)[0]
    return fit, lf - fit


def random_weeks(f: pd.DataFrame, wm: dict, spike: np.ndarray, real: np.ndarray, rng, draws: int,
                 match_year: bool) -> tuple[np.ndarray, int]:
    """Placebo spike weeks (draws × number of real spikes), uniform over the sets that satisfy the rules. A real
    spike's year that cannot hold its placebo weeks draws from all years instead (counted as a fallback)."""
    n, years = len(f), f["y"].to_numpy()
    ok = ~spike
    for s in SERIES:
        for arr in wm[s].values():
            ok &= np.isfinite(arr)
    for s in real:
        ok[max(0, s - GAP + 1):s + GAP] = False
    every = np.flatnonzero(ok)

    def capacity(pool):         # most weeks at least GAP apart that the pool can hold
        k, last = 0, -GAP
        for i in pool:
            if i - last >= GAP:
                k, last = k + 1, i
        return k

    need = pd.Series(years[real]).value_counts().to_dict()
    pools, fallback = [], 0
    for s in real:
        p = np.flatnonzero(ok & (years == years[s])) if match_year else every
        if match_year and capacity(p) < need[years[s]]:
            p, fallback = every, fallback + 1
        pools.append(p)
    got, parts, tries = 0, [], 0
    while got < draws:
        m = max(2_000, 2 * (draws - got))
        cand = np.column_stack([rng.choice(p, size=m) for p in pools])
        good = np.all(np.diff(np.sort(cand, axis=1), axis=1) >= GAP, axis=1) if len(real) > 1 else np.ones(m, bool)
        parts.append(cand[good])
        got += int(good.sum())
        tries += m
        if tries > 200 * draws and got < draws:
            raise SystemExit("random-week placebo: too few valid week sets; loosen GAP or drop the year match")
    return np.concatenate(parts)[:draws], fallback


# --- the placebos ----------------------------------------------------------------------------------------------------

def lag_null(f: pd.DataFrame, wm: dict, series_iter, real: dict) -> dict:
    """Lag profile and event ratios for each placebo fee series; summary against the real values."""
    b = {k: [] for k in TIMINGS}
    p = {k: [] for k in TIMINGS}
    ev = {k: [] for k in real["ratios"]}
    nsp = []
    for x in series_iter:
        prof = lag_profile(f, x)
        for k, (bk, pk) in prof.items():
            b[k].append(bk)
            p[k].append(pk)
        idx = usable(wm, spike_weeks(x)[1])
        nsp.append(len(idx))
        for k, v in ratios(wm, idx).items():
            ev[k].append(v)
    b = {k: np.array(v) for k, v in b.items()}
    p = {k: np.array(v) for k, v in p.items()}
    rb = {k: real["lags"][k]["b"] for k in TIMINGS}
    rp = {k: real["lags"][k]["p"] for k in TIMINGS}
    nsp = np.array(nsp)
    return {
        "draws": len(nsp),
        "lags": {k: {**place(rb[k], b[k]), "share_hac_p_below_0.05": float(np.mean(p[k] < ALPHA)),
                     "share_hac_p_at_or_below_real": float(np.mean(p[k] <= rp[k]))} for k in TIMINGS},
        "max_lag1_4": place(max(rb[k] for k in DELAYED), np.max([b[k] for k in DELAYED], axis=0)),
        "max_all_timings": place(max(rb.values()), np.max([b[k] for k in TIMINGS], axis=0)),
        "mean_lag2_3": place((rb["fee_lag2"] + rb["fee_lag3"]) / 2, (b["fee_lag2"] + b["fee_lag3"]) / 2),
        "spikes_per_draw": {"mean": float(nsp.mean()), "min": int(nsp.min()), "max": int(nsp.max()),
                            "share_none": float(np.mean(nsp == 0)), "real": real["n_spikes"]},
        "event_ratios": {k: place(real["ratios"][k], v) for k, v in ev.items()},
    }


def real_values(f: pd.DataFrame, wm: dict) -> dict:
    p50 = f["p50"].to_numpy(float)
    flag, first = spike_weeks(p50)
    idx = usable(wm, first)
    per_spike = [{"week": f.loc[i, "key"], "p50": float(p50[i]),
                  **{f"{s}_{w}": float(wm[s][w][i]) for s in SERIES for w in ("before", *WINDOWS)}} for i in idx]
    return {"lags": lag_profile(f, p50, full=True), "spike_flag": flag, "idx": idx, "n_spikes": len(idx),
            "spike_weeks": f.loc[idx, "key"].tolist(), "per_spike": per_spike, "ratios": ratios(wm, idx),
            "spikes_with_delayed_window": int(np.sum(np.isfinite(wm["organic"]["w2_3"][idx])))}


def analyse(name: str, f: pd.DataFrame, block_draws: int, sensitivity: bool) -> dict:
    t0 = time.time()
    wm = window_means(f)
    real = real_values(f, wm)
    p50, years = f["p50"].to_numpy(float), f["y"].to_numpy()
    out = {"weeks": len(f), "real": {k: v for k, v in real.items() if k not in ("spike_flag", "idx")}}

    out["random_weeks"] = {}
    for label, match in (("year_matched", True), ("any_year", False)):
        rng = np.random.default_rng(RANDOM_SEED)
        sets, fallback = random_weeks(f, wm, real["spike_flag"], real["idx"], rng, RANDOM_DRAWS, match)
        res = {"draws": len(sets), "seed": RANDOM_SEED, "year_fallbacks": fallback, "event_ratios": {}}
        for s in SERIES:
            for w in WINDOWS:
                null = wm[s][w][sets].sum(axis=1) / wm[s]["before"][sets].sum(axis=1)
                res["event_ratios"][f"{s}_{w}"] = place(real["ratios"][f"{s}_{w}"], null)
        out["random_weeks"][label] = res
    print(f"  [{name}] random weeks done ({time.time() - t0:.0f} s)", flush=True)

    out["block_shuffle"] = {}
    for block in (BLOCK_WEEKS, *(SENSITIVITY_BLOCKS if sensitivity else ())):
        draws = block_draws if block == BLOCK_WEEKS else SENSITIVITY_DRAWS
        rng = np.random.default_rng(RANDOM_SEED)
        it = (block_shuffle(p50, years, block, rng) for _ in range(draws))
        out["block_shuffle"][f"block_{block}w"] = {"block_weeks": block, "seed": RANDOM_SEED, **lag_null(f, wm, it, real)}
        print(f"  [{name}] block shuffle {block} w done ({time.time() - t0:.0f} s)", flush=True)

    fit, dev = deviations(f, p50)
    shifts = range(MIN_SHIFT, len(f) - MIN_SHIFT + 1)
    it = (np.exp(fit + np.roll(dev, s)) for s in shifts)
    out["circular_shift"] = {"min_shift": MIN_SHIFT, "shifts": f"{MIN_SHIFT}..{len(f) - MIN_SHIFT}",
                             **lag_null(f, wm, it, real)}
    print(f"  [{name}] circular shift done ({time.time() - t0:.0f} s)", flush=True)
    return out


def verdict(a: dict) -> dict:
    """The reading the abstract's lead needs, by a rule fixed before the run: the delayed rise counts only if the family-wise
    test (largest lag 1–4 coefficient) rejects at 5% against both fee-series placebos; a wave counts only if the
    [0, +1] ratio is above 95% of random ordinary weeks of the same years."""
    fw = {"block_shuffle": a["block_shuffle"][f"block_{BLOCK_WEEKS}w"]["max_lag1_4"]["p_upper"],
          "circular_shift": a["circular_shift"]["max_lag1_4"]["p_upper"]}
    rw = a["random_weeks"]["year_matched"]["event_ratios"]["organic_w0_1"]
    rw23 = a["random_weeks"]["year_matched"]["event_ratios"]["organic_w2_3"]
    delayed = all(v < ALPHA for v in fw.values())
    return {"family_wise_p_max_lag1_4": fw, "delayed_rise_survives": delayed,
            "wave_p_upper_random_weeks": rw["p_upper"], "wave_detected": rw["p_upper"] < ALPHA,
            "delayed_window_p_upper_random_weeks": rw23["p_upper"],
            "smallest_detectable_wave": rw["q95"] / rw["q50"] - 1,
            "real_vs_typical_week": rw["real"] / rw["q50"] - 1,
            "lead": ("no wave; at most a small delayed rise" if delayed else "no wave (the delayed rise is within placebo noise)")}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)) and not np.isfinite(o):
        return None
    return o


def main() -> int:
    t0 = time.time()
    fj, panel, cal, gaps = load()
    print(f"panel: {len(panel)} weeks {panel['key'].iloc[0]}..{panel['key'].iloc[-1]}, years "
          f"{sorted(panel['y'].unique().tolist())}; calendar: {len(cal)} weeks, {len(gaps)} dropped by 05: {gaps}")
    res = {"panel": analyse("panel", panel, BLOCK_DRAWS, sensitivity=True),
           "calendar": analyse("calendar", cal, SENSITIVITY_DRAWS, sensitivity=False)}

    pr = res["panel"]["real"]
    check = {"lags_match_05": all(abs(pr["lags"][k]["b"] - fj["robustness"][k]["b"]) < 1e-9 for k in TIMINGS),
             "spike_weeks_match_05": pr["spike_weeks"] == [s["week"] for s in fj["spikes"]],
             "ratio_match_05": abs(pr["ratios"]["organic_w0_1"] - fj["spike_ratio"]["organic"]) < 1e-9}
    if not all(check.values()):
        print(f"WARNING: the panel frame does not reproduce 05: {check}")

    for fr in ("panel", "calendar"):
        a = res[fr]
        print(f"\n== {fr} ({a['weeks']} weeks, {a['real']['n_spikes']} spikes: {a['real']['spike_weeks']})")
        rows = {}
        for k in TIMINGS:
            rows[k] = {"b": a["real"]["lags"][k]["b"], "hac_p": a["real"]["lags"][k]["p"]}
            for nl, sub in (("shuffle", a["block_shuffle"][f"block_{BLOCK_WEEKS}w"]), ("shift", a["circular_shift"])):
                rows[k][f"pct_{nl}"] = sub["lags"][k]["percentile"]
                rows[k][f"p_{nl}"] = sub["lags"][k]["p_upper"]
                rows[k][f"q975_{nl}"] = sub["lags"][k]["q975"]
        print(pd.DataFrame(rows).T.round(3).to_string())
        for stat in ("max_lag1_4", "max_all_timings", "mean_lag2_3"):
            print(f"{stat}: real {a['circular_shift'][stat]['real']:.3f}; p shuffle "
                  f"{a['block_shuffle'][f'block_{BLOCK_WEEKS}w'][stat]['p_upper']:.3f}, p shift "
                  f"{a['circular_shift'][stat]['p_upper']:.3f}")
        if "block_4w" in a["block_shuffle"]:
            print("sensitivity (block length): " + ", ".join(
                f"{k}: lag2 p {v['lags']['fee_lag2']['p_upper']:.3f}, max p {v['max_lag1_4']['p_upper']:.3f}"
                for k, v in a["block_shuffle"].items()))
        ev = {}
        for k, real in a["real"]["ratios"].items():
            ev[k] = {"real": real}
            for nl, sub in (("rw_year", a["random_weeks"]["year_matched"]), ("rw_any", a["random_weeks"]["any_year"]),
                            ("shuffle", a["block_shuffle"][f"block_{BLOCK_WEEKS}w"]), ("shift", a["circular_shift"])):
                e = sub["event_ratios"][k]
                ev[k][f"q50_{nl}"] = e.get("q50")
                ev[k][f"pct_{nl}"] = e.get("percentile")
        print(pd.DataFrame(ev).T.round(3).to_string())
        a["verdict"] = verdict(a)
        print(f"verdict: {a['verdict']}")

    write_json("fees_placebo.json", _clean({
        "design": {
            "seed": RANDOM_SEED, "random_week_draws": RANDOM_DRAWS, "block_shuffle_draws": BLOCK_DRAWS,
            "block_weeks": BLOCK_WEEKS, "sensitivity_blocks": list(SENSITIVITY_BLOCKS),
            "sensitivity_draws": SENSITIVITY_DRAWS, "calendar_block_draws": SENSITIVITY_DRAWS,
            "min_shift_weeks": MIN_SHIFT, "min_gap_weeks": GAP, "timings": TIMINGS, "delayed_family": list(DELAYED),
            "event_windows": {"before": list(BEFORE), **{k: list(v) for k, v in WINDOWS.items()}},
            "spike_rule": {"x": FR.SPIKE_X, "min_sat_vb": FR.SPIKE_MIN, "lookback_weeks": FR.SPIKE_LOOKBACK},
            "p_values": "p_upper = (1 + draws >= real) / (1 + draws); percentile = share of draws below the real value"},
        "inputs": {"fees_json_weeks": len(panel), "first_week": panel["key"].iloc[0], "last_week": panel["key"].iloc[-1],
                   "years": sorted(panel["y"].unique().tolist()), "blockstats": fj["blockstats"],
                   "fees_json_matches_blockstats": True, "weeks_dropped_by_05": gaps},
        "check_05": check, "panel": res["panel"], "calendar": res["calendar"],
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "statsmodels": statsmodels.__version__,
                     "duckdb": duckdb.__version__},
        "runtime_s": round(time.time() - t0, 1),
    }))
    print(f"\nwrote data/fees_placebo.json ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
