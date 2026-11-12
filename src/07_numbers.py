"""
07_numbers.py: every number quoted in the text, as LaTeX macros, recomputed from the stage outputs.

Reads data/{snapshot_manifest,census,costs,anchors,zero_fee,fees,sweepers,htlc,lifecycle,taproot_anchor_paths,
fees_placebo,breach_wave,pre_anchor_2026}.json
(pre_anchor_2026 from stage 02b, taproot_anchor_paths from 04c, breach_wave from 04d, fees_placebo from 05b, htlc
from 08, lifecycle from 09; those six are optional here, but the paper needs all of them)
and writes
  data/supporting_numbers.json   name -> value (as printed)
  paper/numbers.tex              one \\newcommand per number

Usage:  python src/07_numbers.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, PAPER_DIR, PUBLIC_FROM, coverage  # noqa: E402

PUBLIC_YEAR = int(PUBLIC_FROM[:4])      # public / cooperative / share / rate numbers from 2019


def load(name: str):
    p = DATA_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def num(x) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}"


def dec(x, nd: int = 1) -> str:
    return f"{x:.{nd}f}"


def long_date(d: str) -> str:
    t = pd.Timestamp(d)
    return f"{t.day}~{t:%B}~{t.year}"


WORDS = {2018: "Eighteen", 2019: "Nineteen", 2020: "Twenty", 2021: "TwentyOne", 2022: "TwentyTwo", 2023: "TwentyThree", 2024: "TwentyFour", 2025: "TwentyFive",
         2026: "TwentySix"}


def main() -> int:
    man, ce, co, an, zf = (load(k) for k in ("snapshot_manifest", "census", "costs", "anchors", "zero_fee"))
    fe, sw = load("fees"), load("sweepers")
    N = {}

    # --- data ---------------------------------------------------------------------------------------------------
    N["CutoffHeight"] = num(man["cutoff_height"])
    N["CutoffDate"] = long_date("2026-10-31")
    N["ScanStartHeight"] = num(man["scan_start_height"])
    N["ScanStartDate"] = long_date("2018-01-12")
    N["NPublicCloses"] = num(man["counts"]["closes"])
    N["NUnlistedForce"] = num(man["counts"]["ul_closes"])
    N["NForceAll"] = num(man["counts"]["force_closes"])
    N["NFundingOutpoints"] = num(man["counts"]["funding_outpoints"])
    N["NOutputs"] = num(man["counts"]["force_close_outputs"])
    va = ce["validation_mempool"]
    N["ValCompared"] = num(va["compared"])
    N["ValAgreePct"] = pct(va["agree"] / va["compared"], 2)

    # --- census -------------------------------------------------------------------------------------------------
    y_all = pd.DataFrame(ce["yearly"]).set_index("y")
    N["NAllForce"] = num(y_all["all_force"].sum())                  # every force close, 2018-2026 (complete)
    for yr, w in WORDS.items():
        if yr in y_all.index:
            N[f"AllForce{w}"] = num(y_all.loc[yr, "all_force"])
    # inventory of the data (Data section): every year, 2018 included
    N["NPublicForce"] = num(y_all["force"].sum())
    N["NPublicMutual"] = num(y_all["mutual"].sum())
    y = y_all[y_all.index >= PUBLIC_YEAR]                            # the public series from 2019
    N["ForceBtcAll"] = num(y["force_btc"].sum())
    N["UnlistedBtcAll"] = num(y["ul_btc"].sum())
    for yr, w in WORDS.items():
        if yr in y.index:
            r = y.loc[yr]
            N[f"ForceShare{w}"] = pct(r["force_share"], 0)
            N[f"PubForce{w}"] = num(r["force"])
            N[f"UnlForce{w}"] = num(r["ul_force"])
            N[f"PubLegacyPct{w}"] = pct(r["pub_fmt_legacy"], 0)
    N["UnlTaprootPctTwentySix"] = pct(y.loc[2026, "ul_fmt_taproot"], 0)
    N["UnlOverPubTwentySix"] = dec(y.loc[2026, "ul_force"] / y.loc[2026, "force"], 1)
    m = pd.DataFrame(ce["monthly"])
    m["y"] = m["month"].str[:4].astype(int)
    m = m[m["y"] >= PUBLIC_YEAR]
    g = m.groupby("y").agg(f=("force_organic", "sum"), o=("open", "mean"), k=("month", "size"))
    rate = 1000 * g["f"] / g["o"] / g["k"]
    N["OrganicRateTwentyTwo"] = dec(rate.loc[2022], 0)
    N["OrganicRateTwentySix"] = dec(rate.loc[2026], 0)
    N["OpenAtCutoff"] = num(m["open"].iloc[-1])
    N["OpenPeak"] = num(m["open"].max())
    ev = pd.DataFrame(ce["mass_events"])
    N["MassMin"] = num(ce["mass_min"])
    ev = ev[ev["y"] >= PUBLIC_YEAR]
    N["MassSharePct"] = pct(ev["closes_in_events"].sum() / y["force"].sum(), 0)
    share = (ev.set_index("y")["closes_in_events"] / y["force"]).dropna()
    N["MassShareMin"] = pct(share.min(), 0)
    N["MassShareMax"] = pct(share.max(), 0)
    for yr, d in ce["cohorts"].items():
        if int(yr) < PUBLIC_YEAR:
            continue
        i = d["age_days"].index(364)
        N[f"CohortOneYear{WORDS[int(yr)]}"] = pct(d["force"][i], 0)
    ue = ce["unlisted_2024_09_01"]
    N["SepEventCloses"] = num(ue["closes"])
    N["SepEventBtc"] = dec(ue["btc"], 1)
    N["SepEventBlocks"] = num(ue["blocks"])                            # first to last block of the burst
    N["SepEventHours"] = num(ue["hours_first_to_last_block"])
    vc = pd.DataFrame(ce["validation_mempool"]["coverage_by_year"]).set_index("y")
    vc = vc.loc[PUBLIC_YEAR:2025, "coverage"]                          # mempool.space gives no reason for 2026
    N["ValCoverageYearMinPct"] = pct(vc.min(), 0)
    N["ValCoverageYearMaxPct"] = pct(vc.max(), 0)
    cp = pd.DataFrame(ce["client_pairs"]).set_index("pair")
    for pair, key in (("LND–LND", "LndLnd"), ("CLN–LND", "ClnLnd"), ("LDK–LDK", "LdkLdk"), ("Eclair–LND", "EclairLnd")):
        if pair in cp.index:
            N[f"PairForcePct{key}"] = pct(cp.loc[pair, "force_share"], 0)
            N[f"PairCloses{key}"] = num(cp.loc[pair, "closes"])

    # --- costs --------------------------------------------------------------------------------------------------
    c = pd.DataFrame(co["by_year"]).set_index("y")
    c = c[c.index >= PUBLIC_YEAR]
    for yr, w in WORDS.items():
        if yr not in c.index:
            continue
        r = c.loc[yr]
        N[f"CostMed{w}"] = num(r["med_party_fee"])
        N[f"CoopMed{w}"] = num(r["fee_med"])
        N[f"CostRatio{w}"] = dec(r["ratio_med"], 1)
        N[f"CostTotalBtc{w}"] = dec(r["total_party_btc"], 2)
    N["CostRatioMin"] = dec(c["ratio_med"].min(), 0)
    N["CostRatioMax"] = dec(c["ratio_med"].max(), 0)
    N["VbRatioMin"] = dec(c.loc[2022:, "ratio_vb"].min(), 1)
    N["VbRatioMax"] = dec(c.loc[2022:, "ratio_vb"].max(), 1)
    N["CostTotalBtcAll"] = dec(co["all_years"]["total_party_btc"], 1)
    N["ToLocalMedDays"] = dec(co["all_years"]["med_to_local_days"], 1)
    N["ToLocalPNinetyMax"] = num(c["p90_to_local_days"].max())
    N["DelayOneDayPct"] = pct(co["all_years"]["share_to_self_delay_144"], 0)
    N["DelayTwoWeeksPct"] = pct(co["all_years"]["share_to_self_delay_ge_2015"], 0)
    N["BtcDaysLockedAll"] = num(c["btc_days_locked"].sum())
    # Break-even rate: the yearly return on the closers' locked balances at which a year's
    # BTC-days would cost as much as that year's force-close fees, i.e. fees / (BTC-days / 365). It sets the wait
    # against the fees without assuming a cost of capital.
    be = 100 * c["total_party_btc"] / (c["btc_days_locked"] / 365)
    N["BreakEvenRateAll"] = dec(100 * c["total_party_btc"].sum() / (c["btc_days_locked"].sum() / 365), 1)
    N["BreakEvenRateTwentyFive"] = dec(be.loc[2025], 1)
    N["BreakEvenRateTwentySix"] = dec(be.loc[2026], 1)
    N["BreakEvenRateEarlyMin"] = dec(be.loc[:2024].min(), 1)
    N["BreakEvenRateEarlyMax"] = dec(be.loc[:2024].max(), 1)
    if not be.loc[2025:].max() < be.loc[:2024].min():
        raise SystemExit("the text says the wait outweighs the fees only in the low-fee years 2025-26")
    bf = pd.DataFrame(co["by_format"])
    bf = bf[bf["y"] >= PUBLIC_YEAR]
    a23 = bf[(bf["y"] == 2023) & (bf["fmt"] == "anchor")].iloc[0]
    N["AnchorBumpedPctTwentyThree"] = pct(a23["share_bumped"], 0)
    N["AnchorVbMed"] = num(bf[bf["fmt"] == "anchor"]["med_vb"].median())
    N["LegacyVbMed"] = num(bf[bf["fmt"] == "legacy"]["med_vb"].median())
    pen = pd.DataFrame(co["penalties"]).set_index("y")
    N["PenaltyTxsAll"] = num(pen["txs"].sum())
    N["PenaltyBtcAll"] = dec(pen["btc"].sum(), 1)

    # --- anchors ------------------------------------------------------------------------------------------------
    fa = pd.DataFrame(an["fate"]).groupby("y").sum(numeric_only=True)
    tot = fa[["owner_bump", "owner_cleanup", "third_fast", "third_late", "unspent"]].sum(axis=1)
    N["AnchorsAll"] = num(fa["anchors"].sum())
    N["AnchorsUnspent"] = num(fa["unspent"].sum())
    N["OwnerPctTwentyTwo"] = pct((fa.loc[2022, "owner_bump"] + fa.loc[2022, "owner_cleanup"]) / tot.loc[2022], 0)
    N["OwnerPctTwentySix"] = pct((fa.loc[2026, "owner_bump"] + fa.loc[2026, "owner_cleanup"]) / tot.loc[2026], 0)
    fd = pd.DataFrame(an["fate"])
    pub = fd[fd["listed"]].set_index("y")
    N["ThirdMedDelayTwentyThree"] = num(pub.loc[2023, "third_med_delay"])
    N["ThirdMedDelayTwentyFour"] = num(pub.loc[2024, "third_med_delay"])
    N["ThirdAtSixteenTwentySix"] = pct(pub.loc[2026, "third_at_16"], 0)
    e = pd.DataFrame(an["sweep_economics"]).set_index("y")
    N["MinersShareTwentyTwo"] = pct(e.loc[2022, "miners_share"], 0)
    N["MinersShareTwentySix"] = pct(e.loc[2026, "miners_share"], 0)
    N["SweepTxsAll"] = num(an["totals"]["sweep_txs"])
    N["AnchorsSweptAll"] = num(an["totals"]["anchors_swept"])
    N["SweepValueBtc"] = dec(an["totals"]["value_btc"], 2)
    N["SweepMinersBtc"] = dec(an["totals"]["miners_btc"], 2)
    N["SweepSweepersBtc"] = dec(an["totals"]["sweepers_btc"], 2)
    N["BreakEven"] = dec(an["breakeven_sat_vb"], 2)
    N["SweepMedFrTwentyTwo"] = dec(e.loc[2022, "med_fr"], 1)
    N["SweepMedFrTwentyFour"] = dec(e.loc[2024, "med_fr"], 1)
    N["TaprootAnchors"] = num(an["taproot_anchors"]["outputs"])
    N["TaprootAnchorsSpent"] = num(an["taproot_anchors"]["spent"])
    N["TaprootAnchorsSpentPct"] = pct(an["taproot_anchors"]["spent"] / an["taproot_anchors"]["outputs"], 0)
    bu = pd.DataFrame(an["bumps"])
    b23 = bu[bu["y"] == 2023].set_index("htlc")
    N["BumpPctHtlcTwentyThree"] = pct(b23.loc[True, "bumped"] / b23.loc[True, "closes"], 0)
    N["BumpPctNoHtlcTwentyThree"] = pct(b23.loc[False, "bumped"] / b23.loc[False, "closes"], 0)

    # --- zero-fee -----------------------------------------------------------------------------------------------
    z, cf = zf["census"], zf["counterfactual"]["totals"]
    N["ZfCloses"] = num(z["closes"])
    N["ZfPublic"] = num(z["public"])
    N["ZfUnlisted"] = num(z["unlisted"])
    N["ZfFirst"] = long_date(z["first"])
    N["ZfFirstPublic"] = long_date(z["first_public"])
    N["ZfPublicNodes"] = num(z["public_nodes"])
    N["ZfPkgFr"] = dec(z["med_pkg_fr"], 1)
    N["CfAnchorVb"] = num(cf["anchor_vb_per_close"])
    N["CfPtwoaVb"] = num(cf["p2a_vb_per_close"])
    N["CfSavingPct"] = pct(1 - cf["p2a_vb"] / cf["anchor_vb"], 0)
    N["CfAnchorBtc"] = dec(cf["anchor_value_btc"], 2)
    N["CfBumpedPct"] = pct(cf["share_bumped"], 0)

    # --- fee regimes (05) ---------------------------------------------------------------------------------------
    if fe:
        el = fe["elasticity"]
        for k, w in (("organic_rate", "All"), ("legacy", "Legacy"), ("anchor", "Anchor"), ("unlisted", "Unlisted")):
            N[f"Elast{w}"] = dec(el[k]["b"], 2)
            N[f"Elast{w}Lo"] = dec(el[k]["lo"], 2)
            N[f"Elast{w}Hi"] = dec(el[k]["hi"], 2)
        # the same elasticities as "what doubling the fee rate does": 2**b - 1, signed
        sgn = lambda x: ("+" if x >= 0 else "\\textminus{}") + dec(abs(x), 1)
        for k, w in (("organic_rate", "All"), ("legacy", "Legacy"), ("anchor", "Anchor")):
            N[f"ElastDouble{w}"] = sgn(100 * (2 ** el[k]["b"] - 1))
            N[f"ElastDouble{w}Lo"] = sgn(100 * (2 ** el[k]["lo"] - 1))
            N[f"ElastDouble{w}Hi"] = sgn(100 * (2 ** el[k]["hi"] - 1))
        rb = fe["robustness"]
        N["ElastLagTwo"] = dec(rb["fee_lag2"]["b"], 2)
        N["ElastLagTwoP"] = dec(rb["fee_lag2"]["p"], 3)
        N["ElastDoubleLagTwo"] = sgn(100 * (2 ** rb["fee_lag2"]["b"] - 1))
        N["ElastDoubleLagTwoLo"] = sgn(100 * (2 ** rb["fee_lag2"]["lo"] - 1))
        N["ElastDoubleLagTwoHi"] = sgn(100 * (2 ** rb["fee_lag2"]["hi"] - 1))
        N["ElastMonthly"] = dec(rb["monthly"]["b"], 2)
        N["NSpikes"] = num(len(fe["spikes"]))
        for k, w in (("organic", "All"), ("legacy", "Legacy"), ("anchor", "Anchor"), ("unlisted", "Unlisted")):
            N[f"SpikeRatio{w}"] = dec(fe["spike_ratio"][k], 2)
        bill = pd.DataFrame(fe["closing_bill"]).set_index("y")
        N["BillForceBtcEighteen"] = dec(bill.loc[2018, "force_btc"], 1)
        bill = bill[bill.index >= PUBLIC_YEAR]
        for yr, w in WORDS.items():
            if yr in bill.index:
                N[f"BillBtc{w}"] = dec(bill.loc[yr, "ln_fee_btc"], 1)
                N[f"BillFeePct{w}"] = pct(bill.loc[yr, "fee_share"], 2)
        N["BillVbPctMax"] = pct(bill["vb_share"].max(), 2)
        gs = pd.DataFrame(fe["fee_guessing"])
        lg = gs[gs["fmt"] == "legacy"].set_index("y")
        N["LegacyOverTwoPctTwentyOne"] = pct(lg.loc[2021, "share_over_2x"], 0)
        N["LegacyOverTwoPctTwentySix"] = pct(lg.loc[2026, "share_over_2x"], 0)
        N["LegacyOverBtcAll"] = dec(lg["over_btc"].sum(), 1)
        sp = pd.DataFrame(fe["spikes"]).set_index("week")
        if "2023-05-01" in sp.index:
            may = sp.loc["2023-05-01"]
            N["SpikeMayTwentyThreeFee"] = num(may["p50"])
            N["SpikeMayTwentyThreeFeeBefore"] = num(may["p50_before"])
            N["SpikeMayTwentyThreePct"] = pct(may["organic_after"] / may["organic_before"] - 1, 0)
        if "2024-04-08" in sp.index:
            apr = sp.loc["2024-04-08"]
            N["SpikeAprTwentyFourPct"] = pct(1 - apr["organic_after"] / apr["organic_before"], 0)
        if "2019-03-04" in sp.index:
            mar = sp.loc["2019-03-04"]
            N["SpikeMarNineteenPct"] = pct(mar["organic_after"] / mar["organic_before"] - 1, 0)
        up = int((sp["organic_after"] > sp["organic_before"]).sum())      # spikes followed by more organic closes
        N["NSpikesUp"] = num(up)
        N["NSpikesDown"] = num(len(sp) - up)
        N["ElastWeeks"] = num(el["organic_rate"]["weeks"])
        N["LegacyMedRatioTwentySix"] = dec(lg.loc[2026, "med_ratio"], 1)
        N["LegacyMedRatioTwentyThree"] = dec(lg.loc[2023, "med_ratio"], 1)
        an_g = gs[gs["fmt"] == "anchor"].set_index("y")
        N["AnchorBelowPTenPctTwentyThree"] = pct(an_g.loc[2023, "share_below_p10"], 0)
        st = pd.DataFrame(fe["sweeper_timing"]).set_index("y")
        N["SweepPFiftyAtSixteenTwentyThree"] = num(st.loc[2023, "med_p50_at_16"])
        N["SweepPFiftyAtSweepTwentyThree"] = num(st.loc[2023, "med_p50_at_sweep"])
        N["SweepCheaperPctTwentyThree"] = pct(st.loc[2023, "share_sweep_block_cheaper"], 0)

    # --- sweepers (04b) -----------------------------------------------------------------------------------------
    if sw:
        N["SweepOperators"] = num(sw["operators"])
        N["SweepLargestPct"] = pct(sw["largest_share"], 0)
        N["SweepLargestFirst"] = long_date(sw["largest_first"])
        N["SweepLargestNetBtc"] = dec(sw["largest_net_btc"], 3)
        N["SweepBurnPct"] = pct(sw["burn_share"], 0)
        pw = sw["penalty_wave"]
        N["PenaltyWaveTxs"] = num(pw["txs"])
        N["PenaltyWaveBtc"] = dec(pw["outputs_btc"], 2)
        N["PenaltyWaveRecipients"] = num(pw["recipient_addresses"])

    more_numbers(N, man, ce, co, an, zf, fe, sw, load("htlc"), load("lifecycle"), load("taproot_anchor_paths"),
                 load("fees_placebo"))

    lines = ["% Generated by src/07_numbers.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(N.items())]
    (PAPER_DIR / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (DATA_DIR / "supporting_numbers.json").write_text(json.dumps(N, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{len(N)} numbers -> paper/numbers.tex")
    return 0


def more_numbers(N, man, ce, co, an, zf, fe, sw, ht, lc, tp=None, fp=None) -> None:
    """Numbers the full draft needs beyond the first set: HTLC deadlines (08), lifecycle cost (09) and a
    few more per-year values from the other stages. Same JSON inputs; no lake reads."""
    sgn = lambda x: ("+" if x >= 0 else "\\textminus{}") + dec(abs(x), 1)
    pv = lambda p: "<0.001" if p < 0.001 else dec(p, 3)         # p-values; use in math mode
    d2 = lambda x: dec(0.0 if round(x, 2) == 0 else x, 2)       # two decimals, no "-0.00"
    pc = lambda x: ">99" if x >= 99.5 else dec(x, 0)             # placebo percentiles; use in math mode

    # --- data ---------------------------------------------------------------------------------------------------
    cnt = man["counts"]
    N["NOutputsSpent"] = num(cnt["force_close_outputs_spent"])
    n_all = cnt["closes"] + cnt["ul_closes"]                   # public closes + force closes outside the list
    N["NClosesAll"] = num(n_all)
    N["NClosesTitle"] = num(round(n_all, -3))                  # the title's N, to the nearest thousand
    N["ValCoveragePct"] = pct(ce["validation_mempool"]["compared"] / cnt["closes"], 0)

    # --- census -------------------------------------------------------------------------------------------------
    y = pd.DataFrame(ce["yearly"]).set_index("y")
    m = pd.DataFrame(ce["monthly"])
    m["y"] = m["month"].str[:4].astype(int)
    g = m.groupby("y").agg(f=("force_organic", "sum"), o=("open", "mean"), k=("month", "size"))
    rate = 1000 * g["f"] / g["o"] / g["k"]
    for yr, w in WORDS.items():
        if yr < PUBLIC_YEAR:                                    # public series from 2019
            continue
        if yr in rate.index:
            N[f"OrganicRate{w}"] = dec(rate.loc[yr], 0)
        if yr in y.index:
            N[f"MutualBtc{w}"] = num(y.loc[yr, "mutual_btc"])
    mi = m.set_index("month")
    N["RateAugTwentySix"] = dec(mi.loc["2026-08", "rate"], 0)
    N["RateOrganicAugTwentySix"] = dec(mi.loc["2026-08", "rate_organic"], 0)
    # February-March 2019 rates against every later month (list coverage ramping?)
    N["RateOrganicFebNineteen"] = dec(mi.loc["2019-02", "rate_organic"], 0)
    N["RateOrganicMarNineteen"] = dec(mi.loc["2019-03", "rate_organic"], 0)
    N["RateOrganicMaxLater"] = dec(mi.loc[mi.index >= "2019-04", "rate_organic"].max(), 0)
    N["OpenJulTwentySix"] = num(mi.loc["2026-07", "open"])
    ev = pd.DataFrame(ce["mass_events"])
    ev = ev[ev["y"] >= PUBLIC_YEAR]                             # public series from 2019
    N["MassEventsMin"] = num(ev["events"].min())
    N["MassEventsMax"] = num(ev["events"].max())
    N["MassLargest"] = num(ev["largest"].max())
    N["SepEventLegacy"] = num(ce["unlisted_2024_09_01"]["legacy"])
    N["PairLdkLdkTopNodePct"] = pct(ce["client_pair_top_node"]["LDK–LDK"]["top_node_share"], 1)
    cp = pd.DataFrame(ce["client_pairs"]).set_index("pair")
    for pair, key in (("LDK–LND", "LdkLnd"), ("CLN–CLN", "ClnCln"), ("LND–Unknown", "LndUnknown")):
        N[f"PairForcePct{key}"] = pct(cp.loc[pair, "force_share"], 0)
        N[f"PairCloses{key}"] = num(cp.loc[pair, "closes"])
    N["PairForcePctLdkLdkExact"] = pct(cp.loc["LDK–LDK", "force_share"], 1)

    # --- costs --------------------------------------------------------------------------------------------------
    c = pd.DataFrame(co["by_year"]).set_index("y")
    parts = ["sum_commitment", "sum_bump", "sum_htlc", "sum_sweeps", "sum_anchor_cleanup"]
    tot = c[parts].sum(axis=1)
    N["BumpShareTwentyThree"] = pct(c.loc[2023, "sum_bump"] / tot.loc[2023], 0)
    N["HtlcShareTwentyThree"] = pct(c.loc[2023, "sum_htlc"] / tot.loc[2023], 0)
    N["OverOnePctCapTwentyOne"] = pct(c.loc[2021, "share_over_1pct"], 0)
    N["OverOnePctCapTwentySix"] = pct(c.loc[2026, "share_over_1pct"], 0)
    N["AnchorValueLostBtc"] = dec(co["all_years"]["anchor_value_lost_btc"], 2)
    pen = pd.DataFrame(co["penalties"]).set_index("y")
    N["PenaltyTxsTwentySix"] = num(pen.loc[2026, "txs"])
    pm = pd.DataFrame(co["penalties_monthly"]).groupby("m")["txs"].sum()
    pm = pm.reindex(pd.period_range(pm.index.min(), "2026-08", freq="M").astype(str), fill_value=0)
    N["PenaltyMonthlyMed"] = num(pm.median())                  # months before the wave (to August 2026)
    N["PenaltyMonthlyMaxBefore"] = num(pm.max())
    t = pd.Timestamp(pm.idxmax() + "-01")
    N["PenaltyMonthlyMaxBeforeMonth"] = f"{t:%B}~{t.year}"
    wd = pd.DataFrame(co["penalty_wave_daily"])
    N["PenaltyWaveFirst"] = long_date(wd["d"].min())
    N["PenaltyWaveLast"] = long_date(wd["d"].max())
    N["PenaltyWaveMaxBlocks"] = num(wd["max_blocks_to_penalty"].max())

    # --- anchors ------------------------------------------------------------------------------------------------
    fa = pd.DataFrame(an["fate"]).groupby("y").sum(numeric_only=True)
    tot_a = fa[["owner_bump", "owner_cleanup", "third_fast", "third_late", "unspent"]].sum(axis=1)
    for yr, w in WORDS.items():
        if yr in fa.index:
            N[f"OwnerPct{w}"] = pct((fa.loc[yr, "owner_bump"] + fa.loc[yr, "owner_cleanup"]) / tot_a.loc[yr], 0)
    pub = pd.DataFrame(an["fate"])
    pub = pub[pub["listed"]].set_index("y")
    for yr, w in WORDS.items():
        if yr in pub.index:
            N[f"ThirdMedDelay{w}"] = num(pub.loc[yr, "third_med_delay"])
    N["ThirdAtSixteenTwentyFive"] = pct(pub.loc[2025, "third_at_16"], 0)
    N["ThirdMedDelayDaysTwentyThree"] = dec(pub.loc[2023, "third_med_delay"] / 144, 0)   # 144 blocks a day
    bu = pd.DataFrame(an["bumps"]).groupby("y")[["bumped", "raised"]].sum()
    for yr in (2025, 2026):
        N[f"NotRaisedPct{WORDS[yr]}"] = pct(1 - bu.loc[yr, "raised"] / bu.loc[yr, "bumped"], 0)
    e = pd.DataFrame(an["sweep_economics"]).set_index("y")
    for yr, w in WORDS.items():
        if yr in e.index:
            N[f"MinersShare{w}"] = pct(e.loc[yr, "miners_share"], 0)
    late = e.loc[[yr for yr in e.index if yr >= 2024], "miners_share"]
    N["MinersShareLateMin"] = pct(late.min(), 0)                # 2024 to the cutoff
    N["MinersShareLateMax"] = pct(late.max(), 0)
    tb = pd.DataFrame(an["taproot_anchors_by_year"]).set_index("y")
    N["TaprootAnchorsTwentySix"] = num(tb.loc[2026, "anchors"])
    N["TaprootAnchorsSpentTwentySix"] = num(tb.loc[2026, "spent"])
    if sw:
        N["SweepOperatorsOverOne"] = num(sw["operators_over_1pct"])
        sy = pd.DataFrame(sw["by_year"]).set_index("y")
        for yr in (2024, 2025, 2026):
            N[f"SweepLargestPct{WORDS[yr]}"] = pct(sy.loc[yr, "largest"], 0)

    # --- fee regimes --------------------------------------------------------------------------------------------
    if fe:
        names = {"baseline": "SameWeek", "fee_lag1": "LagOne", "fee_lag2": "LagTwo", "fee_lag3": "LagThree",
                 "fee_lag4": "LagFour", "fee_lead1": "LeadOne", "fee_p10": "PTen", "with_mass_events": "Mass",
                 "monthly": "Monthly"}
        for k, w in names.items():
            r = fe["robustness"][k]
            N[f"Rb{w}B"] = d2(r["b"])
            N[f"Rb{w}Lo"] = d2(r["lo"])
            N[f"Rb{w}Hi"] = d2(r["hi"])
            N[f"Rb{w}P"] = pv(r["p"])
            N[f"Rb{w}N"] = num(r["weeks"])
        N["ElastDoubleLagThree"] = sgn(100 * (2 ** fe["robustness"]["fee_lag3"]["b"] - 1))
        for k, w in (("organic_rate", "All"), ("legacy", "Legacy"), ("anchor", "Anchor"), ("unlisted", "Unlisted")):
            N[f"Elast{w}P"] = pv(fe["elasticity"][k]["p"])
            N[f"Elast{w}N"] = num(fe["elasticity"][k]["weeks"])
        gs = pd.DataFrame(fe["fee_guessing"])
        lg = gs[gs["fmt"] == "legacy"].set_index("y")
        ag = gs[gs["fmt"] == "anchor"].set_index("y")
        early = lg.loc[[yr for yr in lg.index if yr <= 2024], "med_ratio"]
        N["LegacyMedRatioEarlyMin"] = dec(early.min(), 1)
        N["LegacyMedRatioEarlyMax"] = dec(early.max(), 1)
        N["LegacyMedRatioTwentyFive"] = dec(lg.loc[2025, "med_ratio"], 1)
        N["LegacyOverTwoPctTwentyFive"] = pct(lg.loc[2025, "share_over_2x"], 0)
        N["AnchorMedRatioMin"] = dec(ag["med_ratio"].min(), 2)
        N["AnchorMedRatioMax"] = dec(ag["med_ratio"].max(), 2)
        st = pd.DataFrame(fe["sweeper_timing"]).set_index("y")
        N["SweepCheaperPctTwentyFour"] = pct(st.loc[2024, "share_sweep_block_cheaper"], 0)
        bill = pd.DataFrame(fe["closing_bill"]).set_index("y")
        bill = bill[bill.index >= PUBLIC_YEAR]                  # 2018's cooperative closes are not seen
        N["BillFeePctMin"] = pct(bill["fee_share"].min(), 2)
        N["BillFeePctMax"] = pct(bill["fee_share"].max(), 2)
        N["SpikeRule"] = dec(fe["spike_rule"]["x"], 0)
        N["SpikeMinFee"] = dec(fe["spike_rule"]["min_sat_vb"], 0)

    # --- zero-fee -----------------------------------------------------------------------------------------------
    z, cf = zf["census"], zf["counterfactual"]["totals"]
    N["ZfMedCap"] = num(z["med_capacity_sat"])
    N["ZfBeforeMerge"] = num(sum(r["k"] for r in z["by_month"] if r["m"] < "2026-05"))
    N["ZfChildSameBlockPct"] = pct(z["child_same_block"], 0)
    N["CfCloses"] = num(cf["closes"])
    N["CfNotBumpedPct"] = pct(1 - cf["share_bumped"], 0)

    # --- HTLC deadlines (08) ------------------------------------------------------------------------------------
    if ht:
        d = ht["deadlines"]
        N["HtlcMarginTight"] = num(ht["margins_blocks"][0])
        N["HtlcMarginMid"] = num(ht["margins_blocks"][1])
        g = ht["advertised_cltv_expiry_delta"]
        if g["p10"] != ht["margins_blocks"][0] or g["median"] != ht["margins_blocks"][1]:
            raise SystemExit("the HTLC margins are no longer the p10 / median of advertised cltv_expiry_delta")
        N["CltvDeltaFrom"] = f"{pd.Timestamp(g['first_snapshot']):%B}~{pd.Timestamp(g['first_snapshot']).year}"
        k = d["counts"]
        N["HtlcTimeoutTxs"] = num(k["htlc_timeout_2nd_stage"])
        N["HtlcSuccessTxs"] = num(k["htlc_success_2nd_stage"])
        N["HtlcRemoteTimeoutClaims"] = num(k["htlc_remote_timeout_claim"])
        N["HtlcRemotePreimageClaims"] = num(k["htlc_remote_preimage_claim"])
        ty = pd.DataFrame(d["timeout_by_year"]).T
        ty.index = ty.index.astype(int)
        N["HtlcTimeoutMedMin"] = num(ty["median_blocks"].min())
        N["HtlcTimeoutMedMax"] = num(ty["median_blocks"].max())
        for yr, w in WORDS.items():
            if yr in ty.index:
                N[f"HtlcLate{w}"] = pct(ty.loc[yr, "share_gt40"], 1)
                N[f"HtlcVeryLate{w}"] = pct(ty.loc[yr, "share_gt80"], 1)
        early = ty.loc[2019:2022, "share_gt40"]                      # 2019-2022, before the 2023 fees
        N["HtlcLateEarlyMin"] = pct(early.min(), 1)
        N["HtlcLateEarlyMax"] = pct(early.max(), 1)
        N["HtlcCommitBeforeExpiryPct"] = pct(d["timeout_commitment_confirmed_before_expiry_share"], 0)
        ff = d["timeout_ready_by_fee_format"]
        for b, w in (("<5", "Low"), ("5-20", "Mid"), ("20-50", "High"), ("50+", "Spike")):
            N[f"HtlcLate{w}Legacy"] = pct(ff[f"{b}|legacy"]["share_gt40"], 1)
            N[f"HtlcLate{w}Anchor"] = pct(ff[f"{b}|anchor"]["share_gt40"], 1)
        yf = d["timeout_ready_by_year_format"]
        N["HtlcLateTwentyFourLegacy"] = pct(yf["2024|legacy"]["share_gt40"], 1)
        N["HtlcLateTwentyFourAnchor"] = pct(yf["2024|anchor"]["share_gt40"], 1)
        sy = pd.DataFrame(d["success_by_year"]).T
        sy.index = sy.index.astype(int)
        N["HtlcSuccessMarginMin"] = num(sy["median_margin_blocks"].min())
        N["HtlcSuccessMarginMax"] = num(sy["median_margin_blocks"].max())
        late_years = [2019, 2023, 2024]                           # the years with the most late successes
        for yr in late_years:
            N[f"HtlcSuccessLate{WORDS[yr]}"] = pct(sy.loc[yr, "share_at_or_after_expiry"], 1)
        N["HtlcSuccessLateMinOther"] = pct(sy.drop(late_years)["share_at_or_after_expiry"].min(), 1)
        N["HtlcSuccessLateMaxOther"] = pct(sy.drop(late_years)["share_at_or_after_expiry"].max(), 1)
        ca = ht["cascades"]
        N["HtlcCloses"] = num(ca["htlc_closes"])
        N["CascadeCloses"] = num(ca["closes_involved"])
        N["CascadePct"] = pct(ca["closes_involved"] / ca["htlc_closes"], 0)
        cy = pd.DataFrame(ca["by_year"]).set_index("y")
        N["CascadePctTwentyThree"] = pct(cy.loc[2023, "shared"] / cy.loc[2023, "htlc_closes"], 0)
        N["CascadePayments"] = num(ca["hashes_in_2plus_closes"])
        N["CascadeMaxCloses"] = num(ca["max_closes_per_hash"])
        N["CascadeSpanMed"] = num(ca["median_span_blocks"])
        N["CascadeAllTimeoutPct"] = pct(ca["hashes_all_legs_timed_out_share"], 0)
        # Split resolutions: 2021-2026, the window checked pair by pair by hand, after 08's pairing
        # filters and without the staged 2026 test pairs (reported apart).
        sr = ht["split_resolutions"]["from_2021"]["after_filters"]
        fl = sr["by_loser"]["forwarder_lost"]
        st = ht["split_resolutions"]["staged"]
        N["SplitResStagedPairs"] = num(st["pairs"])
        N["SplitResStagedBtc"] = dec(st["btc"], 4)
        N["SplitResStagedToSelfDelay"] = num(st["max_to_self_delay"])
        N["SplitResFromYear"] = str(min(int(y) for y in sr["by_year"]))
        N["SplitResPayments"] = num(sr["payments"])
        N["SplitResBtc"] = dec(sr["btc"], 2)
        N["SplitResForwarderPayments"] = num(fl["payments"])
        N["SplitResForwarderBtc"] = dec(fl["btc"], 2)
        N["SplitResPreimageFirstPct"] = pct(fl["preimage_on_chain_first"], 0)
        N["SplitResGapMed"] = num(fl["median_gap_blocks"])
        # Replacement-cycling order: a successful attack confirms the downstream preimage
        # claim only after the upstream timeout. The text says no organic forwarder loss has that order and every
        # staged pair does; stop if the data no longer say so.
        if fl["preimage_on_chain_first"] < 1:
            raise SystemExit("a forwarder loss from 2021 has the preimage after the upstream timeout")
        if st["preimage_after_timeout_blocks"]["min"] < 0:
            raise SystemExit("a staged pair has the preimage before the upstream timeout")
        N["SplitResStagedPreimageAfterMed"] = num(st["preimage_after_timeout_blocks"]["median"])

    # --- lifecycle cost (09, P10) -------------------------------------------------------------------------------
    if lc:
        ly = pd.DataFrame(lc["lifecycle_by_close_year"])
        a = ly[ly["y"] == "all"].iloc[0]
        ly = ly[ly["y"] != "all"].set_index("y")
        ly = ly[ly.index.astype(int) >= PUBLIC_YEAR]            # public series from 2019
        N["LcN"] = num(lc["n_lifecycles"])
        N["LcMed"] = num(a["med_total"])
        N["LcOpenMed"] = num(a["med_open"])
        N["LcCloseMed"] = num(a["med_close"])
        N["LcCoopMed"] = num(a["med_total_coop"])
        N["LcForceMed"] = num(a["med_total_force"])
        N["LcForceCoopRatio"] = dec(a["med_total_force"] / a["med_total_coop"], 1)
        r = ly["med_total_force"] / ly["med_total_coop"]
        N["LcForceCoopRatioMin"] = dec(r.min(), 1)
        N["LcForceCoopRatioMax"] = dec(r.max(), 1)
        N["LcCapPct"] = pct(a["med_share_cap"], 2)
        N["LcCapCoopPct"] = pct(a["med_share_cap_coop"], 2)
        N["LcCapForcePct"] = pct(a["med_share_cap_force"], 2)
        N["LcOverOnePct"] = pct(a["share_over_1pct_cap"], 0)
        N["LcOpensPct"] = pct(a["open_share_of_fees"], 0)
        N["LcOpenBtc"] = dec(a["open_btc"], 1)
        N["LcTotalBtc"] = dec(a["total_btc"], 1)
        N["LcLifeDaysMed"] = num(a["med_life_days"])
        lr = pd.DataFrame(lc["lifecycle_by_close_regime"]).set_index("close_regime")
        N["LcMedHighRegime"] = num(lr.loc["2023-24", "med_total"])
        N["LcMedLowRegime"] = num(lr.loc["2025-26", "med_total"])
        N["LcMedEarlyRegime"] = num(lr.loc["2019-20", "med_total"])
        N["LcOpensPctLowRegime"] = pct(lr.loc["2025-26", "open_share_of_fees"], 0)
        s = pd.DataFrame(lc["sensitivity"]["lifecycle_by_close_year"])
        s = s[s["y"] == "all"].iloc[0]
        N["LcRobustValuePct"] = pct(1 - s["med_b"] / s["med_a"], 0)
        N["LcRobustSizePct"] = pct(1 - s["med_c"] / s["med_a"], 0)
        N["LcRobustBatchPct"] = pct(1 - s["med_a2"] / s["med_a"], 0)
        rob = sorted((1 - s["med_b"] / s["med_a"], 1 - s["med_c"] / s["med_a"]))   # value and size rules, low-high
        N["LcRobustMinPct"] = pct(rob[0], 0)
        N["LcRobustMaxPct"] = pct(rob[1], 0)
        N["LcOpenAtCutoff"] = num(lc["snapshot_channels_without_close"]["open_at_cutoff"])
        N["ValLakeCompared"] = num(lc["checks"]["snapshot_closes_compared"])
        N["ValLakeAgree"] = num(lc["checks"]["snapshot_close_txid_equals_lake_spender"])
        N["ValLakeShapeAgree"] = num(lc["checks"]["snapshot_close_type_equals_lake_shape"])
        if not (lc["checks"]["snapshot_closes_compared"] == lc["checks"]["snapshot_close_txid_equals_lake_spender"]
                == lc["checks"]["snapshot_close_type_equals_lake_shape"]):
            raise SystemExit("the chain-index validation no longer agrees on every close: reword Data and the abstract")
        N["LcFundingTxs"] = num(lc["checks"]["funding_txs"])

    # --- simple-taproot anchors (04c: the reveal test and the sweep economics, to the cutoff) -------------
    if tp:
        if tp.get("measured_to_height") != man["cutoff_height"]:
            raise SystemExit("taproot_anchor_paths.json is not measured to the cutoff; rerun src/04c_taproot_anchors.py")
        N["TaprootKeyPathSpends"] = num(tp["anchor_spend_paths"]["key_path"])
        N["TaprootScriptPathSpends"] = num(tp["script_path_spends"])
        N["TaprootScriptPathChainwide"] = num(tp["chainwide_op16_csv_script_spends"]["inputs"])
        N["TaprootAnchorsUnspent"] = num(tp["anchor_spend_paths"]["unspent"])
        ex = tp["exposed_anchors"]
        N["TaprootKeyExposedUnspent"] = num(ex["still_unspent_at_cutoff"])
        N["TaprootKeyExposedDaysMed"] = num(ex["exposed_blocks_median"] / 144)
        e = tp["economics"]["paper"]
        N["TaprootKeyPublicBtc"] = dec(e["key_public_btc"], 3)
        N["TaprootUnspentBtc"] = dec(e["unspent_btc"], 3)
        N["TaprootKeyPublicNetOneBtc"] = dec(e["key_public_net_btc_at_1_sat_vb"], 3)
        N["TaprootScriptPathVb"] = dec(e["script_path_vb"], 1)
        N["TaprootScriptPathBreakEven"] = dec(e["script_path_breakeven_sat_vb"], 1)
        N["TaprootUnspentOutputSharePct"] = pct(e["unspent_output_share"], 0)
        N["TaprootUnspentValueSharePct"] = pct(e["unspent_value_share"], 2)
        N["TaprootCoinBytes"] = num(e["coin_entry_bytes"])
        N["TaprootUtxoSetSharePct"] = pct(e["utxo_set_share"], 2)
        N["TaprootOwnerSpendsEarly"] = num(e["owner_spends_early"])
        N["TaprootOwnerSpendsLate"] = num(e["owner_spends_late"])
        N["TaprootOwnerSpendsLateSameBlock"] = num(e["owner_spends_late_same_block"])
        N["TaprootCosweepToRemotePctEarly"] = pct(e["cosweep_to_remote_early"]["share"], 0)
        N["TaprootCosweepToRemotePctLate"] = pct(e["cosweep_to_remote_late"]["share"], 1)
        N["TaprootOwnerSweepFrEarly"] = dec(e["owner_sweep_feerate_median_early"], 1)
        N["TaprootOwnerSweepFrLate"] = dec(e["owner_sweep_feerate_median_late"], 1)

    # --- breach wave (04d) ---------------------------------------------------------------------
    bw = load("breach_wave")
    if bw:
        sp, la = bw["splices"], bw["look_alikes"]
        N["BreachWavePenalised"] = num(bw["penalised_commitments"])
        N["BreachWaveChannels"] = num(bw["channels"])
        N["BreachWaveBtc"] = dec(bw["btc_without_outlier"], 2)
        N["BreachOpenToSpliceMed"] = num(sp["blocks_open_to_splice"]["p50"])
        N["BreachSpliceToCloseMed"] = num(sp["blocks_splice_to_revoked_close"]["p50"])
        N["BreachFunderIsPenalizer"] = num(sp["funder_is_penalizer_any"])
        N["BreachLookAlikesWeek"] = num(la["splices_in_wave_week"])
        N["BreachRevokedSharePct"] = pct(bw["share_of_pattern_closes_revoked_in_wave_week"], 0)

    # --- 2026 pre-anchor public force closes (02b) ---------------------------------------------
    pa = load("pre_anchor_2026")
    if pa:
        N["LegacyTopFivePct"] = pct(pa["top_nodes_share_of_pre_anchor"], 0)
        N["LegacyTopFiveLdkNoAnchor"] = num(pa["top_nodes_ldk_without_anchor_bits"])
        N["LegacyClusterAgeDays"] = num(pa["top_nodes_pre_anchor_median_age_days"])
        N["LegacyPctExClusterTwentySix"] = pct(pa["without_top_nodes"]["pre_anchor_share"], 0)
        N["LegacyOldAgeYears"] = dec(pa["without_top_nodes"]["pre_anchor_median_age_years"], 1)
        N["LegacyOneLacksAnchorPct"] = pct(pa["pre_anchor_one_endpoint_without_anchor_bits_share"], 0)
        if pct(pa["pre_anchor_share"], 0) != N["PubLegacyPctTwentySix"]:
            raise SystemExit("02b's 2026 pre-anchor share disagrees with the census")

    # --- fee-spike placebos (05b_fee_placebo.py) -------------------------------------------------------
    # data/fees_placebo.json: stage 05b on 05's weekly frame, 2019-2026. The panel frame is 05's weekly table.
    if fp:
        ds, pa = fp["design"], fp["panel"]
        N["PlSeed"] = num(ds["seed"])
        N["PlWeekDraws"] = num(ds["random_week_draws"])
        N["PlBlockDraws"] = num(ds["block_shuffle_draws"])
        N["PlBlockWeeks"] = num(ds["block_weeks"])
        N["PlMinShift"] = num(ds["min_shift_weeks"])
        cs, bs = pa["circular_shift"], pa["block_shuffle"]["block_8w"]
        N["PlShifts"] = num(cs["draws"])
        tim = {"fee_lead1": "LeadOne", "baseline": "SameWeek", "fee_lag1": "LagOne", "fee_lag2": "LagTwo",
               "fee_lag3": "LagThree", "fee_lag4": "LagFour"}
        rows = [(cs["lags"][k], bs["lags"][k], w) for k, w in tim.items()] + [(cs["max_lag1_4"], bs["max_lag1_4"], "MaxLag")]
        for c, b_, w in rows:
            N[f"Pl{w}B"] = d2(c["real"])
            N[f"Pl{w}CircLo"] = d2(c["q025"])
            N[f"Pl{w}CircHi"] = d2(c["q975"])
            N[f"Pl{w}CircPct"] = pc(c["percentile"])
            N[f"Pl{w}CircP"] = pv(c["p_upper"])
            N[f"Pl{w}BlockPct"] = pc(b_["percentile"])
            N[f"Pl{w}BlockP"] = pv(b_["p_upper"])
        hac_b = [bs["lags"][k]["share_hac_p_below_0.05"] for k in tim]
        hac_c = [cs["lags"][k]["share_hac_p_below_0.05"] for k in tim]
        N["PlNwRejectBlockMin"] = pct(min(hac_b), 0)
        N["PlNwRejectBlockMax"] = pct(max(hac_b), 0)
        N["PlNwRejectCircMin"] = pct(min(hac_c), 0)
        N["PlNwRejectCircMax"] = pct(max(hac_c), 0)
        er = pa["random_weeks"]["year_matched"]["event_ratios"]
        for s, sw_ in (("organic", "All"), ("legacy", "Legacy"), ("anchor", "Anchor")):
            for win, ww in (("w0_1", "Now"), ("w2_3", "Later")):
                r = er[f"{s}_{win}"]
                N[f"PlEv{sw_}{ww}"] = dec(r["real"], 2)
                N[f"PlEv{sw_}{ww}Med"] = dec(r["q50"], 2)
                N[f"PlEv{sw_}{ww}Lo"] = dec(r["q05"], 2)
                N[f"PlEv{sw_}{ww}Hi"] = dec(r["q95"], 2)
                N[f"PlEv{sw_}{ww}Pct"] = pc(r["percentile"])
        ps = [s for s in pa["real"]["per_spike"] if s["week"] != "2023-09-04"]
        N["PlEvAllLaterNoSep"] = dec(sum(s["organic_w2_3"] for s in ps) / sum(s["organic_before"] for s in ps), 2)
        v = pa["verdict"]
        N["PlWaveP"] = pv(v["wave_p_upper_random_weeks"])
        N["PlDetectableWavePct"] = pct(v["smallest_detectable_wave"], 0)
        N["PlBelowTypicalPct"] = pct(-v["real_vs_typical_week"], 0)
        N["PlFwPBlock"] = pv(v["family_wise_p_max_lag1_4"]["block_shuffle"])
        N["PlFwPCirc"] = pv(v["family_wise_p_max_lag1_4"]["circular_shift"])


if __name__ == "__main__":
    sys.exit(main())
