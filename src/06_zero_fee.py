"""
06_zero_fee.py (RQ5): the first zero-fee commitment closes on mainnet, and what the new format would have changed for the
anchor force closes we observe.

  * census of version-3 commitments with a pay-to-anchor (P2A) output: dates, public or not, the child that spent the
    P2A output, the package fee rate, and how many distinct public nodes were involved (no node is named);
  * block-space counterfactual for every anchor force close (public or not) at least SPEND_CENSOR_BLOCKS before the
    cutoff, from protocol sizes:
        anchor design: two 43-vB P2WSH anchor outputs, each later spent by its owner (69.75 vB) or a third party
                       (51.75 vB);
        zero-fee:      one 13-vB P2A output, spent in the same package (41.25 vB); closes that were not already bumped
                       by a child also need one (a wallet input and a change output, CHILD_EXTRA_VB);
    plus the anchor value that no longer exists (330 sat per anchor; P2A outputs carry 0 sat).

Writes data/zero_fee.json, paper/tables/table_zero_fee.tex.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANCHOR_SWEEP_INPUT_VB, ANCHOR_VALUE_SAT, SPEND_CENSOR_BLOCKS  # noqa: E402
from fcc.io import connect, df, manifest, write_json  # noqa: E402
from fcc.style import f1, n, tex_table  # noqa: E402

P2WSH_ANCHOR_OUT_VB = 43.0          # 8 value + 1 length + 34 script
OWNER_ANCHOR_IN_VB = 41 + 115 / 4   # outpoint, sequence, empty scriptSig; witness: count, 72-byte sig, 41-byte script
P2A_OUT_VB = 13.0                   # 8 value + 1 length + 4 script (OP_1 <0x4e73>)
P2A_IN_VB = 41 + 1 / 4              # empty witness
CHILD_EXTRA_VB = 10.5 + 57.5 + 43   # a child's own overhead, one P2TR wallet input, one P2TR change output


def census(con) -> dict:
    z = df(con, """
        SELECT fc.listed, fc.close_time_utc, fc.close_height, fc.fee_sat, fc.vsize, fc.capacity,
               o.spend_type, o.blocks_after_close, o.spend_fee_sat, o.spend_vsize, o.spend_n_inputs
        FROM fc JOIN o USING (close_txid) WHERE fc.fmt = 'zero_fee' AND o.role = 'p2a_anchor'
        ORDER BY close_height""")
    z["pkg_fr"] = (z["fee_sat"] + z["spend_fee_sat"]) / (z["vsize"] + z["spend_vsize"])
    nodes = df(con, """SELECT n, count(*) AS k FROM (
                           SELECT node1_pub AS n FROM fc WHERE fmt = 'zero_fee' AND listed
                           UNION ALL SELECT node2_pub FROM fc WHERE fmt = 'zero_fee' AND listed) GROUP BY n
                       ORDER BY k DESC""")
    n_public = int(z["listed"].sum())
    return {
        "closes": int(len(z)), "public": n_public, "unlisted": int((~z["listed"]).sum()),
        "first": str(z["close_time_utc"].min().date()), "first_public": str(z.loc[z["listed"], "close_time_utc"].min().date()),
        "last": str(z["close_time_utc"].max().date()),
        "by_month": z.assign(m=z["close_time_utc"].dt.strftime("%Y-%m")).groupby(["m", "listed"]).size()
                     .rename("k").reset_index().to_dict(orient="records"),
        "commitment_fee_zero": bool((z["fee_sat"] == 0).all()),
        "child_same_block": float((z["blocks_after_close"] == 0).mean()),
        "med_child_fee": float(z["spend_fee_sat"].median()), "med_child_vb": float(z["spend_vsize"].median()),
        "med_pkg_fr": float(z["pkg_fr"].median()),
        "public_nodes": int(len(nodes)), "top_node_share_of_public": float(nodes["k"].max() / n_public) if n_public else 0,
        "med_capacity_sat": float(z["capacity"].median()),
    }


def counterfactual(con, H: int) -> dict:
    r = df(con, f"""
        WITH a AS (
            SELECT close_txid,
                   count(*) AS anchors,
                   count(*) FILTER (spend_type = 'anchor_sweep_owner') AS owner,
                   count(*) FILTER (spend_type = 'anchor_sweep_anyone') AS anyone,
                   bool_or(spend_type = 'anchor_sweep_owner' AND blocks_after_close = 0) AS bumped
            FROM o WHERE role = 'anchor' AND script_type = 'witness_v0_scripthash' GROUP BY close_txid)
        SELECT year(fc.close_time_utc) AS y, count(*) AS closes, sum(a.anchors) AS anchors, sum(a.owner) AS owner,
               sum(a.anyone) AS anyone, sum(a.bumped::INT) AS bumped, sum(fc.vsize) AS commitment_vb
        FROM fc JOIN a USING (close_txid)
        WHERE fc.fmt = 'anchor' AND fc.close_height <= {H - SPEND_CENSOR_BLOCKS}
        GROUP BY y ORDER BY y""")
    r["anchor_vb"] = P2WSH_ANCHOR_OUT_VB * r["anchors"] + OWNER_ANCHOR_IN_VB * r["owner"] + ANCHOR_SWEEP_INPUT_VB * r["anyone"]
    r["p2a_vb"] = (P2A_OUT_VB + P2A_IN_VB) * r["closes"] + CHILD_EXTRA_VB * (r["closes"] - r["bumped"])
    r["anchor_value_btc"] = ANCHOR_VALUE_SAT * r["anchors"] / 1e8
    tot = r.sum(numeric_only=True)
    return {"by_year": r.to_dict(orient="records"),
            "totals": {"closes": int(tot["closes"]), "anchors": int(tot["anchors"]),
                       "anchor_vb": float(tot["anchor_vb"]), "p2a_vb": float(tot["p2a_vb"]),
                       "anchor_vb_per_close": float(tot["anchor_vb"] / tot["closes"]),
                       "p2a_vb_per_close": float(tot["p2a_vb"] / tot["closes"]),
                       "share_bumped": float(tot["bumped"] / tot["closes"]),
                       "anchor_value_btc": float(tot["anchor_value_btc"])},
            "sizes_vb": {"p2wsh_anchor_out": P2WSH_ANCHOR_OUT_VB, "owner_anchor_in": OWNER_ANCHOR_IN_VB,
                         "anyone_anchor_in": ANCHOR_SWEEP_INPUT_VB, "p2a_out": P2A_OUT_VB, "p2a_in": P2A_IN_VB,
                         "child_extra": CHILD_EXTRA_VB}}


def main() -> int:
    con = connect()
    H = manifest()["cutoff_height"]
    c = census(con)
    cf = counterfactual(con, H)
    print(c)
    print(cf["totals"])
    rows = [[str(int(r["y"])) + (r"\textsuperscript{a}" if r["y"] == 2026 else ""), n(r["closes"]),
             f1(100 * r["bumped"] / r["closes"]), f1(r["anchor_vb"] / r["closes"]), f1(r["p2a_vb"] / r["closes"]),
             f"{r['anchor_value_btc']:.3f}"] for r in cf["by_year"]]
    tex_table("table_zero_fee", ["Year", "Anchor closes", "Bumped \\%", "Anchors vB", "P2A vB", "Anchor BTC"], rows,
              "lrrrrr", "Block space of the anchor mechanism per force close, observed (two P2WSH anchors and their "
              "spends) and under zero-fee commitments (one pay-to-anchor output spent by a child).", "tab:zero_fee",
              note=r"\textsuperscript{a}Closes to 4 October 2026. Zero-fee closes that were not already bumped need a "
                   r"child of their own (wallet input and change output, " + f1(CHILD_EXTRA_VB) + r" vB). Anchor BTC: "
                   r"value locked in 330-sat anchors, which pay-to-anchor outputs (0 sat) do not create.")
    write_json("zero_fee.json", {"census": c, "counterfactual": cf})
    return 0


if __name__ == "__main__":
    sys.exit(main())
