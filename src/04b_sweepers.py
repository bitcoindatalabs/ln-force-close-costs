"""
04b_sweepers.py (RQ3, continued): who collects third-party anchor sweeps, and who received the penalties of the
September–October 2026 breach wave. Aggregates only: no address or node is named in any released output.

Fetches (once, resumable; read-only getrawtransaction from our node, single-threaded) every third-party sweep
transaction (data/work/third_party_sweeps.parquet, from 04_anchors.py) and every penalty transaction, keeping their
inputs' outpoints and their outputs (value, script type, address) in data/work/fetched_txs.jsonl (not committed).

Operators: sweep transactions are linked when one spends an output of another (a sweeper reusing its own coins) or
when they pay the same address; connected components are operators. Transactions whose outputs are all OP_RETURN give
everything to fees and are counted apart.

Writes data/sweepers.json.
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ANCHOR_VALUE_SAT, WORK_DIR  # noqa: E402
from fcc.io import connect, df, write_json  # noqa: E402

CACHE = WORK_DIR / "fetched_txs.jsonl"


def fetch(txids: list) -> dict:
    have = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            t = json.loads(line)
            have[t["txid"]] = t
    todo = [t for t in txids if t not in have]
    print(f"fetch: {len(have):,} cached, {len(todo):,} to fetch", flush=True)
    if todo and not os.environ.get("FCC_ALLOW_RPC"):
        sys.exit(f"{len(todo):,} transactions are not in {CACHE}; the 2018-2026 rerun takes no RPC "
                 "(set FCC_ALLOW_RPC=1 to fetch them from the node)")
    if todo:
        from fcc.rpc import Rpc
        rpc = Rpc()
        with CACHE.open("a", encoding="utf-8") as f:
            for k, txid in enumerate(todo):
                tx = rpc.call("getrawtransaction", [txid, 1])
                t = {"txid": txid, "vin": [f"{i['txid']}:{i['vout']}" for i in tx["vin"]],
                     "vout": [{"v": int(round(o["value"] * 1e8)), "type": o["scriptPubKey"].get("type"),
                               "addr": o["scriptPubKey"].get("address")} for o in tx["vout"]]}
                f.write(json.dumps(t) + "\n")
                have[txid] = t
                if (k + 1) % 2000 == 0:
                    print(f"  {k + 1:,}/{len(todo):,}", flush=True)
    return have


def operators(sw: pd.DataFrame, txs: dict) -> pd.DataFrame:
    ids = list(sw["spend_txid"])
    parent = {t: t for t in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    first_by_addr = {}
    for t in ids:
        tx = txs[t]
        for prev in tx["vin"]:
            p = prev.split(":")[0]
            if p in parent:
                union(t, p)
        for o in tx["vout"]:
            if o["addr"]:
                if o["addr"] in first_by_addr:
                    union(t, first_by_addr[o["addr"]])
                else:
                    first_by_addr[o["addr"]] = t
    sw = sw.copy()
    sw["opreturn_only"] = [all(o["type"] == "nulldata" for o in txs[t]["vout"]) for t in ids]
    sw["operator"] = [("burn" if b else find(t)) for t, b in zip(ids, sw["opreturn_only"])]
    return sw


def main() -> int:
    con = connect()
    sw = pd.read_parquet(WORK_DIR / "third_party_sweeps.parquet")
    # only the penalties of the 2026-09-28 .. 2026-10-05 wave are read below, so only they are fetched
    pen = df(con, """SELECT DISTINCT spend_txid, spend_time_utc, listed FROM o
                     WHERE spend_type IN ('to_local_revocation_breach', 'htlc_revocation_breach')
                       AND spend_time_utc >= TIMESTAMP '2026-09-28' AND spend_time_utc < TIMESTAMP '2026-10-06'""")
    txs = fetch(sorted(set(sw["spend_txid"]) | set(pen["spend_txid"])))

    sw = operators(sw, txs)
    sw["y"] = sw["t"].dt.year
    tot = sw.groupby("operator")["anchors"].sum().sort_values(ascending=False)
    ops = tot.drop("burn", errors="ignore")
    top = ops.index[0]
    sw["who"] = sw["operator"].map(lambda o: "largest" if o == top else ("burn" if o == "burn" else "others"))
    by_year = sw.pivot_table(index="y", columns="who", values="anchors", aggfunc="sum", fill_value=0)
    by_year = by_year.div(by_year.sum(axis=1), axis=0)
    by_month = sw.assign(m=sw["t"].dt.strftime("%Y-%m")).pivot_table(index="m", columns="who", values="anchors",
                                                                      aggfunc="sum", fill_value=0)
    largest = sw[sw["operator"] == top]
    gain = (ANCHOR_VALUE_SAT * largest["anchors"] - largest["fee"]).sum()
    share = ops / ops.sum()
    out = {
        "operators": int(len(ops)), "anchors": int(tot.sum()), "txs": int(len(sw)),
        "largest_share": float(tot[top] / tot.sum()),
        "largest_first": str(largest["t"].min().date()), "largest_last": str(largest["t"].max().date()),
        "largest_txs": int(len(largest)), "largest_net_btc": float(gain / 1e8),
        "operators_over_1pct": int((share > 0.01).sum()),
        "burn_share": float(tot.get("burn", 0) / tot.sum()), "burn_txs": int(sw["opreturn_only"].sum()),
        "by_year": by_year.reset_index().to_dict(orient="records"),
        "by_month": by_month.reset_index().to_dict(orient="records"),
    }

    # penalties of the 2026-09-28 .. 2026-10-05 wave: how many distinct recipients
    wave = pen[(pen["spend_time_utc"] >= "2026-09-28") & (pen["spend_time_utc"] < "2026-10-06") & ~pen["listed"]]
    addrs = pd.Series([o["addr"] for t in wave["spend_txid"] for o in txs[t]["vout"] if o["addr"]])
    vals = pd.Series([o["v"] for t in wave["spend_txid"] for o in txs[t]["vout"]])
    out["penalty_wave"] = {"txs": int(len(wave)), "recipient_addresses": int(addrs.nunique()),
                           "top_address_share_of_txs": float(addrs.value_counts().iloc[0] / len(wave)) if len(addrs) else 0,
                           "outputs_btc": float(vals.sum() / 1e8),
                           "output_types": pd.Series([o["type"] for t in wave["spend_txid"] for o in txs[t]["vout"]])
                                             .value_counts().to_dict()}
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_month",)}, indent=1, default=str))
    write_json("sweepers.json", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
