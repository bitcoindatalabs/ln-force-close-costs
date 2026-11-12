"""
04c_taproot_anchors.py: what happens to the anchors of simple-taproot channels, what an
unswept one is worth, and why their owners stopped spending them.

Two steps.

  --extract STAGE [args]   (reads the chain lake; run outside the cron busy windows, one short command per chunk
                            group, each under the lightning_lake lock)
      closes              (no lake read) the taproot force closes of the work table -> work/taproot_closes.parquet
      commits A B         chunks A..B (height // 1000): each commitment's tx_key, every output (script) and the
                          transaction that spent it -> work/taproot_commit_outputs_A_B.parquet
      inputs L A B        spends at hop L (1: spends of commitment outputs; 2: spends of the same-index output of a
                          hop-1 spender, i.e. second-level HTLC outputs) confirmed in chunks A..B: witness fields, plus
                          the spender's output at the same index and its own spend -> work/taproot_inputs_L{L}_A_B.parquet
      scan16 A B          chain-wide: every input whose tapscript leaf is OP_16 OP_CHECKSEQUENCEVERIFY (0x60b2), the
                          script path of a simple-taproot anchor -> work/taproot_scan16_A_B.parquet
      coinstats           the last gettxoutsetinfo checkpoint at or before the cutoff -> work/taproot_coinstats.parquet
  (default)   (reads only the work tables, the snapshot, data/work/blockstats.parquet and data/sweepers.json)
      1. Spend paths. For every anchor of every taproot force close, every 32-byte push in the leaf scripts
         of every spend of that commitment's outputs (hop 1) and of its second-level outputs (hop 2) is tweaked with
         the anchor leaf (BIP 341) and compared with the anchor's output key: an exact match marks the transaction
         that put the anchor's internal key on-chain (the reveal). Each anchor spend is classified by its witness:
         key path (one 64/65-byte signature, the owner) or the 16-block script path (anyone, with the key).
      2. Sweep prize and footprint. Gross and net value of the unspent anchors, and of those whose key is
         on-chain, against a 50.5 vB script-path input; their share of the force-close outputs unspent at the cutoff
         and of the UTXO set; the chainstate size of each.
      3. Owner spends. Owner key-path spends by half-year; for every anchor whose owner swept the matching
         to_local / to_remote / second-level output while the anchor was unspent, whether the anchor went into that
         sweep, and the sweep's fee rate against the anchor's break-even.

Every height is truncated at the cutoff (data/snapshot_manifest.json): spends, reveals and the chain-wide scan count
only up to block cutoff_height, so "unspent" means unspent at the cutoff. Spec: lightning/bolts
bolt-simple-taproot.md ("Anchor Outputs"); anchor internal key = local_delayedpubkey or remotepubkey, one leaf
OP_16 OP_CSV; to_local / to_remote use a NUMS internal key, so sweeping them reveals the anchor's key in a leaf
script.

Writes data/taproot_anchor_paths.json (aggregates only: no transaction, key or address) and
work/taproot_anchors.parquet (one row per anchor; not released).

Usage (chunk groups; taproot closes start in chunk 805, the scan in chunk 709):
  python src/04c_taproot_anchors.py --extract closes
  python src/04c_taproot_anchors.py --extract commits 805 879     # ... up to 974
  python src/04c_taproot_anchors.py --extract inputs 1 805 929    # ... up to 974; then L2
  python src/04c_taproot_anchors.py --extract scan16 709 849      # ... up to 974
  python src/04c_taproot_anchors.py --extract coinstats
  python src/04c_taproot_anchors.py
"""

import argparse
import json
import sys
import time
from hashlib import sha256
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (ANCHOR_VALUE_SAT, CHAIN_LAKE_DIR, DATA_DIR, SNAPSHOT_DIR, SPEND_CENSOR_BLOCKS,  # noqa: E402
                    WORK_DIR)
from fcc.io import manifest, write_json  # noqa: E402

PARTS = CHAIN_LAKE_DIR / "parts"
LINKED = CHAIN_LAKE_DIR / "linked"
SCAN_FROM_CHUNK = 709                  # taproot activated at block 709,632
CSV_BLOCKS = 16                        # anchor script path: OP_16 OP_CHECKSEQUENCEVERIFY
FIXED_RATES = [0.1, 1.0, 2.0, 5.0]     # 0.1 sat/vB = Bitcoin Core's default minrelaytxfee since 29.1
BATCH_BLOCK_MIN = 50                   # a block with >= 50 taproot force closes counts as a batch-close block
# halves of the year compared for the owners' spends (block time, UTC); the late one ends at the cutoff
EARLY, LATE = "2025H1", "2026H2"

# Input sizes (BIP 141 weight). Non-witness part of any input: outpoint 36 + scriptSig length 1 + nSequence 4 = 41 B.
# Script path: witness item count 1 + (len 1 + leaf script 0x60b2 2) + (len 1 + control block 33) = 38 WU.
# Key path:    witness item count 1 + (len 1 + Schnorr signature 64) = 66 WU (SIGHASH_DEFAULT, no annex).
SCRIPT_PATH_WU = 41 * 4 + (1 + 1 + 2 + 1 + 33)
KEY_PATH_WU = 41 * 4 + (1 + 1 + 64)
SCRIPT_PATH_VB = SCRIPT_PATH_WU / 4
KEY_PATH_VB = KEY_PATH_WU / 4
TX_OVERHEAD_VB = 10.5 + 43             # per sweep transaction: header and counts, one P2TR output


# ---------------------------------------------------------------------------------------------------------------------
# Step 1: extract from the chain lake
# ---------------------------------------------------------------------------------------------------------------------

def lake_file(kind: str, c: int) -> str:
    """The chain-lake file of a chunk (parts/ or linked/; the two can end at different heights)."""
    folder = LINKED if kind.endswith("_link") else PARTS
    hits = sorted(folder.glob(f"{kind}_{c * 1000}_*.parquet"))
    if len(hits) != 1:
        raise SystemExit(f"chunk {c}: expected one {kind} file in {folder}, found {hits}")
    return str(hits[0])


def lake_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET threads=4")
    return con


def cutoff() -> int:
    return int(manifest()["cutoff_height"])


def check_range(a: int, b: int):
    if b > cutoff() // 1000:
        raise SystemExit(f"chunk {b} lies past the cutoff ({cutoff():,}); nothing there counts")


def extract_closes():
    con = lake_con()
    d = con.execute(f"""
        SELECT close_txid, close_height, n_outputs, n_anchor_outputs, has_htlc_like_outputs, close_time_utc
        FROM '{WORK_DIR / 'force_closes.parquet'}' WHERE fmt = 'taproot'""").df()
    d.to_parquet(WORK_DIR / "taproot_closes.parquet", index=False)
    print(f"{len(d):,} taproot force closes, heights {d['close_height'].min():,}..{d['close_height'].max():,}")


def extract_commits(a: int, b: int):
    check_range(a, b)
    closes = pd.read_parquet(WORK_DIR / "taproot_closes.parquet")
    con = lake_con()
    out = []
    for c in range(a, b + 1):
        sel = closes[closes["close_height"] // 1000 == c]
        if sel.empty:
            continue
        want = pd.DataFrame({"height": sel["close_height"].astype("int32"),
                             "txid": [bytes.fromhex(t) for t in sel["close_txid"]]})
        con.register("want", want)
        keys = con.execute(f"""
            SELECT t.tx_key, t.height, lower(hex(t.txid)) AS close_txid
            FROM '{lake_file('txs', c)}' t JOIN want w ON t.height = w.height AND t.txid = w.txid""").df()
        con.unregister("want")
        if len(keys) != len(sel):
            print(f"chunk {c}: {len(sel)} closes, {len(keys)} found in the lake", file=sys.stderr)
        con.register("k", keys)
        rows = con.execute(f"""
            SELECT k.close_txid, k.height AS close_height, o.tx_key AS close_tx_key, o.vout, o.value, o.script_type,
                   o.script, l.spend_tx_key, l.spend_vin
            FROM '{lake_file('outputs', c)}' o JOIN k ON o.tx_key = k.tx_key
            LEFT JOIN '{lake_file('outputs_link', c)}' l ON l.tx_key = o.tx_key AND l.vout = o.vout""").df()
        con.unregister("k")
        out.append(rows)
        print(f"chunk {c}: {len(sel)} closes, {len(rows)} outputs, {rows['spend_tx_key'].notna().sum()} spent")
    if out:
        pd.concat(out).to_parquet(WORK_DIR / f"taproot_commit_outputs_{a}_{b}.parquet", index=False)


def extract_inputs(level: int, a: int, b: int):
    check_range(a, b)
    con = lake_con()
    if level == 1:
        src = con.execute(f"""SELECT close_txid, close_tx_key AS prev_tx_key, vout AS prev_vout,
                                     spend_tx_key AS tx_key, spend_vin AS vin
                              FROM '{WORK_DIR}/taproot_commit_outputs_*.parquet' WHERE spend_tx_key IS NOT NULL""").df()
    else:
        src = con.execute(f"""SELECT close_txid, tx_key AS prev_tx_key, vin AS prev_vout,
                                     same_out_spend_tx_key AS tx_key, same_out_spend_vin AS vin
                              FROM '{WORK_DIR}/taproot_inputs_L{level - 1}_*.parquet'
                              WHERE same_out_spend_tx_key IS NOT NULL AND witness_items = 4""").df()
    src["chunk"] = src["tx_key"].astype("int64") // 32768 // 1000
    out = []
    for c in range(a, b + 1):
        sel = src[src["chunk"] == c].drop(columns="chunk")
        if sel.empty:
            continue
        con.register("w", sel)
        rows = con.execute(f"""
            SELECT w.close_txid, w.prev_tx_key, w.prev_vout, i.tx_key, i.vin, i.height, i.sequence,
                   i.witness_items, i.witness_len, i.witness_script, i.tap_depth, i.tap_leaf_version, i.has_annex,
                   i.schnorr_len, t.n_in, t.n_out, t.vsize, lt.fee, lt.input_value,
                   so.value AS same_out_value, so.script_type AS same_out_type,
                   sl.spend_tx_key AS same_out_spend_tx_key, sl.spend_vin AS same_out_spend_vin
            FROM '{lake_file('inputs', c)}' i JOIN w ON i.tx_key = w.tx_key AND i.vin = w.vin
            JOIN '{lake_file('txs', c)}' t ON t.tx_key = i.tx_key
            LEFT JOIN '{lake_file('txs_link', c)}' lt ON lt.tx_key = i.tx_key
            LEFT JOIN '{lake_file('outputs', c)}' so ON so.tx_key = i.tx_key AND so.vout = i.vin
            LEFT JOIN '{lake_file('outputs_link', c)}' sl ON sl.tx_key = i.tx_key AND sl.vout = i.vin""").df()
        con.unregister("w")
        if len(rows) != len(sel):
            print(f"chunk {c}: wanted {len(sel)} inputs, got {len(rows)}", file=sys.stderr)
        out.append(rows)
        print(f"L{level} chunk {c}: {len(rows)} inputs")
    if out:
        pd.concat(out).to_parquet(WORK_DIR / f"taproot_inputs_L{level}_{a}_{b}.parquet", index=False)


def extract_scan16(a: int, b: int):
    check_range(a, b)
    con = lake_con()
    out = []
    for c in range(a, b + 1):
        rows = con.execute(f"""
            SELECT i.tx_key, i.vin, i.height, i.sequence, i.witness_items, i.tap_depth, l.prev_tx_key, l.prev_value,
                   l.prev_script_type, i.prev_vout
            FROM '{lake_file('inputs', c)}' i
            JOIN '{lake_file('inputs_link', c)}' l ON l.tx_key = i.tx_key AND l.vin = i.vin
            WHERE i.witness_script_len = 2 AND i.witness_script = '\\x60\\xb2'::BLOB""").df()
        out.append(rows)
        print(f"scan16 chunk {c}: {len(rows)}", flush=True)
    pd.concat(out).to_parquet(WORK_DIR / f"taproot_scan16_{a}_{b}.parquet", index=False)


def extract_coinstats():
    con = lake_con()
    d = con.execute(f"""SELECT height, txouts, bogosize FROM '{CHAIN_LAKE_DIR / 'coinstats_every100.parquet'}'
                        WHERE height <= {cutoff()} ORDER BY height DESC LIMIT 1""").df()
    d.to_parquet(WORK_DIR / "taproot_coinstats.parquet", index=False)
    print(d.to_string(index=False))


# ---------------------------------------------------------------------------------------------------------------------
# secp256k1 / BIP 340-341, minimal (exact output-key recomputation)
# ---------------------------------------------------------------------------------------------------------------------
Pp = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
     0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)


def tagged_hash(tag: str, msg: bytes) -> bytes:
    th = sha256(tag.encode()).digest()
    return sha256(th + th + msg).digest()


def padd(p1, p2):
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    if p1[0] == p2[0] and (p1[1] + p2[1]) % Pp == 0:
        return None
    if p1 == p2:
        lam = 3 * p1[0] * p1[0] * pow(2 * p1[1], -1, Pp) % Pp
    else:
        lam = (p2[1] - p1[1]) * pow(p2[0] - p1[0], -1, Pp) % Pp
    x = (lam * lam - p1[0] - p2[0]) % Pp
    return x, (lam * (p1[0] - x) - p1[1]) % Pp


G_POW = [G]
for _ in range(255):
    G_POW.append(padd(G_POW[-1], G_POW[-1]))


def mul_g(k: int):
    r = None
    for i in range(256):
        if (k >> i) & 1:
            r = padd(r, G_POW[i])
    return r


def lift_x(x: int):
    if x >= Pp:
        return None
    c = (pow(x, 3, Pp) + 7) % Pp
    y = pow(c, (Pp + 1) // 4, Pp)
    if y * y % Pp != c:
        return None
    return x, (y if y % 2 == 0 else Pp - y)


def tapleaf(script: bytes, version: int = 0xC0) -> bytes:
    return tagged_hash("TapLeaf", bytes([version, len(script)]) + script)   # len < 253


def taproot_output_key(xonly: bytes, merkle_root: bytes):
    P = lift_x(int.from_bytes(xonly, "big"))
    if P is None:
        return None
    t = int.from_bytes(tagged_hash("TapTweak", xonly + merkle_root), "big")
    if t >= N:
        return None
    return padd(P, mul_g(t))[0].to_bytes(32, "big")


ANCHOR_SCRIPT = bytes([0x60, 0xB2])          # OP_16 OP_CHECKSEQUENCEVERIFY
ANCHOR_ROOT = tapleaf(ANCHOR_SCRIPT)


# ---- Bitcoin Core chainstate serialization (src/coins.h, src/compressor.{h,cpp}, src/serialize.h, src/txdb.cpp) ----
def core_varint_len(n: int) -> int:
    """Length of Bitcoin Core's VARINT (MSB base-128 with the +1 offset per continuation byte)."""
    size = 1
    while n > 0x7F:
        n = (n >> 7) - 1
        size += 1
    return size


def compress_amount(n: int) -> int:
    if n == 0:
        return 0
    e = 0
    while n % 10 == 0 and e < 9:
        n //= 10
        e += 1
    if e < 9:
        d = n % 10
        n //= 10
        return 1 + (n * 9 + d - 1) * 10 + e
    return 1 + (n - 1) * 10 + 9


def coin_entry_bytes(height: int, vout: int, value: int, script_len: int) -> tuple:
    """(key bytes, value bytes) of a chainstate 'C' entry for a non-coinbase P2TR output (P2TR gets no special-script
    compression, so the script is stored as VARINT(len + 6) + script)."""
    key = 1 + 32 + core_varint_len(vout)
    val = core_varint_len(height << 1) + core_varint_len(compress_amount(value)) + \
        core_varint_len(script_len + 6) + script_len
    return key, val


def selftest():
    # BIP 341 wallet test vector (scriptPubKey[1]): one leaf, internal key 1877..cf27
    ik = bytes.fromhex("187791b6f712a8ea41c8ecdd0ee77fab3e85263b37e1ec18a3651926b3a6cf27")
    leaf = bytes.fromhex("20d85a959b0290bf19bb89ed43c916be835475d013da4b362117393e25a48229b8ac")
    assert tapleaf(leaf).hex() == "5b75adecf53548f3ec6ad7d78383bf84cc57b55a3127c72b9a2481752dd88b21"
    assert taproot_output_key(ik, tapleaf(leaf)).hex() == \
        "147c9c57132f6e7ecddba9800bb0c4449251c92a1e60371ee77557b6620f3ea3"
    assert compress_amount(330) == 292 and core_varint_len(292) == 2
    assert core_varint_len(127) == 1 and core_varint_len(128) == 2 and core_varint_len(16511) == 2
    assert core_varint_len(16512) == 3 and core_varint_len(2113663) == 3 and core_varint_len(2113664) == 4
    assert SCRIPT_PATH_VB == 50.5 and KEY_PATH_VB == 57.5


def pushes32(script: bytes) -> list:
    """All 32-byte data pushes in a script."""
    out, i = [], 0
    while i < len(script):
        op = script[i]
        i += 1
        if 1 <= op <= 75:
            n = op
        elif op == 0x4C:
            n = script[i]
            i += 1
        elif op == 0x4D:
            n = int.from_bytes(script[i:i + 2], "little")
            i += 2
        elif op == 0x4E:
            n = int.from_bytes(script[i:i + 4], "little")
            i += 4
        else:
            continue
        if n == 32 and i + 32 <= len(script):
            out.append(script[i:i + 32])
        i += n
    return out


def reveal_leaf(script: bytes, hop: int) -> str:
    """Name the leaf a reveal came from (spec master and LND's earlier encodings)."""
    s = script.hex()
    if hop == 2:
        return "second_level_delay"
    if len(script) >= 34 and script[0] == 0x20 and script[33] == 0x75:
        return "to_local_revoke"            # <local_delayedpubkey> OP_DROP <revocationpubkey> OP_CHECKSIG
    if s.endswith("51b2") or s.endswith("51b275"):
        return "to_remote"                  # <remotepubkey> OP_CHECKSIG[VERIFY] 1 OP_CSV [OP_DROP]
    if s.endswith("b2") or s.endswith("b275"):
        return "to_local_delay"             # <local_delayedpubkey> OP_CHECKSIG[VERIFY] <to_self_delay> OP_CSV
    return "htlc_leaf"


def spend_leaf(script, depth) -> str:
    """Leaf a hop-1 tapscript spend used (same rules as reveal_leaf, plus the key path and the anchor leaf)."""
    if script is None or depth is None or pd.isna(depth) or depth < 0:
        return "key_path"
    script = bytes(script)
    if script == ANCHOR_SCRIPT:
        return "anchor_16csv"
    return reveal_leaf(script, 1)


def half_label(ts) -> np.ndarray:
    """'2025H1' style labels, as a plain array so assignment never aligns on a pandas index."""
    ts = pd.DatetimeIndex(ts)
    return np.asarray(ts.year.astype(str)) + np.where(ts.month > 6, "H2", "H1")


def sat_btc(x) -> float:
    return round(float(x) / 1e8, 8)


def jsonable(o):
    """Turn a nested structure with numpy / pandas keys and values into plain JSON types."""
    if isinstance(o, dict):
        return {str(k.item() if hasattr(k, "item") else k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, float) and np.isnan(o):
        return None
    return o


# ---------------------------------------------------------------------------------------------------------------------
# Step 2: analysis
# ---------------------------------------------------------------------------------------------------------------------

def load(con, H: int):
    """Work tables truncated at the cutoff: a spend or reveal after block H does not exist."""
    co = con.execute(f"""
        SELECT c.close_txid, c.close_height, c.close_tx_key, c.vout, c.value, c.script_type, c.script,
               octet_length(c.script) AS slen,
               CASE WHEN c.spend_tx_key // 32768 <= {H} THEN c.spend_tx_key END AS spend_tx_key,
               o.role, o.spend_txid AS snap_spend_txid
        FROM '{WORK_DIR}/taproot_commit_outputs_*.parquet' c
        LEFT JOIN '{WORK_DIR / 'outputs.parquet'}' o ON o.close_txid = c.close_txid AND o.n = c.vout""").df()
    closes = pd.read_parquet(WORK_DIR / "taproot_closes.parquet")
    l1 = con.execute(f"SELECT *, 1 AS hop FROM '{WORK_DIR}/taproot_inputs_L1_*.parquet' WHERE height <= {H}").df()
    l2 = con.execute(f"SELECT *, 2 AS hop FROM '{WORK_DIR}/taproot_inputs_L2_*.parquet' WHERE height <= {H}").df()
    for d in (l1, l2):
        d["same_out_spend_tx_key"] = d["same_out_spend_tx_key"].where(d["same_out_spend_tx_key"] // 32768 <= H)
    return co, closes, l1, l2


def spend_paths(con, H: int, co, closes, l1, l2) -> tuple:
    """reveals, anchor spend paths, exposure. Returns (json section, anchors table)."""
    sp = pd.concat([l1, l2], ignore_index=True)
    link = lake_file_or_none("outputs_link", H // 1000)        # how far the chain lake's spend links reach
    res = {"lake_end_height": int(Path(link).stem.split("_")[-1]) if link else None,
           "measured_to_height": H, "snapshot_cutoff_height": H,
           "closes_in_list": int(len(closes)), "closes_found_in_lake": int(co["close_txid"].nunique())}
    anchors = co[co["role"] == "anchor"].copy()
    res["anchors"] = int(len(anchors))
    res["anchors_role_mismatch"] = int(((co["role"] == "anchor") & ((co["value"] != ANCHOR_VALUE_SAT) |
                                                                  (co["script_type"] != "p2tr"))).sum())
    res["outputs_without_role"] = int(co["role"].isna().sum())

    # every revealed 32-byte key per commitment
    rev = []
    for r in sp.itertuples(index=False):
        if r.witness_script is None or r.tap_depth is None or pd.isna(r.tap_depth) or r.tap_depth < 0:
            continue
        ws = bytes(r.witness_script)
        for k in pushes32(ws):
            rev.append((r.close_txid, k, int(r.tx_key), int(r.height), r.hop, reveal_leaf(ws, r.hop)))
    rev = pd.DataFrame(rev, columns=["close_txid", "key", "rev_tx_key", "rev_height", "hop", "leaf"])
    res["revealed_keys"] = int(len(rev))

    # match every anchor against the keys revealed in its own commitment's spends
    cache = {}
    by_close = {c: g for c, g in rev.groupby("close_txid")}
    rows = []
    for a in anchors.itertuples(index=False):
        out_key = bytes(a.script)[2:]
        g = by_close.get(a.close_txid)
        if g is None:
            continue
        for r in g.itertuples(index=False):
            if r.key not in cache:
                cache[r.key] = taproot_output_key(r.key, ANCHOR_ROOT)
            if cache[r.key] == out_key:
                rows.append((a.close_txid, a.vout, r.key.hex(), r.rev_tx_key, r.rev_height, r.hop, r.leaf))
    m = pd.DataFrame(rows, columns=["close_txid", "vout", "internal_key", "rev_tx_key", "rev_height", "hop", "rev_leaf"])
    res["tweaks_computed"] = len(cache)
    first = (m.sort_values("rev_tx_key").groupby(["close_txid", "vout"], as_index=False)
             .agg(internal_key=("internal_key", "first"), rev_tx_key=("rev_tx_key", "first"),
                  rev_height=("rev_height", "first"), rev_hop=("hop", "first"), rev_leaf=("rev_leaf", "first"),
                  n_reveals=("rev_tx_key", "size")))
    res["anchors_with_two_distinct_matching_keys"] = int((m.groupby(["close_txid", "vout"])["internal_key"]
                                                          .nunique() > 1).sum())

    # anchor spends, by witness
    asp = l1.merge(anchors[["close_txid", "vout"]], left_on=["close_txid", "prev_vout"],
                   right_on=["close_txid", "vout"])[["close_txid", "vout", "tx_key", "height", "witness_items",
                                                     "witness_script", "tap_depth", "has_annex", "schnorr_len",
                                                     "n_in", "vsize", "fee"]]
    asp = asp.rename(columns={"tx_key": "spend_tx_key", "height": "spend_height"})

    def path(r):
        ws = bytes(r.witness_script) if r.witness_script is not None else b""
        if r.tap_depth == 0 and ws == ANCHOR_SCRIPT:
            return "script_16csv"
        if r.tap_depth < 0 and r.witness_items - int(bool(r.has_annex)) == 1:
            return "key_path"
        return "other"
    asp["path"] = [path(r) for r in asp.itertuples(index=False)]

    A = anchors[["close_txid", "close_height", "close_tx_key", "vout", "value", "slen", "snap_spend_txid"]].merge(
        asp, on=["close_txid", "vout"], how="left").merge(first, on=["close_txid", "vout"], how="left")
    t_utc = pd.to_datetime(closes.set_index("close_txid")["close_time_utc"], utc=True)
    A["close_time_utc"] = A["close_txid"].map(t_utc)
    A["year"] = A["close_time_utc"].dt.year                    # UTC, as in 04_anchors.py
    A["path"] = A["path"].fillna("unspent")
    A["blocks_to_spend"] = A["spend_height"] - A["close_height"]
    A["revealed"] = A["rev_tx_key"].notna()

    def order(r):
        if r.path != "script_16csv":
            return None
        if pd.isna(r.rev_tx_key):
            return "no_reveal_in_commitment"
        if r.rev_tx_key < r.spend_tx_key:
            return "after_reveal_block" if r.rev_height < r.spend_height else "after_reveal_same_block"
        if r.rev_tx_key == r.spend_tx_key:
            return "same_tx_as_reveal"
        return "before_reveal"
    A["order"] = [order(r) for r in A.itertuples(index=False)]

    # 1. the reveal hypothesis: every script-path (third-party) spend comes after the reveal
    res["anchor_spend_paths"] = A["path"].value_counts().to_dict()
    res["anchors_spent_in_lake_not_in_snapshot"] = int(((A["path"] != "unspent") & A["snap_spend_txid"].isna()).sum())
    res["anchors_spent_in_snapshot_not_in_lake"] = int(((A["path"] == "unspent") & A["snap_spend_txid"].notna()).sum())
    s16 = A[A["path"] == "script_16csv"]
    res["script_path_spends"] = int(len(s16))
    res["script_path_order"] = s16["order"].value_counts().to_dict()
    viol = s16[~s16["order"].isin(["after_reveal_block", "after_reveal_same_block"])]
    res["violations"] = int(len(viol))
    if len(viol):
        viol.drop(columns=["witness_script"]).to_parquet(WORK_DIR / "taproot_violations.parquet", index=False)
    scan = sorted(WORK_DIR.glob("taproot_scan16_*.parquet"))
    if scan:
        n, chunks = con.execute(f"""SELECT count(*) FILTER (WHERE height <= {H}),
                                           count(DISTINCT height // 1000)
                                    FROM '{WORK_DIR}/taproot_scan16_*.parquet'""").fetchone()
        res["chainwide_op16_csv_script_spends"] = {"inputs": int(n), "files": len(scan),
                                                   "heights_scanned": f"{SCAN_FROM_CHUNK * 1000}..{H}"}
    if len(s16):
        res["script_path_csv_ok"] = int((s16["blocks_to_spend"] >= CSV_BLOCKS).sum())

    # 2. how the spent anchors were spent
    kp = A[A["path"] == "key_path"]
    res["key_path"] = {
        "spends": int(len(kp)), "witness_items_1": int((kp["witness_items"] == 1).sum()),
        "schnorr_64_or_65": int(kp["schnorr_len"].isin([64, 65]).sum()),
        "same_block_as_close": int((kp["blocks_to_spend"] == 0).sum()),
        "within_16_blocks": int((kp["blocks_to_spend"] < CSV_BLOCKS).sum()),
        "blocks_to_spend_median": float(kp["blocks_to_spend"].median()),
        "after_key_revealed": int((kp["rev_tx_key"] < kp["spend_tx_key"]).sum()),
        "by_close_year": kp.groupby("year").size().to_dict(),
        "same_block_by_close_year": kp[kp["blocks_to_spend"] == 0].groupby("year").size().to_dict()}

    # 3. the other direction: anchors whose key is public, swept or not?
    tab = (A.assign(key=A["revealed"].map({True: "revealed", False: "not revealed"}))
           .groupby(["year", "key", "path"]).size().unstack(fill_value=0))
    res["by_year_key_path"] = {f"{y}|{k}": row.to_dict() for (y, k), row in tab.iterrows()}
    old = A[A["close_height"] <= H - SPEND_CENSOR_BLOCKS]
    tab2 = (old.assign(key=old["revealed"].map({True: "revealed", False: "not revealed"}))
            .groupby(["year", "key", "path"]).size().unstack(fill_value=0))
    res["by_year_key_path_censored"] = {f"{y}|{k}": row.to_dict() for (y, k), row in tab2.iterrows()}

    rv = A[A["revealed"]].copy()
    rv["open"] = (rv["path"] == "unspent") | (rv["spend_tx_key"] > rv["rev_tx_key"])   # unspent when the key went public
    op = rv[rv["open"]].copy()
    op["exposed_blocks"] = op["spend_height"].fillna(H) - op["rev_height"]               # key public, anchor unspent
    still = int((op["path"] == "unspent").sum())
    res["exposed_anchors"] = {
        "anchors_unspent_when_key_revealed": int(len(op)),
        "later_swept_by_script_path": int((op["path"] == "script_16csv").sum()),
        "later_spent_by_owner_key_path": int((op["path"] == "key_path").sum()),
        "still_unspent_at_lake_end": still,          # name kept for 07_numbers.py: unspent at measured_to_height
        "still_unspent_at_cutoff": still,
        "exposed_ge_16_blocks": int((op["exposed_blocks"] >= 16).sum()),
        "exposed_ge_1008_blocks": int((op["exposed_blocks"] >= 1008).sum()),
        "exposed_ge_4032_blocks": int((op["exposed_blocks"] >= 4032).sum()),
        "exposed_blocks_median": float(op["exposed_blocks"].median()),
        "anchor_block_exposure_total": int(op["exposed_blocks"].sum()),
        "reveal_leaf": op["rev_leaf"].value_counts().to_dict(),
        "by_reveal_year": op.groupby(pd.cut(op["rev_height"], [0, 823786, 877259, 929893, H],
                                            labels=["2023", "2024", "2025", "2026"]), observed=False)
                            .size().to_dict()}
    res["revealed_anchors"] = {"anchors": int(len(rv)), "first_reveal_leaf": rv["rev_leaf"].value_counts().to_dict(),
                               "blocks_close_to_reveal_median": float((rv["rev_height"] - rv["close_height"]).median())}

    nr = A[~A["revealed"]]
    na = co[(co["role"] != "anchor") & (co["script_type"] == "p2tr")]
    unspent_other = na[na["spend_tx_key"].isna()].groupby("close_txid").size()
    res["not_revealed"] = {"anchors": int(len(nr)), "unspent": int((nr["path"] == "unspent").sum()),
                           "key_path": int((nr["path"] == "key_path").sum()),
                           "commitment_has_unspent_non_anchor_p2tr": int(nr["close_txid"].isin(unspent_other.index).sum()),
                           "commitment_has_no_unspent_non_anchor_p2tr":
                               int((~nr["close_txid"].isin(unspent_other.index)).sum())}
    res["script_path_sweep_input"] = {"weight": SCRIPT_PATH_WU, "vbytes": SCRIPT_PATH_VB,
                                      "breakeven_sat_vb": ANCHOR_VALUE_SAT / SCRIPT_PATH_VB}
    return res, A


def lake_file_or_none(kind: str, c: int):
    folder = LINKED if kind.endswith("_link") else PARTS
    hits = sorted(folder.glob(f"{kind}_{c * 1000}_*.parquet"))
    return str(hits[0]) if len(hits) == 1 else None


def economics(con, H: int, A: pd.DataFrame, co, l1, l2) -> dict:
    """sweep prize, UTXO footprint, owner spends. All block statistics from stage 00's blockstats."""
    blk = con.execute(f"""SELECT height, time, txs, minfeerate AS minfr, p10, p25, p50
                          FROM '{WORK_DIR / 'blockstats.parquet'}' WHERE height BETWEEN 800000 AND {H}
                          ORDER BY height""").df()
    assert blk["height"].iloc[-1] == H and (blk["height"].diff().dropna() == 1).all()
    blk["t"] = pd.to_datetime(blk["time"], unit="s", utc=True)
    blk["half"] = half_label(blk["t"])
    # coinbase-only blocks report minfeerate 0 by construction: excluded from the cheapest-feerate search
    blk["minfr_tx"] = np.where(blk["txs"] > 1, blk["minfr"], np.nan)
    h0 = int(blk["height"].iloc[0])
    suf = pd.Series(blk["minfr_tx"].fillna(np.inf).to_numpy()[::-1]).cummin().to_numpy()[::-1]
    suf = np.where(np.isinf(suf), np.nan, suf)       # cheapest mined feerate from height h to the cutoff
    has_tx = (blk["txs"] > 1).to_numpy()
    nxt = np.full(len(blk), len(blk) - 1)
    last = len(blk) - 1
    for i in range(len(blk) - 1, -1, -1):
        if has_tx[i]:
            last = i
        nxt[i] = last

    def at(h, col):
        return blk[col].to_numpy()[np.asarray(h, dtype=int) - h0]

    A = A.copy()
    A["close_height"] = A["close_height"].astype(int)
    unspent = A[A["path"] == "unspent"].copy()
    keypub = unspent[unspent["revealed"]].copy()
    out: dict = {"blockstats": "data/work/blockstats.parquet (stage 00, chain lake getblockstats), to the cutoff",
                 "minfeerate_note": "getblockstats minfeerate is an integer in sat/vB, truncated: 0 means the "
                                    "cheapest transaction paid below 1 sat/vB. Coinbase-only blocks excluded."}

    # 1. sweep prize
    p1: dict = {"script_path_input": {"weight_wu": SCRIPT_PATH_WU, "vbytes": SCRIPT_PATH_VB,
                                      "breakeven_sat_vb": ANCHOR_VALUE_SAT / SCRIPT_PATH_VB},
                "key_path_input": {"weight_wu": KEY_PATH_WU, "vbytes": KEY_PATH_VB,
                                   "breakeven_sat_vb": ANCHOR_VALUE_SAT / KEY_PATH_VB},
                "tx_overhead_vb_per_batch": TX_OVERHEAD_VB}
    for name, d_ in (("all_unspent", unspent), ("key_public_unspent", keypub)):
        n = len(d_)
        d = {"anchors": n, "gross_sat": n * ANCHOR_VALUE_SAT, "gross_btc": sat_btc(n * ANCHOR_VALUE_SAT)}
        for r in FIXED_RATES:
            net = n * max(0.0, ANCHOR_VALUE_SAT - r * SCRIPT_PATH_VB)
            d[f"net_btc_at_{r:g}_sat_vb"] = sat_btc(net)
        p1[name] = d
    # cheapest mined feerate once each key-public anchor became sweepable: first eligible block max(close+16, reveal)
    elig = np.minimum(np.maximum(keypub["close_height"].to_numpy() + CSV_BLOCKS,
                                 keypub["rev_height"].to_numpy().astype(int)), H)
    f_window = suf[elig - h0]
    f_first = at(nxt[elig - h0] + h0, "minfr")
    assert not np.isnan(f_window).any()

    def net_at(f):
        return float(np.clip(ANCHOR_VALUE_SAT - np.asarray(f, dtype=float) * SCRIPT_PATH_VB, 0, None).sum())
    p1["key_public_unspent"]["post_reveal_min_block_feerate"] = {
        "rule": "first eligible block = max(close_height + 16, reveal height), up to the cutoff; minfeerate is "
                "truncated, so the net at minfeerate is an upper bound and at minfeerate + 1 a lower bound",
        "feerate_sat_vb_distribution": pd.Series(f_window).value_counts().sort_index().to_dict(),
        "net_btc_upper": sat_btc(net_at(f_window)), "net_btc_lower": sat_btc(net_at(f_window + 1)),
        "first_eligible_block_share_minfeerate_zero": round(float((f_first == 0).mean()), 4)}
    sw = json.loads((DATA_DIR / "sweepers.json").read_text())
    p1["largest_p2wsh_sweeper"] = {
        "net_btc": sw["largest_net_btc"],
        "key_public_gross_over_largest_net": round(p1["key_public_unspent"]["gross_btc"] / sw["largest_net_btc"], 3),
        "all_unspent_gross_over_largest_net": round(p1["all_unspent"]["gross_btc"] / sw["largest_net_btc"], 3)}
    out["1_sweep_prize"] = p1

    # 2. UTXO footprint
    ua = unspent.copy()
    assert (ua["value"] == ANCHOR_VALUE_SAT).all() and (ua["slen"] == 34).all()
    kb = [coin_entry_bytes(int(h), int(v), int(x), int(s)) for h, v, x, s in
          zip(ua["close_height"], ua["vout"], ua["value"], ua["slen"])]
    ua["coin_b"] = [k + v for k, v in kb]
    p2: dict = {"unspent_anchors": int(len(ua)), "unspent_btc": sat_btc(ua["value"].sum()),
                "ctxout_bytes_each": 8 + 1 + 34,
                "coin_entry_bytes_each": sorted(set(ua["coin_b"].tolist())),
                "coin_entry_bytes_total": int(ua["coin_b"].sum()),
                "coin_entry_note": "LevelDB key 'C'+txid+VARINT(vout), value VARINT(height*2)+VARINT(CompressAmount(330))"
                                   "+VARINT(34+6)+34-byte script; chainstate LevelDB uses kNoCompression; excludes "
                                   "LevelDB per-record overhead",
                "dust_threshold_p2tr_at_3_sat_vb": (43 + 32 + 4 + 1 + 107 // 4 + 4) * 3}
    v3 = con.execute(f"""
        SELECT fmt, role, count(*) FILTER (WHERE spend_txid IS NULL) AS unspent,
               coalesce(sum(value_sat) FILTER (WHERE spend_txid IS NULL), 0) AS unspent_sat
        FROM '{WORK_DIR / 'outputs.parquet'}' WHERE role <> 'nulldata' GROUP BY ALL""").df()
    tot_u, tot_sat = int(v3["unspent"].sum()), int(v3["unspent_sat"].sum())
    tap = v3[(v3["fmt"] == "taproot") & (v3["role"] == "anchor")].iloc[0]
    p2["force_close_outputs_unspent_at_cutoff"] = {
        "rule": "outputs of all force closes in the window (listed + unlisted), unspent at the cutoff, OP_RETURN "
                "outputs excluded (work/outputs.parquet)",
        "total_unspent": tot_u, "total_unspent_btc": sat_btc(tot_sat),
        "taproot_anchor_unspent": int(tap["unspent"]),
        "taproot_anchor_share_count": int(tap["unspent"]) / tot_u,
        "taproot_anchor_share_value": int(tap["unspent_sat"]) / tot_sat,
        "p2wsh_anchor_unspent": int(v3[(v3["fmt"] == "anchor") & (v3["role"] == "anchor")]["unspent"].sum())}
    cs = pd.read_parquet(WORK_DIR / "taproot_coinstats.parquet").iloc[0]
    hcs = int(cs["height"])
    live = A[(A["close_height"] <= hcs) & (A["spend_height"].isna() | (A["spend_height"] > hcs))]
    p2["utxo_set_share"] = {"checkpoint_height": hcs, "utxo_set_txouts": int(cs["txouts"]),
                            "taproot_anchors_unspent_at_checkpoint": int(len(live)),
                            "share_of_txouts": len(live) / int(cs["txouts"]),
                            "share_of_bogosize": len(live) * 84 / int(cs["bogosize"])}
    out["2_utxo_footprint"] = p2

    # 3. owner key-path spends
    roles = con.execute(f"SELECT close_txid, n AS prev_vout, role FROM '{WORK_DIR / 'outputs.parquet'}' "
                        "WHERE fmt = 'taproot'").df()
    s1 = l1.merge(roles, on=["close_txid", "prev_vout"], how="left")
    assert s1["role"].notna().all()
    s1["leaf"] = [spend_leaf(w, d) for w, d in zip(s1["witness_script"], s1["tap_depth"])]
    sib = (s1.assign(non_anchor=s1["role"] != "anchor")
           .groupby(["close_txid", "tx_key"]).agg(n_non_anchor=("non_anchor", "sum")).reset_index())
    anchors_in_tx = s1[s1["role"] == "anchor"].groupby("tx_key").size().rename("anchors_in_tx")
    k = A[A["path"] == "key_path"].copy()
    k["spend_tx_key"] = k["spend_tx_key"].astype("int64")
    k["spend_height"] = k["spend_height"].astype(int)
    k = k.merge(sib, left_on=["close_txid", "spend_tx_key"], right_on=["close_txid", "tx_key"], how="left")
    k = k.merge(anchors_in_tx, left_on="spend_tx_key", right_index=True, how="left")
    k["spend_half"] = half_label(at(k["spend_height"], "t"))
    k["kind"] = np.select([k["blocks_to_spend"] == 0, k["n_non_anchor"] > 0],
                          ["same_block_as_close", "with_own_commitment_output"], "anchor_only_tx")
    k["spend_feerate"] = k["fee"] / k["vsize"]
    p3: dict = {"halves": f"block time, UTC; {LATE} ends at the cutoff"}
    by_half = k.groupby(["spend_half", "kind"]).size().unstack(fill_value=0)
    by_half["total"] = by_half.sum(axis=1)
    hb = blk.groupby("half").agg(first_t=("t", "min"), last_t=("t", "max"))
    by_half["months"] = ((hb["last_t"] - hb["first_t"]).dt.total_seconds() / (30.4375 * 86400)).round(2) \
        .reindex(by_half.index)
    by_half["per_month"] = (by_half["total"] / by_half["months"]).round(1)
    p3["key_path_spends_by_spend_half"] = by_half.reset_index().to_dict(orient="records")
    p3["key_path_spend_txs"] = {
        "median_anchors_in_tx_by_spend_half": k.groupby("spend_half")["anchors_in_tx"].median().to_dict(),
        "median_tx_feerate_by_spend_half": k.groupby("spend_half")["spend_feerate"].median().round(2).to_dict()}

    A["close_half"] = half_label(at(A["close_height"], "t"))
    old = A[A["close_height"] <= H - SPEND_CENSOR_BLOCKS].copy()
    old["spent_in_window"] = (old["path"] == "key_path") & (old["blocks_to_spend"] <= SPEND_CENSOR_BLOCKS)
    coh = old.groupby("close_half").agg(anchors=("vout", "size"), owner_spent_4032=("spent_in_window", "sum"))
    coh["share"] = (coh["owner_spent_4032"] / coh["anchors"]).round(4)
    p3["close_cohorts_owner_spent_within_4032_blocks"] = coh.reset_index().to_dict(orient="records")

    # the opportunity test: the owner swept the matching main output; did the anchor go into that same transaction?
    txfee = (pd.concat([l1[["tx_key", "fee", "vsize", "n_in"]], l2[["tx_key", "fee", "vsize", "n_in"]]])
             .drop_duplicates("tx_key").set_index("tx_key")
             .rename(columns={"fee": "sweep_fee", "vsize": "sweep_vsize", "n_in": "sweep_n_in"}))
    r = A[A["revealed"] & A["rev_leaf"].isin(["to_local_delay", "to_remote", "second_level_delay"])].copy()
    r["rev_tx_key"] = r["rev_tx_key"].astype("int64")
    r["rev_height"] = r["rev_height"].astype(int)
    r = r[(r["path"] == "unspent") | (r["spend_tx_key"] >= r["rev_tx_key"])].copy()
    r["in_same_tx"] = (r["path"] == "key_path") & (r["spend_tx_key"] == r["rev_tx_key"])
    r["spent_later"] = (r["path"] == "key_path") & (r["spend_tx_key"] > r["rev_tx_key"])
    r["rev_half"] = half_label(at(r["rev_height"], "t"))
    r = r.join(txfee, on="rev_tx_key")
    r["sweep_feerate"] = r["sweep_fee"] / r["sweep_vsize"]
    r["below_breakeven"] = r["sweep_feerate"] < ANCHOR_VALUE_SAT / KEY_PATH_VB
    r["rev_block_p50"] = at(r["rev_height"], "p50")
    blocksize = A.groupby("close_height")["close_txid"].nunique().rename("taproot_closes_in_block")
    r = r.join(blocksize, on="close_height")
    r["batch_block"] = r["taproot_closes_in_block"] >= BATCH_BLOCK_MIN
    opp = r.groupby("rev_half").agg(
        owner_sweeps=("vout", "size"), anchor_in_same_tx=("in_same_tx", "sum"), anchor_spent_later=("spent_later", "sum"),
        median_sweep_feerate=("sweep_feerate", "median"), share_sweep_below_breakeven=("below_breakeven", "mean"),
        median_block_p50=("rev_block_p50", "median"), share_from_batch_blocks=("batch_block", "mean"))
    opp["share_in_same_tx"] = opp["anchor_in_same_tx"] / opp["owner_sweeps"]
    p3["owner_sweeps_of_matching_output_by_half"] = opp.round(4).reset_index().to_dict(orient="records")
    p3["owner_sweeps_note"] = (
        "One row per anchor whose owner swept the matching main output while the anchor was unspent (reveal leaf "
        "to_local_delay / to_remote / second_level_delay). anchor_in_same_tx: the anchor was an input of that sweep. "
        "share_sweep_below_breakeven: the sweep paid less than 330/57.5 = 5.74 sat/vB, so adding the anchor by key "
        "path would have added value to the owner.")
    p3["owner_sweeps_by_half_leaf_batch_block"] = (r.groupby(["rev_half", "rev_leaf", "batch_block"])
                                                   .agg(n=("vout", "size"), in_same_tx=("in_same_tx", "sum"))
                                                   .reset_index().to_dict(orient="records"))

    # CPFP need: the commitment's feerate against the block before it, and any output spent in its own block
    ul = con.execute(f"SELECT close_txid, feerate_sat_vb FROM '{SNAPSHOT_DIR / 'ul_closes.parquet'}'").df()
    cl = A.groupby("close_txid").agg(close_height=("close_height", "first"), close_half=("close_half", "first"),
                                     anchor_same_block=("blocks_to_spend", lambda s: bool((s == 0).any())))
    cl = cl.join(ul.set_index("close_txid"))
    same = co.assign(sh=co["spend_tx_key"] // 32768)
    same = same[same["sh"] == same["close_height"]].groupby("close_txid").size()
    cl["child_same_block"] = cl.index.isin(same.index)
    hh = cl["close_height"].to_numpy().astype(int)
    cl["clears_prev_block_p10"] = cl["feerate_sat_vb"] >= at(hh - 1, "p10")
    cl = cl.join(blocksize, on="close_height")
    cl["batch_block"] = cl["taproot_closes_in_block"] >= BATCH_BLOCK_MIN
    cp = cl.groupby("close_half").agg(
        closes=("close_height", "size"), median_commit_feerate=("feerate_sat_vb", "median"),
        share_clears_prev_block_p10=("clears_prev_block_p10", "mean"),
        share_child_same_block=("child_same_block", "mean"), share_anchor_same_block=("anchor_same_block", "mean"),
        share_in_batch_blocks=("batch_block", "mean"))
    p3["commitments_by_close_half"] = cp.round(4).reset_index().to_dict(orient="records")
    tb = cl.groupby("close_height").size()
    p3["close_concentration"] = {"batch_block_min_closes": BATCH_BLOCK_MIN, "blocks_with_taproot_closes": int(len(tb)),
                                 "batch_blocks": int((tb >= BATCH_BLOCK_MIN).sum()),
                                 "share_of_closes_in_batch_blocks":
                                     round(float(tb[tb >= BATCH_BLOCK_MIN].sum() / tb.sum()), 4)}
    out["3_owner_key_path"] = p3

    # ---- the values the paper quotes (07_numbers.py formats them) ------------------------------------------------
    def cosweep(half, leaf, batch=None):
        g = r[(r["rev_half"] == half) & (r["rev_leaf"] == leaf)]
        if batch is not None:
            g = g[g["batch_block"] == batch]
        return {"n": int(len(g)), "in_same_tx": int(g["in_same_tx"].sum()),
                "share": float(g["in_same_tx"].mean()) if len(g) else None}
    spends = {x["spend_half"]: x for x in p3["key_path_spends_by_spend_half"]}
    opp_d = {x["rev_half"]: x for x in p3["owner_sweeps_of_matching_output_by_half"]}
    fc = p2["force_close_outputs_unspent_at_cutoff"]
    out["paper"] = {
        "early_half": EARLY, "late_half": LATE,
        "unspent_anchors": p1["all_unspent"]["anchors"], "unspent_btc": p1["all_unspent"]["gross_btc"],
        "key_public_unspent_anchors": p1["key_public_unspent"]["anchors"],
        "key_public_btc": p1["key_public_unspent"]["gross_btc"],
        "key_public_net_btc_at_1_sat_vb": p1["key_public_unspent"]["net_btc_at_1_sat_vb"],
        "script_path_vb": SCRIPT_PATH_VB, "script_path_breakeven_sat_vb": ANCHOR_VALUE_SAT / SCRIPT_PATH_VB,
        "largest_p2wsh_sweeper_net_btc": sw["largest_net_btc"],
        "unspent_output_share": fc["taproot_anchor_share_count"], "unspent_value_share": fc["taproot_anchor_share_value"],
        "force_close_outputs_unspent_at_cutoff": fc["total_unspent"],
        "coin_entry_bytes": p2["coin_entry_bytes_each"][0] if len(p2["coin_entry_bytes_each"]) == 1 else None,
        "utxo_set_share": p2["utxo_set_share"]["share_of_txouts"],
        "utxo_set_checkpoint_height": hcs,
        "owner_spends_early": int(spends.get(EARLY, {}).get("total", 0)),
        "owner_spends_late": int(spends.get(LATE, {}).get("total", 0)),
        "owner_spends_late_same_block": int(spends.get(LATE, {}).get("same_block_as_close", 0)),
        "cosweep_to_remote_early": cosweep(EARLY, "to_remote"),
        "cosweep_to_remote_late": cosweep(LATE, "to_remote"),
        "cosweep_to_remote_late_outside_batch_blocks": cosweep(LATE, "to_remote", batch=False),
        "owner_sweep_feerate_median_early": opp_d[EARLY]["median_sweep_feerate"],
        "owner_sweep_feerate_median_late": opp_d[LATE]["median_sweep_feerate"],
    }
    return out


def analyse() -> int:
    selftest()
    H = cutoff()
    con = duckdb.connect()
    con.execute("SET threads=6")
    t0 = time.time()
    co, closes, l1, l2 = load(con, H)
    res, A = spend_paths(con, H, co, closes, l1, l2)
    res["economics"] = economics(con, H, A, co, l1, l2)
    A.drop(columns=["witness_script", "internal_key"]).to_parquet(WORK_DIR / "taproot_anchors.parquet", index=False)
    res = jsonable(res)
    path = write_json("taproot_anchor_paths.json", res)
    print(json.dumps({k: res[k] for k in ("anchors", "anchor_spend_paths", "script_path_spends",
                                          "chainwide_op16_csv_script_spends", "anchors_spent_in_lake_not_in_snapshot",
                                          "anchors_spent_in_snapshot_not_in_lake")}, indent=1))
    print(json.dumps(res["exposed_anchors"], indent=1))
    print(json.dumps(res["economics"]["paper"], indent=1))
    print(f"wrote {path} ({time.time() - t0:.0f} s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--extract", nargs="+", metavar="STAGE",
                    help="closes | commits A B | inputs L A B | scan16 A B | coinstats (reads the chain lake)")
    a = ap.parse_args()
    if not a.extract:
        return analyse()
    t0 = time.time()
    cmd, *args = a.extract
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if cmd == "closes":
        extract_closes()
    elif cmd == "commits":
        extract_commits(int(args[0]), int(args[1]))
    elif cmd == "inputs":
        extract_inputs(int(args[0]), int(args[1]), int(args[2]))
    elif cmd == "scan16":
        extract_scan16(int(args[0]), int(args[1]))
    elif cmd == "coinstats":
        extract_coinstats()
    else:
        raise SystemExit(f"unknown stage {cmd}")
    print(f"{cmd} {' '.join(args)}: {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
