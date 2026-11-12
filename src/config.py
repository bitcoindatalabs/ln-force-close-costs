"""
Paths and parameters shared by the pipeline stages of paper 03 (what a Lightning force close costs).

Every stage reads the frozen snapshot written by 01_snapshot.py (data/snapshot/), never the live lake, so reruns
reproduce the published numbers. Set LN_ONCHAIN_DIR to the scanner's output folder to rebuild the snapshot.
"""

import os
from pathlib import Path

# --- paths ---------------------------------------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PAPER_ROOT = SRC_DIR.parent
PAPER_DIR = PAPER_ROOT / "paper"
FIGURES_DIR = PAPER_DIR / "figures"
TABLES_DIR = PAPER_DIR / "tables"
DATA_DIR = PAPER_ROOT / "data"                 # *.json aggregates are released; everything else is ignored by git
SNAPSHOT_DIR = DATA_DIR / "snapshot"           # frozen inputs (not committed)
WORK_DIR = DATA_DIR / "work"                   # per-stage intermediate tables (not committed)

# Input: the close scanner's tables (ln_close_tx_scanner.py) and its parts/ folder. The 2018-2026 rerun
# reads the v3 rescan directly: LN_ONCHAIN_DIR=/data/Lightning/Data/parquet/onchain/v3.
ONCHAIN_DIR = Path(os.environ.get("LN_ONCHAIN_DIR", "/data/Lightning/Data/parquet/onchain"))
LN_PARQUET_DIR = Path(os.environ.get("LN_PARQUET_DIR", "/data/Lightning/Data/parquet"))
# The list of public channels (ln_funding_outpoints, ln_funding_txs) stays in the live onchain/ folder, also when
# ONCHAIN_DIR points at a rescan folder that has no copy of it.
FUNDING_DIR = Path(os.environ.get("LN_FUNDING_DIR", LN_PARQUET_DIR / "onchain"))
# Optional: the chain lake's blocks table (bitcoin_chain_extractor.py); 00_blockstats.py prefers it when it covers
# the study window and otherwise asks the node.
CHAIN_LAKE_DIR = Path(os.environ.get("CHAIN_LAKE_DIR", "/data/bitcoin/data/chain"))

# --- study window --------------------------------------------------------------------------------------------------
# The v3 rescan starts at block 503,816 (2018-01-12 04:49:23 UTC, Lightning's first mainnet channels; the paper's window) and ends at block 974,575 (2026-11-04), so spends after that height are not seen (censored). The cutoff is the
# last block mined before 2026-11-01 00:00 UTC; 01_snapshot.py resolves it to a height and records it in
# data/snapshot_manifest.json.
SCAN_START_HEIGHT = 503_816
SCAN_START_TS = 1_515_732_563          # block 503,816's timestamp
SCAN_END_HEIGHT = 974_575              # last block of the v3 rescan
CUTOFF_UTC = "2026-11-01T00:00:00Z"

# Registry coverage: the list of public channels barely covers channels that closed in 2018 (the lnresearch
# gossip archive and mempool.space history start densely in January 2019), so for 2018 only force-close totals
# (listed + unlisted) are complete. The public/unlisted split, cooperative counts, force share and per-channel rates
# are coverage-limited before PUBLIC_FROM; January 2019 is still ramping (24% of force closes listed). Outputs carry a coverage flag per year
# (and per month). tables and figures show 2018 as force-close totals only.
PUBLIC_FROM = "2019-01-01"   # public, cooperative, share and rate series start here
COVERAGE = {2018: "registry_limited", 2019: "ramping_january"}     # every other year: "full"


def coverage(year: int) -> str:
    return COVERAGE.get(int(year), "full")

# Spends are right-censored near the cutoff. Statistics about what happens to a close's outputs use only closes
# at least this many blocks (about 4 weeks) before the cutoff.
SPEND_CENSOR_BLOCKS = 4_032

# --- protocol constants (BOLT 3) -----------------------------------------------------------------------------------
ANCHOR_VALUE_SAT = 330
ANCHOR_CSV_BLOCKS = 16                 # anyone can spend an anchor 16 blocks after the commitment confirms
# A third-party anchor spend is an input with an empty signature: 41 B outpoint+sequence+len, witness 1+1+41+... ;
# measured on mainnet sweeps: 207 weight units per input (a13_gap_econ.py), i.e. 51.75 vB.
ANCHOR_SWEEP_INPUT_VB = 207 / 4

# --- fee regimes ---------------------------------------------------------------------------------------------------
# Calendar periods, checked against the median block fee rate (blockstats, half-year medians of the block p50 in
# sat/vB: 2018 9 / 6, 2019 20 / 14, 2020 14 / 46, 2021 59 / 5, 2022 5 / 9, 2023 15 / 16, 2024 24 / 4, 2025 2 / 1,
# 2026 1 / 1).
FEE_PERIODS = [
    ("2018", "2018-01-01", "2019-01-01"),
    ("2019-20", "2019-01-01", "2021-01-01"),
    ("2021", "2021-01-01", "2022-01-01"),
    ("2022", "2022-01-01", "2023-01-01"),
    ("2023-24", "2023-01-01", "2025-01-01"),
    ("2025-26", "2025-01-01", "2026-11-01"),
]

# --- dates that matter for the story -------------------------------------------------------------------------------
EVENTS = {
    "core_28": ("2024-10-04", "Bitcoin Core 28.0: TRUC, 1p1c relay, pay-to-anchor"),
    "core_29_1": ("2025-09-04", "Bitcoin Core 29.1: 0.1 sat/vB relay floor"),
    "core_31": ("2026-04-14", "Bitcoin Core 31.0: CPFP carve-out removed"),
    "bolts_zero_fee": ("2026-05-01", "BOLTs: zero-fee commitments merged"),
}

RANDOM_SEED = 42

# --- figure palette ------------------------------------------------------------------------------------------------
# Matches papers 01–02 and the deep-dives site; series are also labelled directly. Pairs that share a chart were
# checked with the dataviz validator (light surface): owner/third party/miners and force/unlisted pass all checks;
# green vs terracotta is in the 6–8 CVD band, so wherever both appear they are direct-labelled.
PALETTE = {
    "force": "#D4603A",        # terracotta: the costly outcome
    "mutual": "#8893A0",       # neutral grey: the baseline
    "unlisted": "#4a3aa7",     # purple: channels outside our public list
    "owner": "#2a78d6",        # blue: the channel owner
    "third_party": "#D4603A",  # terracotta: sweepers
    "miners": "#4a3aa7",       # purple: fees
    "legacy": "#8893A0",       # neutral grey: the old format
    "anchor": "#2a78d6",
    "taproot": "#4a3aa7",
    "zero_fee": "#1a9e6e",
    "ink": "#242D38",
    "ink_mark": "#4A5563",
    "muted": "#8893A0",
    "grid": "#EFEBE4",
}
CLIENT_PALETTE = {"LND": "#2a78d6", "CLN": "#D4603A", "Eclair": "#1a9e6e", "LDK": "#4a3aa7", "Unknown": "#8893A0"}
