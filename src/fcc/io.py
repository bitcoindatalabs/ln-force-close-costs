"""Loading the frozen snapshot, and small JSON helpers shared by the stages."""

import json
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, SNAPSHOT_DIR, WORK_DIR  # noqa: E402

TABLES = {
    "c": "closes",               # every close of a channel in the public list (force, mutual, splice)
    "s": "spends",               # every spend of an output of a listed force close
    "u": "ul_closes",            # force closes of channels outside the public list (found by the commitment pattern)
    "us": "ul_spends",           # spends of their outputs, plus anchor-script spends of commitments we never saw
    "fo": "funding_outpoints",   # every public channel ever seen: scid -> funding outpoint, capacity
    "h": "client_history",       # paper 01's client labels with validity intervals (from 2023-05-08)
}


def manifest() -> dict:
    return json.loads((DATA_DIR / "snapshot_manifest.json").read_text(encoding="utf-8"))


def connect(threads: int = 6) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with one view per snapshot table, plus `fc`: all force closes, listed or not, with format."""
    con = duckdb.connect()
    con.execute(f"SET threads={threads}")
    con.execute("SET TimeZone = 'UTC'")
    for view, name in TABLES.items():
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM '{SNAPSHOT_DIR / name}.parquet'")
    for view, name in (("fc", "force_closes"), ("o", "outputs")):
        if (WORK_DIR / f"{name}.parquet").exists():
            con.execute(f"CREATE VIEW {view} AS SELECT * FROM '{WORK_DIR / name}.parquet'")
    if (WORK_DIR / "blockstats.parquet").exists() and (DATA_DIR / "snapshot_manifest.json").exists():
        h = manifest()["cutoff_height"]     # the fetch may run past the cutoff; the study window stops there
        con.execute(f"CREATE VIEW bs AS SELECT * FROM '{WORK_DIR / 'blockstats.parquet'}' WHERE height <= {h}")
    return con


def df(con, sql: str):
    """Run a query and return a pandas DataFrame (DuckDB 2.0: relations keep no result)."""
    return con.execute(sql).df()


def write_json(name: str, obj) -> Path:
    path = DATA_DIR / name
    path.write_text(json.dumps(obj, indent=1, default=_default) + "\n", encoding="utf-8")
    return path


def read_json(name: str):
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(type(o))
