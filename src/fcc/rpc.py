"""Minimal Bitcoin Core JSON-RPC client.

Credentials come from BITCOIN_RPC_HOST / _PORT / _USER / _PASSWORD in the environment, or from the dotenv file named
by BDL_ENV_FILE (default: .env in the working directory). Only read-only calls are used.
"""

import os
import time
from pathlib import Path

import requests

KEYS = ("BITCOIN_RPC_HOST", "BITCOIN_RPC_PORT", "BITCOIN_RPC_USER", "BITCOIN_RPC_PASSWORD")


def _credentials() -> dict:
    env = {k: os.environ.get(k) for k in KEYS}
    if all(env.values()):
        return env
    path = Path(os.environ.get("BDL_ENV_FILE", ".env"))
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() in KEYS and not env.get(k.strip()):
                env[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in KEYS if not env.get(k)]
    if missing:
        raise RuntimeError(f"RPC credentials missing: {missing} (set them or BDL_ENV_FILE)")
    return env


class Rpc:
    def __init__(self, timeout: int = 120):
        c = _credentials()
        self.url = f"http://{c['BITCOIN_RPC_HOST']}:{c['BITCOIN_RPC_PORT']}/"
        self.auth = (c["BITCOIN_RPC_USER"], c["BITCOIN_RPC_PASSWORD"])
        self.timeout = timeout
        self.s = requests.Session()

    def call(self, method: str, params=None, retries: int = 5):
        payload = {"jsonrpc": "1.0", "id": "paper03", "method": method, "params": params or []}
        for attempt in range(retries):
            try:
                r = self.s.post(self.url, json=payload, auth=self.auth, timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
                if body.get("error"):
                    raise RuntimeError(body["error"])
                return body["result"]
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
