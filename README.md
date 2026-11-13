# What Does It Cost to Force-Close a Lightning Channel? Evidence from 974,000 Channel Closes, 2018–2026

Saurabh Kumar, Bitcoin Data Labs

**Paper:** [arXiv:2611.16311](https://arxiv.org/abs/2611.16311) ([HTML](https://arxiv.org/html/2611.16311),
[PDF](https://arxiv.org/pdf/2611.16311)) · DOI [10.48550/arXiv.2611.16311](https://doi.org/10.48550/arXiv.2611.16311)
· [Bitcoin Data Labs research](https://bitcoindatalabs.org/research.html#ln-force-close-costs) ·
[plain-language guide](https://bitcoindatalabs.github.io/deep-dives/analyses/2026-11-lightning-force-closes/)
The third paper of a series, after
[*Does Client Diversity Protect the Lightning Network?*](https://github.com/bitcoindatalabs/ln-client-diversity)
(arXiv:2610.20952), whose client labels it uses, and
[*Who Patches the Lightning Network?*](https://github.com/bitcoindatalabs/ln-upgrade-dynamics) (arXiv:2610.30350).

Code, aggregate data and LaTeX source for the paper. arXiv categories: `cs.CR`, cross-list `cs.NI`.
Data cutoff: block 974,041 (31 October 2026).

## Summary
Bitcoin's Lightning payment network returns the funds of a channel that cannot close cooperatively through
pre-signed transactions whose fees and delays fall on its operators. That fee design is being replaced: Bitcoin Core
31.0 removed the mempool rule anchor channels relied on, and the Lightning specification adopted zero-fee commitments
in May 2026. The paper traces every public channel close and every force close found on-chain from January 2018 to
October 2026, 973,581 closes, and follows each output to its spend. Fee spikes did not bring a wave of force closes. A
force close costs 2–6 times a cooperative close in fees and locks the closer's balance for days, which outweighs the
fees only in low-fee years. When fees are high, HTLC timeouts confirm late, yet forwarding nodes lost few payments; the
only losses in the order a replacement-cycling attack leaves were a test one operator ran against its own node.
Anchor outputs did their job for HTLCs, not commitments, and became dust that third parties sweep and miners collect;
no third party sweeps simple-taproot anchors.

## Key findings

**Do Bitcoin fee spikes bring waves of Lightning force closes?** No. Force closes in a spike week and the next were
0.97 times the four weeks before, against 1.02 after ordinary weeks of the same years.

**What does a force close cost?** 2–6 times a cooperative close in fees, and it locks the closer's balance for days,
which outweighs the fees only in low-fee years.

**Do late HTLC timeouts lose payments?** Rarely. In 2023, 18.2% of HTLC timeouts confirmed more than 40 blocks after
expiry. Yet since 2021, forwarding nodes lost only 38 payments, each with the preimage already on-chain.

**What are anchor outputs used for?** Anchor outputs cut late timeouts in fee spikes (20.6% against 30.5%), yet owners
rarely bump commitments with them, and miners take most of what third parties sweep. Zero-fee commitments would save
only 12% of the anchor mechanism's block space.

## Repository structure
* `paper/`: LaTeX source (`main.tex`, `sections/`, `references.bib`, `figures/`, `tables/`, `numbers.tex`) and the
  compiled `main.pdf`.
* `src/00_*.py` … `src/09_*.py`: the pipeline stages; `src/fcc/`: input/output, read-only Bitcoin Core RPC and plot
  style helpers; `src/config.py`: the cutoff, the input locations and the protocol constants.
* `data/*.json`: the aggregate outputs behind every figure, table and in-text number.
* `scripts/package_arxiv.py`: builds the arXiv upload bundle.

## Reproducing the results

### 1. Dependencies
```bash
pip install -r requirements.txt
```

### 2. Input data
The pipeline reads four inputs, all derived from public data:
* the tables of an on-chain scan of Lightning channel closes and of the spends of their outputs (Parquet;
  `LN_ONCHAIN_DIR`);
* daily snapshots of the public Lightning gossip graph (`LN_PARQUET_DIR`);
* an index of the Bitcoin chain: blocks, transactions, inputs and outputs as Parquet (`CHAIN_LAKE_DIR`; funding
  transactions under `LN_FUNDING_DIR`);
* per-block fee statistics from a Bitcoin Core node (`getblockstats`; stages 00 and 04b read the node with
  `BITCOIN_RPC_HOST`, `BITCOIN_RPC_PORT`, `BITCOIN_RPC_USER`, `BITCOIN_RPC_PASSWORD`, or a `.env` file named by
  `BDL_ENV_FILE`).

The underlying blocks and gossip are public. Our scan tables, channel list and gossip snapshots are available from the
author on request.

### 3. Pipeline
Every stage truncates its inputs at the cutoff (`src/config.py`) and writes figures to `paper/figures/`, tables to
`paper/tables/` and numbers to `data/*.json`.
```bash
python src/00_blockstats.py --end 974041     # fee statistics per block (node or chain index), resumable
python src/01_snapshot.py                    # freeze the inputs at the cutoff, with checks
python src/02b_pre_anchor_2026.py --extract  # stages with --extract read the chain index and gossip first
python src/04d_breach_wave.py --extract
python src/09_lifecycle.py --extract
# 04c --extract runs in chunk groups; its docstring has the commands
for s in 02_census 02b_pre_anchor_2026 03_costs 04_anchors 04b_sweepers 04c_taproot_anchors 04d_breach_wave \
         05_fee_regimes 05b_fee_placebo 06_zero_fee 09_lifecycle 08_htlc_deadlines; do python src/$s.py; done
python src/07_numbers.py                     # last: every number quoted in the text -> paper/numbers.tex
```

### 4. Paper
```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main     # or: latexmk -pdf main.tex
cd .. && python scripts/package_arxiv.py                                        # -> dist/ln-force-close-costs-arxiv.tar.gz
```

## Data policy
Aggregates only: no list, ranking or label of nodes, sweeping operators, addresses or transactions is published, in
the paper or in `data/`.

## How to cite
Please cite the paper (GitHub's *Cite this repository* button gives the same entry from [`CITATION.cff`](./CITATION.cff)):
```bibtex
@misc{kumar2026forceclosecosts,
  title  = {What Does It Cost to Force-Close a Lightning Channel? Evidence from 974,000 Channel Closes, 2018--2026},
  author = {Saurabh Kumar},
  year   = {2026},
  url    = {https://github.com/bitcoindatalabs/ln-force-close-costs}
}
```
The arXiv identifier and DOI are added here on announcement.

## License
MIT (see `LICENSE`).
