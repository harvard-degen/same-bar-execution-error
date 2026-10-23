# Same-Bar Execution Error — Replication Package

[![DOI](https://zenodo.org/badge/1430428589.svg)](https://doi.org/10.5281/zenodo.23590933)

Replication code for **"Look-Ahead Bias Has No Deflator: Quantifying the Same-Bar Execution Error in Systematic Backtests"** (R. Chan, SSRN Working Paper No. 7654039, 2026).
Paper page: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7654039>

## What this shows

One line of execution-timing code — crediting a backtest position with returns that began *before* the decision that created it — changes a deliberately trivial Bitcoin momentum strategy's terminal wealth by a factor of **423** (14.41x → 6,094.94x), raises the Sharpe ratio from 1.01 to 2.81, and simultaneously *reduces* maximum drawdown (45.6% → 18.9%). On ETHUSDT the same error inflates terminal wealth **~4,697x**. The exact decomposition shows 100% of the gap arises at position-flip intervals: entry-day bars average +341 bps open-to-close versus +12 bps unconditionally.

The paper gives the timeline semantics (decision **W** / fill **F** / credit window **R**), a seven-arm taxonomy of when-bugs, and a four-test diagnostic gate — including a contemporaneous-correlation test an allocator can run on a manager's track record without re-running their engine.

## Reproduce

Requirements: **Python 3.10+, standard library only** (numpy optional). No API keys.

```bash
python3 w1_experiment_v4.py   # regenerates the results JSON: arms, grids, purged CV, gap decomposition
python3 make_fig1_v3.py       # rebuilds the figure; asserts every plotted arm matches the stored JSON
python3 w1_repro.py           # independent adversarial re-implementation; prints a JSON report to stdout
```

- `w1_results_v4.json` is the committed output of the 2026-10-19 run (BTC inflation 422.9x, ETH 4,697x).
- Data: Binance public API (BTCUSDT / ETHUSDT 1d klines, `data.binance.vision` monthly-zip fallback), fetched live at runtime. The window edge moves with the current date, so re-runs differ slightly at the boundary; the JSON in this repo is the frozen run the paper cites.
- The S&P 500 exhibit **silently skips** unless you place a FRED `fredgraph.csv`-format file at `sp500_fred.csv` (download SP500 daily from <https://fred.stlouisfed.org/series/SP500>).
- `make_fig1_v3.py` recomputes every plotted arm and asserts equality with the stored JSON before rendering — the figure cannot drift from the numbers.

## Contents

| File | Role |
|---|---|
| `w1_experiment_v4.py` | Main generator: arms A–G, flip decomposition, correlation diagnostics, parameter grids, purged CV, subperiod split |
| `w1_repro.py` | Independent re-implementation written for adversarial review; prints JSON to stdout, writes nothing |
| `make_fig1_v3.py` | Figure builder with equality assertions against the stored results |
| `w1_probe_d.py` | Diagnostic probe used during the audit trail |
| `w1_results_v4.json` | Committed results (the frozen run the paper cites) |
| `fig1_timeline_arms_v3.svg` | Figure 1 |

All scripts are pure standard library: `urllib`, `json`, `csv`, `zipfile`, `math`.

## Citation

```
Chan, Ronald (2026). "Look-Ahead Bias Has No Deflator: Quantifying the Same-Bar
Execution Error in Systematic Backtests." SSRN Working Paper No. 7654039.
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7654039 · DOI: 10.2139/ssrn.7654039
```

Code (this repository, v1.0.0): [DOI 10.5281/zenodo.23590934](https://doi.org/10.5281/zenodo.23590934) · concept DOI: 10.5281/zenodo.23590933

Author ORCID: [0009-0002-4400-9106](https://orcid.org/0009-0002-4400-9106)

*Apex Quant Capital is the systematic-trading research brand of Saprolings Pte Ltd, Singapore.*

## License

MIT — see `LICENSE`.
