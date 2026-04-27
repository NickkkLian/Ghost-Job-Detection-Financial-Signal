"""
Ghost Intensity — Concentration Comparison
===========================================
Runs the walk-forward backtest at multiple portfolio concentration
levels (default: 5%, 10%, 25%) and produces a side-by-side
comparison of returns, risk, and Sharpe ratio.

Imports ghost_backtest_v2.py — both files must be in the same directory.

USAGE (command line)
────────────────────
    python ghost_concentration_compare.py \\
        --ghost_file  ghost_scores_fq.parquet \\
        --start_date  2019-01-01 \\
        --concentrations 0.05 0.10 0.25

USAGE (from your notebook)
──────────────────────────
    from ghost_concentration_compare import run_comparison
    results = run_comparison(ghost_file='ghost_scores_fq.parquet',
                             concentrations=[0.05, 0.10, 0.25])
"""

import argparse
import importlib
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Import the v2 backtester engine ──────────────────────────────────────────
# Force a fresh reload every time this script runs to avoid Colab/Jupyter
# returning a stale cached version of ghost_backtest_v2 that may be missing
# strategies added in later edits.
try:
    import ghost_backtest_v2 as _gbv2
    importlib.reload(_gbv2)          # ← always re-reads the .py file
    from ghost_backtest_v2 import (
        DEFAULTS,
        load_ghost_scores,
        download_prices,
        run_backtest,
        performance_metrics,
        STRATEGY_LABELS,
        PALETTE,
    )
except ModuleNotFoundError:
    sys.exit(
        "ERROR: ghost_backtest_v2.py not found in the same directory.\n"
        "Both files must be in the same folder."
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONCENTRATION LEVELS TO TEST
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CONCENTRATIONS = [0.05, 0.10, 0.25]

# Strategies we care about for the concentration comparison
# (omit benchmark — it's identical at every concentration)
FOCUS_STRATEGIES = [
    "Short High Ghost",
    "Long High Ghost",
    "Long Low Ghost",
    "Long / Short",
    "Ghost Momentum",
    "Threshold (is_ghost)",
    "Ghost Barbell (Long Both)",
    "Universe (Equal Weight)",
]

# ─────────────────────────────────────────────────────────────────────────────
# LINE STYLES PER CONCENTRATION
# ─────────────────────────────────────────────────────────────────────────────

CONC_STYLE = {
    0.05: {"lw": 2.2, "dash": (1, 0),  "label": "5%"},
    0.10: {"lw": 2.0, "dash": (5, 2),  "label": "10%"},
    0.25: {"lw": 1.6, "dash": (2, 2),  "label": "25%"},
}

def _style(conc):
    """Return line style dict for a given concentration, generating one if not pre-set."""
    if conc in CONC_STYLE:
        return CONC_STYLE[conc]
    return {"lw": 1.6, "dash": (3, 1, 1, 1), "label": f"{conc*100:.0f}%"}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARISON RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(
    ghost_file:           str   = DEFAULTS["ghost_file"],
    start_date:           str   = DEFAULTS["start_date"],
    end_date:             str   = DEFAULTS["end_date"],
    concentrations:       list  = DEFAULT_CONCENTRATIONS,
    lag_days:             int   = DEFAULTS["lag_days"],
    rebal_freq:           str   = DEFAULTS["rebal_freq"],
    tc_bps:               float = DEFAULTS["tc_bps"],
    benchmark_ticker:     str   = DEFAULTS["benchmark_ticker"],
    max_daily_ret:        float = DEFAULTS["max_daily_ret"],
    vol_filter_threshold: float = DEFAULTS["vol_filter_threshold"],
    min_adv_usd:          float = DEFAULTS["min_adv_usd"],
    identity_check:       bool  = DEFAULTS["identity_check"],
    output_dir:           str   = "backtest_output",
) -> dict[float, pd.DataFrame]:
    """
    Run the walk-forward backtest at each concentration level.
    Prices are downloaded ONCE and reused across all runs.

    Returns
    -------
    results : dict  {concentration (float) → nav DataFrame}
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    concentrations = sorted(concentrations)

    print("\n" + "═" * 62)
    print("  GHOST INTENSITY — CONCENTRATION COMPARISON")
    print("═" * 62)
    print(f"  Concentrations    : {[f'{c*100:.0f}%' for c in concentrations]}")
    print(f"  Ghost file        : {ghost_file}")
    print(f"  Date range        : {start_date}  →  {end_date or 'today'}")
    print(f"  Publication lag   : {lag_days} days")
    print(f"  Rebalance freq    : {rebal_freq}")
    print(f"  Transaction cost  : {tc_bps} bps one-way")
    print(f"  Return winsor cap : ±{max_daily_ret*100:.0f}%/day")
    print(f"  Vol filter        : drop if ann. vol > {vol_filter_threshold*100:.0f}%")
    print(f"  Min ADV           : ${min_adv_usd/1e6:.1f}M USD")
    print("═" * 62)

    # ── Step 1: load ghost scores once ───────────────────────────────────
    print("\n[1 / 3]  Loading ghost scores ...")
    ghost_df = load_ghost_scores(ghost_file, lag_days)
    tickers  = ghost_df["ticker"].unique().tolist()

    ghost_company_map = None
    if identity_check and "company" in ghost_df.columns:
        ghost_company_map = ghost_df.groupby("ticker")["company"].first().to_dict()

    # ── Step 2: download prices once ─────────────────────────────────────
    print("\n[2 / 3]  Downloading prices + applying data-quality filters ...")
    prices = download_prices(
        tickers             = tickers,
        benchmark           = benchmark_ticker,
        start               = start_date,
        end                 = end_date or "2100-01-01",
        min_days            = DEFAULTS["min_price_history"],
        vol_filter          = vol_filter_threshold,
        min_adv_usd         = min_adv_usd,
        ghost_company_map   = ghost_company_map,
        identity_check      = identity_check,
        output_dir          = out_dir,
    )

    # ── Step 3: backtest at each concentration ────────────────────────────
    print(f"\n[3 / 3]  Running {len(concentrations)} backtests ...")
    results  = {}
    metrics_all = {}

    for conc in concentrations:
        label = f"{conc*100:.0f}%"
        print(f"\n  ── Concentration {label} ──")
        nav = run_backtest(
            ghost_df         = ghost_df,
            prices           = prices,
            start_date       = start_date,
            end_date         = end_date,
            lag_days         = lag_days,
            concentration    = conc,
            rebal_freq       = rebal_freq,
            tc_bps           = tc_bps,
            benchmark_ticker = benchmark_ticker,
            max_daily_ret    = max_daily_ret,
        )
        results[conc]      = nav
        metrics_all[conc]  = performance_metrics(nav)
        nav.to_csv(out_dir / f"nav_conc_{label}.csv")
        # Diagnostic: confirm exactly which strategy columns came back
        strat_cols = [c for c in nav.columns if c != f"Buy & Hold ({benchmark_ticker})"]
        print(f"  Strategies in nav  : {strat_cols}")

    # ── Print summary table ───────────────────────────────────────────────
    _print_comparison_table(metrics_all, concentrations)

    # ── Plot ──────────────────────────────────────────────────────────────
    plot_concentration_comparison(results, metrics_all, concentrations,
                                  benchmark_ticker, out_dir)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison_table(metrics_all: dict, concentrations: list) -> None:
    """Print a condensed comparison table across concentration levels."""
    key_metrics = ["Total Return", "CAGR", "Sharpe", "Max Drawdown", "Ann. Vol"]

    for strat in FOCUS_STRATEGIES + ["Buy & Hold (SPY)"]:
        rows = []
        for conc in concentrations:
            m = metrics_all[conc]
            if strat not in m.index:
                continue
            row = {"Concentration": f"{conc*100:.0f}%"}
            for k in key_metrics:
                row[k] = m.loc[strat, k]
            rows.append(row)

        if not rows:
            continue
        df = pd.DataFrame(rows).set_index("Concentration")
        div = "─" * 62
        print(f"\n{div}")
        print(f"  {strat}")
        print(div)
        print(df.to_string())

    print("\n" + "─" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_concentration_comparison(
    results:          dict,
    metrics_all:      dict,
    concentrations:   list,
    benchmark_ticker: str,
    out_dir:          Path,
) -> None:
    """
    Cumulative return panel per strategy + Sharpe/drawdown bar charts.

    Strategies to plot are discovered DIRECTLY from the nav DataFrames
    returned by run_backtest, not from the hardcoded FOCUS_STRATEGIES list.
    This prevents silent omissions when the module cache is stale or when
    a strategy name in FOCUS_STRATEGIES doesn't exactly match a nav column.
    """
    benchmark_col = f"Buy & Hold ({benchmark_ticker})"

    # ── Discover strategies from actual nav columns ───────────────────────
    # Union of all non-benchmark columns across every concentration run,
    # ordered by FOCUS_STRATEGIES preference where possible.
    seen       = set()
    plot_strats = []

    # First: add strategies that appear in FOCUS_STRATEGIES order
    for s in FOCUS_STRATEGIES:
        for nav in results.values():
            if s in nav.columns and s not in seen and s != benchmark_col:
                plot_strats.append(s)
                seen.add(s)
                break

    # Then: pick up any additional strategies not in FOCUS_STRATEGIES
    for nav in results.values():
        for col in nav.columns:
            if col != benchmark_col and col not in seen:
                plot_strats.append(col)
                seen.add(col)

    if not plot_strats:
        print("WARNING: no strategy columns found in nav DataFrames — nothing to plot.")
        return

    print(f"\n  Plotting strategies: {plot_strats}")

    n_strats = len(plot_strats)
    # Build a grid with enough cells for all strategies
    n_cols   = 3
    n_rows   = (n_strats + n_cols - 1) // n_cols   # ceiling division

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 6 * n_rows + 8))
    gs_top = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                               top=0.93, bottom=0.38,
                               hspace=0.45, wspace=0.30)
    gs_bot = gridspec.GridSpec(1, 2, figure=fig,
                               top=0.32, bottom=0.05,
                               wspace=0.30)

    strat_axes = [fig.add_subplot(gs_top[r, c])
                  for r in range(n_rows) for c in range(n_cols)]
    ax_sharpe  = fig.add_subplot(gs_bot[0, 0])
    ax_dd      = fig.add_subplot(gs_bot[0, 1])

    # Hide unused grid cells
    for idx in range(n_strats, len(strat_axes)):
        strat_axes[idx].set_visible(False)

    fig.suptitle(
        "Ghost Intensity — Strategy Performance by Concentration Level\n"
        "Walk-forward backtest, no lookahead bias",
        fontsize=14, fontweight="bold", y=0.97,
    )

    # ── Per-strategy cumulative return panels ─────────────────────────────
    for idx, strat in enumerate(plot_strats):
        ax    = strat_axes[idx]
        color = PALETTE.get(strat, "#444")

        # Plot one line per concentration level
        has_data = False
        for conc in concentrations:
            nav = results[conc]
            if strat not in nav.columns:
                print(f"  WARNING: '{strat}' missing from nav at concentration "
                      f"{conc*100:.0f}% — available: {list(nav.columns)}")
                continue
            st  = _style(conc)
            ret = (nav[strat] - 1) * 100
            ax.plot(nav.index, ret,
                    color=color,
                    linewidth=st["lw"],
                    linestyle=(0, st["dash"]),
                    label=st["label"],
                    alpha=0.9)
            has_data = True

        # SPY reference line
        for conc in concentrations:
            nav = results[conc]
            if benchmark_col in nav.columns:
                spy_ret = (nav[benchmark_col] - 1) * 100
                ax.plot(nav.index, spy_ret,
                        color="#888780", linewidth=1.2,
                        linestyle=(0, (3, 2)), alpha=0.5,
                        label="SPY" if conc == concentrations[0] else "_")
                break

        ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
        ax.set_title(strat, fontsize=11, color=color, fontweight="500", pad=6)
        ax.set_ylabel("Return (%)", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda y, _: f"{y:.0f}%"))
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.2)
        if has_data:
            ax.legend(fontsize=8, framealpha=0.7, title="Conc.", title_fontsize=8)

    # ── Sharpe ratio bar chart ────────────────────────────────────────────
    _bar_metric_chart(
        ax             = ax_sharpe,
        metrics_all    = metrics_all,
        concentrations = concentrations,
        strategies     = plot_strats + [benchmark_col],
        metric_col     = "Sharpe",
        title          = "Sharpe Ratio by Concentration",
        ylabel         = "Sharpe Ratio",
        ref_line       = 0,
    )

    # ── Max drawdown bar chart ────────────────────────────────────────────
    _bar_metric_chart(
        ax             = ax_dd,
        metrics_all    = metrics_all,
        concentrations = concentrations,
        strategies     = plot_strats + [benchmark_col],
        metric_col     = "Max Drawdown",
        title          = "Max Drawdown by Concentration",
        ylabel         = "Max Drawdown (%)",
        ref_line       = None,
    )

    out_path = out_dir / "concentration_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved  →  {out_path}")
    plt.show()


def _bar_metric_chart(ax, metrics_all, concentrations, strategies,
                      metric_col, title, ylabel, ref_line):
    """Grouped bar chart of one metric across strategies and concentrations."""
    n_strats    = len(strategies)
    n_conc      = len(concentrations)
    bar_w       = 0.65 / n_conc
    x           = np.arange(n_strats)

    for ci, conc in enumerate(concentrations):
        m      = metrics_all[conc]
        st     = _style(conc)
        values = []
        for strat in strategies:
            if strat not in m.index:
                values.append(np.nan)
                continue
            raw = m.loc[strat, metric_col]
            # Strip % signs and convert to float
            try:
                v = float(str(raw).replace("%", "").replace("+", ""))
            except ValueError:
                v = np.nan
            values.append(v)

        offsets = (ci - (n_conc - 1) / 2) * bar_w
        bars = ax.bar(x + offsets, values, width=bar_w * 0.9,
                      label=st["label"], alpha=0.85,
                      color=[PALETTE.get(s, "#888") for s in strategies])

    if ref_line is not None:
        ax.axhline(ref_line, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in strategies],
                       fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(title="Conc.", fontsize=8, title_fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    if metric_col == "Max Drawdown":
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda y, _: f"{y:.0f}%"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ghost Intensity Concentration Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ghost_file",     default=DEFAULTS["ghost_file"])
    parser.add_argument("--start_date",     default=DEFAULTS["start_date"])
    parser.add_argument("--end_date",       default=DEFAULTS["end_date"])
    parser.add_argument("--concentrations", nargs="+", type=float,
                        default=DEFAULT_CONCENTRATIONS,
                        help="List of concentration levels, e.g. 0.05 0.10 0.25")
    parser.add_argument("--lag_days",       type=int,   default=DEFAULTS["lag_days"])
    parser.add_argument("--rebal_freq",     default=DEFAULTS["rebal_freq"],
                        choices=["Q", "M", "W"])
    parser.add_argument("--tc_bps",         type=float, default=DEFAULTS["tc_bps"])
    parser.add_argument("--benchmark",      default=DEFAULTS["benchmark_ticker"])
    parser.add_argument("--max_daily_ret",  type=float, default=DEFAULTS["max_daily_ret"])
    parser.add_argument("--vol_filter",     type=float,
                        default=DEFAULTS["vol_filter_threshold"])
    parser.add_argument("--min_adv_usd",    type=float, default=DEFAULTS["min_adv_usd"])
    parser.add_argument("--no_identity_check", action="store_true")
    parser.add_argument("--output_dir",     default="backtest_output")

    args = parser.parse_args()

    run_comparison(
        ghost_file           = args.ghost_file,
        start_date           = args.start_date,
        end_date             = args.end_date,
        concentrations       = args.concentrations,
        lag_days             = args.lag_days,
        rebal_freq           = args.rebal_freq,
        tc_bps               = args.tc_bps,
        benchmark_ticker     = args.benchmark,
        max_daily_ret        = args.max_daily_ret,
        vol_filter_threshold = args.vol_filter,
        min_adv_usd          = args.min_adv_usd,
        identity_check       = not args.no_identity_check,
        output_dir           = args.output_dir,
    )

    sys.exit(0)
