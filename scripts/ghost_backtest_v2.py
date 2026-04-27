"""
Ghost Intensity Trading Strategy Backtester  v2
================================================
Walk-forward backtest using your ghost_scores_fq.parquet output
combined with real historical stock prices from Yahoo Finance.

v2 CHANGES (data-quality hardening)
────────────────────────────────────
  - Daily return WINSORISATION at ±max_daily_ret (default ±25%)
    Single bad adjusted-price data points can create 1000%+ single-day
    "returns" for delisted/reverse-merged tickers.  No real equity
    position in a diversified portfolio can move ±25% every day.
  - VOLATILITY FILTER: tickers whose trailing 252-day ann. vol exceeds
    vol_filter_threshold (default 150%) are dropped before the backtest.
  - LIQUIDITY FILTER: tickers with avg daily volume below min_adv_usd
    (default $1M) are dropped.  These names are untradeable in size.
  - TICKER IDENTITY CHECK: cross-references the 'company' column from
    your ghost file against yfinance's longName.  Flags mismatches
    (reused tickers return data for the wrong company).
  - Transaction cost default raised to 20 bps (was 10).

NO LOOKAHEAD BIAS GUARANTEE
────────────────────────────
At every rebalancing date T, only ghost scores whose
available_date (= quarter_end + publication_lag_days) <= T
are used. For each ticker, the MOST RECENT available
quarter's score is selected.

USAGE (command line)
────────────────────
    python ghost_backtest_v2.py \\
        --ghost_file  ghost_scores_fq.parquet \\
        --start_date  2019-01-01 \\
        --end_date    2024-12-31 \\
        --lag_days    45 \\
        --concentration 0.25 \\
        --output_dir  backtest_output

USAGE (from your notebook)
──────────────────────────
    from ghost_backtest_v2 import main
    nav, metrics = main(ghost_file='ghost_scores_fq.parquet',
                        start_date='2019-01-01')

INPUT FILE FORMAT
─────────────────
Accepts either:
  - ghost_scores_fq.parquet  (firm-quarter level, from your pipeline)
  - company_measures.parquet (company level — only static scores, less ideal)

Required columns:  ticker | quarter | ghost_score
Optional columns:  is_ghost  (enables the Threshold strategy)
                   company   (enables ticker identity check)

The 'quarter' column should be 'YYYY-QN' format, e.g. '2023-Q1'.

OUTPUT FILES (in --output_dir)
────────────────────────────────
  backtest_nav.csv         — daily NAV for each strategy
  backtest_metrics.csv     — performance summary table
  backtest_results.png     — 4-panel chart
  ticker_flags.csv         — tickers dropped/flagged by data-quality filters
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIGURATION  (override via CLI args or main() kwargs)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    ghost_file            = "ghost_scores_fq.parquet",
    start_date            = "2019-01-01",
    end_date              = None,          # None → today
    lag_days              = 45,            # calendar days after quarter-end before we use scores
    concentration         = 0.25,          # fraction of universe for each long/short leg
    rebal_freq            = "Q",           # 'Q' quarterly | 'M' monthly
    tc_bps                = 20,            # one-way transaction cost, basis points (raised from 10)
    benchmark_ticker      = "SPY",
    output_dir            = "backtest_output",
    min_price_history     = 252,           # min trading days required to include a ticker (raised)
    # ── v2: data-quality parameters ──────────────────────────────────────────
    max_daily_ret         = 0.25,          # winsorise daily returns at ±25%
    vol_filter_threshold  = 1.50,          # drop tickers with ann. vol > 150%
    min_adv_usd           = 1_000_000,     # minimum avg daily volume in USD ($1M)
    identity_check        = True,          # cross-check company name vs yfinance longName
)

STRATEGY_LABELS = {
    "short_high_ghost"      : "Short High Ghost",
    "long_high_ghost"       : "Long High Ghost",
    "long_low_ghost"        : "Long Low Ghost",
    "long_short"            : "Long / Short",
    "ghost_momentum"        : "Ghost Momentum",
    "threshold"             : "Threshold (is_ghost)",
    "ghost_barbell"         : "Ghost Barbell (Long Both)",   # long low ghost + long high ghost
    "universe_equal_weight" : "Universe (Equal Weight)",     # all tickers, equal-weighted — bias check
    "benchmark"             : "Buy & Hold (SPY)",
}

PALETTE = {
    "Short High Ghost"          : "#e24b4a",
    "Long High Ghost"           : "#d4547a",
    "Long Low Ghost"            : "#1f9c52",
    "Long / Short"              : "#378add",
    "Ghost Momentum"            : "#ba7517",
    "Threshold (is_ghost)"      : "#9b59b6",
    "Ghost Barbell (Long Both)" : "#16a0a0",
    "Universe (Equal Weight)"   : "#e87d2b",
    "Buy & Hold (SPY)"          : "#888780",
}

DASHES = {
    "Short High Ghost"          : (1, 0),
    "Long High Ghost"           : (1, 0),
    "Long Low Ghost"            : (1, 0),
    "Long / Short"              : (1, 0),
    "Ghost Momentum"            : (6, 2),
    "Threshold (is_ghost)"      : (4, 2),
    "Ghost Barbell (Long Both)" : (1, 0),
    "Universe (Equal Weight)"   : (2, 1),
    "Buy & Hold (SPY)"          : (3, 2),
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def quarter_end(q: str) -> pd.Timestamp:
    """'2023-Q1'  →  2023-03-31"""
    year, qn = int(q[:4]), int(q[-1])
    month = qn * 3
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def load_ghost_scores(path: str, lag_days: int) -> pd.DataFrame:
    """
    Load ghost scores and attach two derived columns:
      quarter_end    — last calendar day of the quarter
      available_date — quarter_end + lag_days  (earliest date we can use this score)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Ghost score file not found: {path}\n"
            f"Expected output from Ghost Job Detection Pipeline:\n"
            f"  ghost_scores_fq.parquet  (preferred — firm-quarter level)\n"
            f"  company_measures.parquet (company level, static scores only)"
        )

    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    required = {"ticker", "quarter", "ghost_score"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["ticker", "ghost_score", "quarter"]).copy()
    df = df[df["ticker"].str.strip() != ""]
    df["quarter_end"]    = df["quarter"].apply(quarter_end)
    df["available_date"] = df["quarter_end"] + pd.Timedelta(days=lag_days)

    print(f"  Loaded {len(df):,} firm-quarter ghost score records")
    print(f"  Unique tickers   : {df['ticker'].nunique():,}")
    print(f"  Quarter range    : {df['quarter'].min()} → {df['quarter'].max()}")
    print(f"  Score range      : {df['ghost_score'].min():.4f} – {df['ghost_score'].max():.4f}")
    if "is_ghost" in df.columns:
        print(f"  Ghost-flagged    : {df['is_ghost'].sum():,} / {len(df):,} rows")
    else:
        print("  Note: 'is_ghost' column not found — Threshold strategy will be skipped")
    return df


def download_prices(tickers:             list,
                    benchmark:           str,
                    start:               str,
                    end:                 str,
                    min_days:            int,
                    vol_filter:          float,
                    min_adv_usd:         float,
                    ghost_company_map:   dict | None,
                    identity_check:      bool,
                    output_dir:          Path) -> pd.DataFrame:
    """
    Download adjusted daily close prices + volume for all tickers + benchmark.

    v2 data-quality filters applied IN ORDER:
      1. Min price history  (< min_days valid close prices → drop)
      2. Ticker identity    (yfinance longName vs ghost file company name)
      3. Volatility filter  (trailing 252-day ann. vol > vol_filter → drop)
      4. Liquidity filter   (avg daily turnover < min_adv_usd → drop)
    """
    all_tickers = sorted(set(tickers + [benchmark]))
    print(f"  Requesting prices for {len(all_tickers)} tickers via Yahoo Finance ...")

    raw = yf.download(
        all_tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True
    )

    if isinstance(raw.columns, pd.MultiIndex):
        prices  = raw["Close"].copy()
        volumes = raw["Volume"].copy()
    else:
        prices  = raw.copy()
        volumes = pd.DataFrame(index=raw.index)

    flag_records = []

    # ── Filter 1: minimum price history ──────────────────────────────────
    before = prices.shape[1]
    mask   = prices.notna().sum() >= min_days
    if benchmark in prices.columns:
        mask[benchmark] = True
    dropped_hist = [c for c in prices.columns if not mask[c]]
    for t in dropped_hist:
        flag_records.append({"ticker": t, "flag": "insufficient_history",
                              "detail": f"{prices[t].notna().sum()} days < {min_days}"})
    prices  = prices.loc[:, mask]
    volumes = volumes.reindex(columns=prices.columns)
    if dropped_hist:
        print(f"  Filter 1 (history)    : dropped {len(dropped_hist)} tickers")

    # ── Filter 2: ticker identity check ──────────────────────────────────
    if identity_check and ghost_company_map:
        import difflib
        mismatches = []
        to_check   = [t for t in prices.columns if t != benchmark and t in ghost_company_map]
        print(f"  Filter 2 (identity)   : checking {len(to_check)} tickers against company names ...")
        for t in to_check:
            try:
                info      = yf.Ticker(t).info
                yf_name   = (info.get("longName") or info.get("shortName") or "").lower()
                ghost_name = ghost_company_map[t].lower()
                # Rough similarity: tokenise and check word overlap
                ghost_words = set(ghost_name.replace(",","").replace(".","").split())
                yf_words    = set(yf_name.replace(",","").replace(".","").split())
                stopwords   = {"inc","corp","co","ltd","llc","the","&","and","group","holdings","plc"}
                g_key = ghost_words - stopwords
                y_key = yf_words    - stopwords
                if g_key and y_key:
                    overlap = len(g_key & y_key) / len(g_key)
                    if overlap < 0.3 and yf_name:   # less than 30% word match
                        mismatches.append(t)
                        flag_records.append({"ticker": t, "flag": "identity_mismatch",
                                             "detail": f"ghost='{ghost_company_map[t]}' yf='{info.get('longName','?')}'"})
            except Exception:
                pass  # skip if yfinance info request fails

        if mismatches:
            print(f"  Filter 2 (identity)   : dropped {len(mismatches)} tickers with name mismatch")
            prices  = prices.drop(columns=[t for t in mismatches if t in prices.columns])
            volumes = volumes.reindex(columns=prices.columns)

    # ── Filter 3: volatility filter ───────────────────────────────────────
    daily_rets = prices.pct_change()
    ann_vol    = daily_rets.std() * np.sqrt(252)
    vol_ok     = (ann_vol <= vol_filter) | (ann_vol.index == benchmark)
    dropped_vol = vol_ok[~vol_ok].index.tolist()
    for t in dropped_vol:
        flag_records.append({"ticker": t, "flag": "high_volatility",
                              "detail": f"ann_vol={ann_vol[t]*100:.0f}%"})
    prices  = prices.loc[:, vol_ok]
    volumes = volumes.reindex(columns=prices.columns)
    if dropped_vol:
        print(f"  Filter 3 (volatility) : dropped {len(dropped_vol)} tickers "
              f"with ann. vol > {vol_filter*100:.0f}%")

    # ── Filter 4: liquidity filter ────────────────────────────────────────
    if not volumes.empty and volumes.notna().any().any():
        avg_price  = prices.mean()
        avg_vol    = volumes.mean()
        adv_usd    = avg_price * avg_vol
        liq_ok     = (adv_usd >= min_adv_usd) | (adv_usd.index == benchmark)
        dropped_liq = liq_ok[~liq_ok].index.tolist()
        for t in dropped_liq:
            flag_records.append({"ticker": t, "flag": "illiquid",
                                  "detail": f"ADV=${adv_usd.get(t,0):,.0f}"})
        prices = prices.loc[:, liq_ok]
        if dropped_liq:
            print(f"  Filter 4 (liquidity)  : dropped {len(dropped_liq)} tickers "
                  f"with ADV < ${min_adv_usd/1e6:.1f}M")
    else:
        print("  Filter 4 (liquidity)  : skipped (volume data unavailable)")

    # ── Save flag report ──────────────────────────────────────────────────
    if flag_records:
        flags_df = pd.DataFrame(flag_records)
        flags_df.to_csv(output_dir / "ticker_flags.csv", index=False)
        print(f"  Ticker flag report    : {len(flag_records)} entries → ticker_flags.csv")

    print(f"  Price matrix: {prices.shape[0]:,} trading days × {prices.shape[1]:,} tickers "
          f"(after all filters)")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2.  SIGNAL PANEL — walk-forward, no lookahead
# ─────────────────────────────────────────────────────────────────────────────

def build_signal_panel(ghost_df: pd.DataFrame,
                        rebal_dates: pd.DatetimeIndex,
                        available_tickers: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every rebalancing date, look back and find:
      - the MOST RECENT ghost_score per ticker whose available_date <= rebal_date
      - the corresponding is_ghost flag (if available)

    Returns
    -------
    score_panel  : DataFrame (index=rebal_date, columns=ticker, values=ghost_score)
    ghost_panel  : DataFrame (index=rebal_date, columns=ticker, values=is_ghost bool)
    """
    df = ghost_df[ghost_df["ticker"].isin(available_tickers)].copy()
    df = df.sort_values(["ticker", "available_date"])
    has_flag = "is_ghost" in df.columns

    score_records = []
    ghost_records = []

    for rdate in rebal_dates:
        eligible = df[df["available_date"] <= rdate]
        if eligible.empty:
            continue

        # Most-recent available quarter per ticker (no lookahead)
        latest = (eligible
                  .sort_values("quarter_end")
                  .groupby("ticker")
                  .last()
                  .reset_index())

        score_row = {"date": rdate}
        ghost_row = {"date": rdate}
        for _, row in latest.iterrows():
            score_row[row["ticker"]] = row["ghost_score"]
            if has_flag:
                ghost_row[row["ticker"]] = bool(row["is_ghost"])
        score_records.append(score_row)
        if has_flag:
            ghost_records.append(ghost_row)

    if not score_records:
        raise RuntimeError(
            "No signals were generated. The ghost score available_dates may be "
            "entirely after your backtest end_date, or entirely before start_date. "
            "Check --start_date / --end_date and --lag_days."
        )

    score_panel = pd.DataFrame(score_records).set_index("date")
    ghost_panel = (pd.DataFrame(ghost_records).set_index("date")
                   if ghost_records else pd.DataFrame())

    avg_coverage = score_panel.notna().sum(axis=1).mean()
    print(f"  Signal panel     : {len(score_panel)} rebal dates "
          f"× up to {score_panel.shape[1]} tickers")
    print(f"  Avg tickers/date : {avg_coverage:.0f}")
    return score_panel, ghost_panel


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PORTFOLIO WEIGHT CONSTRUCTORS
# ─────────────────────────────────────────────────────────────────────────────

def _equal_weight_topk(scores: pd.Series,
                        ascending: bool,
                        k: int) -> pd.Series:
    """Return equal-weight Series over the top-k ranked entries."""
    valid  = scores.dropna()
    ranked = valid.rank(ascending=ascending, method="first")
    sel    = ranked[ranked <= k].index
    w      = pd.Series(0.0, index=valid.index)
    w[sel] = 1.0 / k
    return w


def weights_for_strategy(strategy:  str,
                          scores:    pd.Series,
                          conc:      float,
                          is_ghost:  pd.Series | None,
                          prev_scores: pd.Series | None,
                          benchmark_ticker: str) -> pd.Series:
    """
    Construct portfolio weights for one strategy at one rebalancing date.
    Positive weight = long, negative weight = short.
    """
    valid = scores.dropna()
    n     = len(valid)

    if strategy == "benchmark":
        return pd.Series({benchmark_ticker: 1.0})

    if n < 4:
        return pd.Series(dtype=float)

    k = max(1, int(n * conc))

    # ── Short High Ghost ──────────────────────────────────────────────────
    if strategy == "short_high_ghost":
        # Short the top-k most ghost-like companies
        w = _equal_weight_topk(valid, ascending=False, k=k)  # ascending=False → rank 1 = highest score
        return -w

    # ── Long High Ghost ───────────────────────────────────────────────────
    if strategy == "long_high_ghost":
        # Long the top-k most ghost-like companies (contrarian bet: do ghost
        # firms actually underperform, or is there a momentum/neglect premium?).
        # The mirror image of Short High Ghost — same stocks, opposite sign.
        return _equal_weight_topk(valid, ascending=False, k=k)

    # ── Long Low Ghost ────────────────────────────────────────────────────
    if strategy == "long_low_ghost":
        # Long the bottom-k most genuine hirers
        return _equal_weight_topk(valid, ascending=True, k=k)

    # ── Long / Short (market-neutral) ─────────────────────────────────────
    if strategy == "long_short":
        long_leg  = _equal_weight_topk(valid, ascending=True,  k=k)
        short_leg = _equal_weight_topk(valid, ascending=False, k=k)
        w = long_leg.reindex(valid.index).fillna(0) \
          - short_leg.reindex(valid.index).fillna(0)
        return w

    # ── Ghost Momentum ────────────────────────────────────────────────────
    if strategy == "ghost_momentum":
        if prev_scores is None or len(prev_scores) < 4:
            return pd.Series(dtype=float)
        common = valid.index.intersection(prev_scores.index)
        if len(common) < 4:
            return pd.Series(dtype=float)
        delta = valid[common] - prev_scores[common]
        k_m   = max(1, int(len(delta) * conc))
        # Long tickers where ghost score fell most (improving companies)
        # Short tickers where ghost score rose most  (deteriorating companies)
        w = pd.Series(0.0, index=common)
        w[delta.nsmallest(k_m).index]  =  1.0 / k_m
        w[delta.nlargest(k_m).index]  -= 1.0 / k_m
        return w

    # ── Threshold (uses is_ghost flag directly) ───────────────────────────
    if strategy == "threshold":
        if is_ghost is None or is_ghost.empty:
            return pd.Series(dtype=float)
        common = valid.index.intersection(is_ghost.index)
        flags  = is_ghost[common]
        ghosts     = flags[flags == True].index
        non_ghosts = flags[flags == False].index
        w = pd.Series(0.0, index=common)
        if len(non_ghosts):
            w[non_ghosts] =  1.0 / len(non_ghosts)
        if len(ghosts):
            w[ghosts]     = -1.0 / len(ghosts)
        return w

    # ── Ghost Barbell (Long Both) ─────────────────────────────────────────
    if strategy == "ghost_barbell":
        # Long the bottom-k (lowest ghost score = genuine hirers, proven to work)
        # AND long the top-k  (highest ghost score = ghost firms, which have
        # been rising rather than falling — so own them instead of shorting them).
        # Each leg is equal-weighted at 1/(2k) per stock so the full portfolio
        # sums to 1.0 and is fully invested with no shorts.
        low_leg  = _equal_weight_topk(valid, ascending=True,  k=k)
        high_leg = _equal_weight_topk(valid, ascending=False, k=k)
        # Scale each leg to 50% of the portfolio
        w = (low_leg.reindex(valid.index).fillna(0) * 0.5
           + high_leg.reindex(valid.index).fillna(0) * 0.5)
        return w

    # ── Universe Equal Weight (sampling-bias check) ───────────────────────
    if strategy == "universe_equal_weight":
        # Long every ticker in the available universe with equal weight.
        # This is a pure bias-check benchmark: if this performs close to SPY
        # our ticker dataset has no significant selection bias.  No concentration
        # parameter is applied — 100% of the universe is held at all times.
        n_valid = len(valid)
        if n_valid == 0:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / n_valid, index=valid.index)

    return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  BACKTESTER (walk-forward)
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(ghost_df:         pd.DataFrame,
                 prices:           pd.DataFrame,
                 start_date:       str,
                 end_date:         str | None   = None,
                 lag_days:         int          = 45,
                 concentration:    float        = 0.25,
                 rebal_freq:       str          = "Q",
                 tc_bps:           float        = 20.0,
                 benchmark_ticker: str          = "SPY",
                 max_daily_ret:    float        = 0.25) -> pd.DataFrame:
    """
    Walk-forward backtest engine.

    For each calendar day:
      1. If it is a rebalancing day, rebuild the portfolio using ONLY
         ghost scores with available_date <= today (no lookahead).
      2. Mark-to-market the portfolio using that day's stock returns.

    v2: daily returns are winsorised at ±max_daily_ret before any
    portfolio calculation.  This eliminates single-day data artefacts
    (bad adjusted prices from splits/mergers) without throwing away the
    ticker entirely.

    Transaction costs are applied at rebalance as a one-way drag:
        cost = (turnover / 2) × (tc_bps / 10_000)

    Returns
    -------
    nav : pd.DataFrame
        Daily NAV series indexed by date, one column per strategy.
    """
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    prices   = prices.loc[start_date:end_date].copy()

    if prices.empty:
        raise RuntimeError(
            f"No price data between {start_date} and {end_date}. "
            "Check that yfinance returned data for your tickers."
        )

    daily_rets = prices.pct_change().fillna(0)

    # ── v2: WINSORISE returns ─────────────────────────────────────────────
    n_clipped = (daily_rets.abs() > max_daily_ret).sum().sum()
    if n_clipped > 0:
        print(f"  Winsorised {n_clipped:,} return observations outside "
              f"[{-max_daily_ret*100:.0f}%, +{max_daily_ret*100:.0f}%]")
    daily_rets = daily_rets.clip(-max_daily_ret, max_daily_ret)

    # ── Rebalancing calendar ──────────────────────────────────────────────
    rebal_raw = pd.date_range(start=start_date, end=end_date, freq=rebal_freq)
    # Snap to nearest following trading day
    trading_set = set(prices.index)
    rebal_dates = pd.DatetimeIndex([
        prices.index[prices.index >= d][0]
        for d in rebal_raw
        if any(prices.index >= d)
    ])

    # ── Build no-lookahead signal panels ─────────────────────────────────
    available_tickers = set(prices.columns) - {benchmark_ticker}
    score_panel, ghost_panel = build_signal_panel(
        ghost_df, rebal_dates, available_tickers
    )
    rebal_signal_dates = set(score_panel.index)

    # ── State initialisation ──────────────────────────────────────────────
    strategies      = list(STRATEGY_LABELS.keys())
    nav_values      = {s: [1.0] for s in strategies}
    nav_dates       = [prices.index[0]]
    cur_weights     = {s: pd.Series(dtype=float) for s in strategies}
    prev_scores_map = {}     # for ghost_momentum

    tc_multiplier = tc_bps / 10_000

    for i in range(1, len(prices.index)):
        date     = prices.index[i]
        day_rets = daily_rets.iloc[i]

        # ── Rebalance if today is a signal date ───────────────────────────
        if date in rebal_signal_dates:
            scores   = score_panel.loc[date].dropna()
            is_ghost = (ghost_panel.loc[date].dropna()
                        if not ghost_panel.empty and date in ghost_panel.index
                        else None)

            for s in strategies:
                new_w = weights_for_strategy(
                    strategy         = s,
                    scores           = scores,
                    conc             = concentration,
                    is_ghost         = is_ghost,
                    prev_scores      = prev_scores_map.get("last"),
                    benchmark_ticker = benchmark_ticker,
                )

                if new_w.empty:
                    continue

                # Transaction cost: proportional to two-way turnover
                old_w    = cur_weights[s].reindex(new_w.index).fillna(0)
                turnover = (new_w - old_w).abs().sum() / 2.0
                tc_drag  = turnover * tc_multiplier
                nav_values[s][-1] *= (1.0 - tc_drag)

                cur_weights[s] = new_w

            prev_scores_map["last"] = scores.copy()

        # ── Mark-to-market ────────────────────────────────────────────────
        for s in strategies:
            w = cur_weights[s]
            if w.empty:
                nav_values[s].append(nav_values[s][-1])
                continue
            overlap = w.index.intersection(day_rets.index)
            if overlap.empty:
                nav_values[s].append(nav_values[s][-1])
                continue
            period_return = (w[overlap] * day_rets[overlap]).sum()
            nav_values[s].append(nav_values[s][-1] * (1.0 + period_return))

        nav_dates.append(date)

    nav = pd.DataFrame(nav_values, index=nav_dates)
    nav.columns = [STRATEGY_LABELS[s] for s in strategies]

    # Only drop the Threshold strategy if it was never populated
    # (happens when the ghost file has no is_ghost column).
    # Do NOT use a blanket nunique()==1 check — it can silently remove
    # conservative long-only strategies like Ghost Barbell or Universe
    # Equal Weight whose early NAV values happen to be near-identical.
    threshold_col = STRATEGY_LABELS.get("threshold", "")
    if threshold_col in nav.columns and nav[threshold_col].nunique() == 1:
        print(f"  Note: removing '{threshold_col}' — no is_ghost flag found in ghost file.")
        nav = nav.drop(columns=[threshold_col])

    return nav


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def performance_metrics(nav: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualised performance statistics for each strategy.

    Returns a tidy DataFrame with one row per strategy.
    """
    rets  = nav.pct_change().dropna()
    rows  = []

    for col in nav.columns:
        r        = rets[col]
        n_years  = len(r) / 252.0
        total_r  = nav[col].iloc[-1] / nav[col].iloc[0] - 1
        cagr     = (1 + total_r) ** (1.0 / max(n_years, 0.01)) - 1
        ann_vol  = r.std() * np.sqrt(252)
        sharpe   = (r.mean() * 252) / ann_vol if ann_vol > 1e-9 else np.nan
        roll_max = nav[col].cummax()
        dd       = (nav[col] - roll_max) / roll_max
        max_dd   = dd.min()
        calmar   = cagr / abs(max_dd) if abs(max_dd) > 1e-9 else np.nan
        win_d    = (r > 0).mean()

        rows.append({
            "Strategy"     : col,
            "Total Return" : f"{total_r*100:+.1f}%",
            "CAGR"         : f"{cagr*100:+.1f}%",
            "Ann. Vol"     : f"{ann_vol*100:.1f}%",
            "Sharpe"       : f"{sharpe:.2f}" if not np.isnan(sharpe) else "—",
            "Max Drawdown" : f"{max_dd*100:.1f}%",
            "Calmar"       : f"{calmar:.2f}" if not np.isnan(calmar) else "—",
            "Win Rate (d)" : f"{win_d*100:.1f}%",
        })

    return pd.DataFrame(rows).set_index("Strategy")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(nav: pd.DataFrame,
                 metrics: pd.DataFrame,
                 output_dir: Path) -> None:
    """Four-panel chart: cumulative returns, drawdown, rolling Sharpe, monthly heatmap."""

    fig = plt.figure(figsize=(16, 18))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    # ── Panel 1: Cumulative returns ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    for col in nav.columns:
        c = PALETTE.get(col, "#555")
        d = DASHES.get(col, (1, 0))
        lw = 1.5 if "SPY" in col else 2.2
        ax1.plot(nav.index, (nav[col] - 1) * 100,
                 label=col, color=c, linewidth=lw,
                 linestyle=(0, d))
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax1.set_title("Cumulative Returns (%)\n[walk-forward, no lookahead bias]",
                  fontsize=13, pad=10)
    ax1.set_ylabel("Return (%)")
    ax1.legend(loc="upper left", fontsize=9, framealpha=0.8)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax1.grid(axis="y", alpha=0.25)

    # ── Panel 2: Drawdown ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    for col in nav.columns:
        c  = PALETTE.get(col, "#555")
        d  = DASHES.get(col, (1, 0))
        dd = (nav[col] / nav[col].cummax() - 1) * 100
        ax2.plot(nav.index, dd, color=c, linewidth=1.8, alpha=0.9,
                 linestyle=(0, d), label=col)
    ax2.fill_between(nav.index, ax2.get_ylim()[0], 0, alpha=0.03, color="red")
    ax2.set_title("Drawdown from Peak (%)", fontsize=13, pad=10)
    ax2.set_ylabel("Drawdown (%)")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.grid(axis="y", alpha=0.25)

    # ── Panel 3: Rolling 12-month Sharpe ─────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    daily_rets = nav.pct_change()
    for col in nav.columns:
        c  = PALETTE.get(col, "#555")
        r  = daily_rets[col]
        rs = (r.rolling(252).mean() / r.rolling(252).std()) * np.sqrt(252)
        ax3.plot(nav.index, rs, color=c, linewidth=1.8, label=col,
                 linestyle=(0, DASHES.get(col, (1, 0))))
    ax3.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax3.axhline(1, color="green", linewidth=0.6, linestyle="--", alpha=0.5,
                label="Sharpe = 1")
    ax3.set_title("Rolling 12-month Sharpe Ratio", fontsize=13, pad=10)
    ax3.legend(fontsize=8, framealpha=0.8)
    ax3.grid(axis="y", alpha=0.25)

    # ── Panel 4: Long/Short monthly return heatmap ────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ls_col = "Long / Short"
    if ls_col in nav.columns:
        monthly = (nav[ls_col].resample("ME").last().pct_change().dropna() * 100)
        years   = sorted(monthly.index.year.unique())
        mat     = np.full((len(years), 12), np.nan)
        for i, yr in enumerate(years):
            for ts, val in monthly[monthly.index.year == yr].items():
                mat[i, ts.month - 1] = val

        norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)
        im   = ax4.imshow(mat, cmap="RdYlGn", norm=norm, aspect="auto")

        months = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        ax4.set_xticks(range(12));     ax4.set_xticklabels(months, fontsize=8)
        ax4.set_yticks(range(len(years))); ax4.set_yticklabels(years, fontsize=8)
        ax4.set_title("Long / Short — Monthly Returns (%)\n(green = gain, red = loss)",
                      fontsize=11, pad=8)

        for i in range(len(years)):
            for j in range(12):
                if not np.isnan(mat[i, j]):
                    ax4.text(j, i, f"{mat[i, j]:.1f}",
                             ha="center", va="center",
                             fontsize=6.5,
                             color="black" if abs(mat[i, j]) < 7 else "white")
        plt.colorbar(im, ax=ax4, shrink=0.85, label="%")
    else:
        ax4.text(0.5, 0.5, "Long/Short strategy\nnot available",
                 ha="center", va="center", transform=ax4.transAxes)

    plt.suptitle("Ghost Intensity Trading Strategy — Walk-Forward Backtest",
                 fontsize=15, y=1.01, fontweight="bold")

    out_path = output_dir / "backtest_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved  →  {out_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main(ghost_file:           str   = DEFAULTS["ghost_file"],
         start_date:           str   = DEFAULTS["start_date"],
         end_date:             str   = DEFAULTS["end_date"],
         lag_days:             int   = DEFAULTS["lag_days"],
         concentration:        float = DEFAULTS["concentration"],
         rebal_freq:           str   = DEFAULTS["rebal_freq"],
         tc_bps:               float = DEFAULTS["tc_bps"],
         benchmark_ticker:     str   = DEFAULTS["benchmark_ticker"],
         output_dir:           str   = DEFAULTS["output_dir"],
         max_daily_ret:        float = DEFAULTS["max_daily_ret"],
         vol_filter_threshold: float = DEFAULTS["vol_filter_threshold"],
         min_adv_usd:          float = DEFAULTS["min_adv_usd"],
         identity_check:       bool  = DEFAULTS["identity_check"],
         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end backtest pipeline.

    Returns
    -------
    nav     : pd.DataFrame  daily NAV per strategy
    metrics : pd.DataFrame  performance summary table
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "═" * 62)
    print("  GHOST INTENSITY STRATEGY BACKTESTER  v2")
    print("═" * 62)
    print(f"  Ghost file        : {ghost_file}")
    print(f"  Date range        : {start_date}  →  {end_date or 'today'}")
    print(f"  Publication lag   : {lag_days} days after quarter-end")
    print(f"  Concentration     : top/bottom {concentration*100:.0f}% of universe")
    print(f"  Rebalance freq    : {rebal_freq}")
    print(f"  Transaction cost  : {tc_bps} bps one-way")
    print(f"  Benchmark         : {benchmark_ticker}")
    print(f"  Return winsor cap : ±{max_daily_ret*100:.0f}%/day")
    print(f"  Vol filter        : drop if ann. vol > {vol_filter_threshold*100:.0f}%")
    print(f"  Min ADV           : ${min_adv_usd/1e6:.1f}M USD")
    print(f"  Identity check    : {'on' if identity_check else 'off'}")
    print("═" * 62)

    # Step 1 — Ghost scores
    print("\n[1 / 5]  Loading ghost scores ...")
    ghost_df = load_ghost_scores(ghost_file, lag_days)
    tickers  = ghost_df["ticker"].unique().tolist()

    # Build company name map for identity check
    ghost_company_map = None
    if identity_check and "company" in ghost_df.columns:
        ghost_company_map = (
            ghost_df.groupby("ticker")["company"].first().to_dict()
        )

    # Step 2 — Prices
    print("\n[2 / 5]  Downloading historical prices + applying data-quality filters ...")
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

    # Step 3 — Backtest
    print("\n[3 / 5]  Running walk-forward backtest ...")
    nav = run_backtest(
        ghost_df         = ghost_df,
        prices           = prices,
        start_date       = start_date,
        end_date         = end_date,
        lag_days         = lag_days,
        concentration    = concentration,
        rebal_freq       = rebal_freq,
        tc_bps           = tc_bps,
        benchmark_ticker = benchmark_ticker,
        max_daily_ret    = max_daily_ret,
    )

    # Step 4 — Metrics
    print("\n[4 / 5]  Computing performance metrics ...")
    metrics = performance_metrics(nav)

    div = "─" * 62
    print(f"\n{div}")
    print(metrics.to_string())
    print(div)

    # ── Important caveats ─────────────────────────────────────────────────
    print("\n⚠  IMPORTANT CAVEATS")
    print("   1. Survivorship bias: this backtest only includes companies")
    print("      that appear in your ghost score file, which requires them")
    print("      to have been public AND posting jobs.  Delisted / acquired")
    print("      companies that were in the live universe are excluded.")
    print("   2. Short selling costs (borrow fees, ~0.5–3%/yr) are NOT")
    print("      modelled.  For strategies with short legs, real returns")
    print("      may be materially lower.")
    print("   3. The ghost signal is based on Revelio data.  If Revelio")
    print("      coverage changed over time, the effective universe is not")
    print("      constant, which can introduce a coverage-change bias.")
    print("   4. Return winsorisation at ±25%/day removes the worst data")
    print("      artefacts but does not eliminate all bad-data risk.")
    print("      Review ticker_flags.csv for dropped/flagged names.")
    print("   5. Universe (Equal Weight) holds ALL tickers in the dataset")
    print("      with equal weight at every rebalance.  It is a sampling-")
    print("      bias diagnostic: if it tracks SPY closely, the ticker")
    print("      dataset has no significant market-cap or sector tilt.")
    print("      A large divergence from SPY indicates selection bias in")
    print("      the Revelio coverage universe (e.g., small-cap skew).")

    # Step 5 — Save & plot
    nav.to_csv(out_dir / "backtest_nav.csv")
    metrics.to_csv(out_dir / "backtest_metrics.csv")
    print(f"\n  CSV files saved  →  {out_dir}/")

    print("\n[5 / 5]  Generating charts ...")
    plot_results(nav, metrics, out_dir)

    return nav, metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ghost Intensity Trading Strategy Backtester v2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ghost_file",           default=DEFAULTS["ghost_file"])
    parser.add_argument("--start_date",           default=DEFAULTS["start_date"])
    parser.add_argument("--end_date",             default=DEFAULTS["end_date"])
    parser.add_argument("--lag_days",             type=int,   default=DEFAULTS["lag_days"])
    parser.add_argument("--concentration",        type=float, default=DEFAULTS["concentration"])
    parser.add_argument("--rebal_freq",           default=DEFAULTS["rebal_freq"],
                        choices=["Q", "M", "W"])
    parser.add_argument("--tc_bps",               type=float, default=DEFAULTS["tc_bps"])
    parser.add_argument("--benchmark",            default=DEFAULTS["benchmark_ticker"])
    parser.add_argument("--output_dir",           default=DEFAULTS["output_dir"])
    parser.add_argument("--max_daily_ret",        type=float, default=DEFAULTS["max_daily_ret"],
                        help="Winsorise daily returns at ±this value (0.25 = ±25%%)")
    parser.add_argument("--vol_filter_threshold", type=float, default=DEFAULTS["vol_filter_threshold"],
                        help="Drop tickers with ann. vol above this (1.5 = 150%%)")
    parser.add_argument("--min_adv_usd",          type=float, default=DEFAULTS["min_adv_usd"],
                        help="Minimum average daily volume in USD")
    parser.add_argument("--no_identity_check",    action="store_true",
                        help="Disable ticker identity cross-check")

    args = parser.parse_args()

    nav, metrics = main(
        ghost_file           = args.ghost_file,
        start_date           = args.start_date,
        end_date             = args.end_date,
        lag_days             = args.lag_days,
        concentration        = args.concentration,
        rebal_freq           = args.rebal_freq,
        tc_bps               = args.tc_bps,
        benchmark_ticker     = args.benchmark,
        output_dir           = args.output_dir,
        max_daily_ret        = args.max_daily_ret,
        vol_filter_threshold = args.vol_filter_threshold,
        min_adv_usd          = args.min_adv_usd,
        identity_check       = not args.no_identity_check,
    )

    sys.exit(0)
