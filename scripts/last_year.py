#!/usr/bin/env python3
"""
Last-year portfolio value vs S&P 500.

Reads:
- data/computed/combined.csv             (current holdings by lot; includes CASH rows)
- data/history/{SYMBOL}.csv              (EOD OHLCV time series)

Computes for each trading day in the last ~1 year:
- sum(cash + stocks) for symbols held at that time (include if no purchase date)
- sum(cash) only
- sum(stocks) only

Notes/assumptions:
- combined.csv rows may have blank Trade Date. If blank → treat as position existing for the entire chart window.
- Quantity may repeat across accounts; we aggregate per symbol across all accounts.
- CASH rows are treated as a single symbol "CASH" with price fixed at 1.0.
- Uses Close prices.
- S&P 500 reference uses SPY from data/history/SPY.csv and is normalized to start at the same value as total on the first chart day, plotted on a right y-axis as a % series.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("data")
COMPUTED_DIR = DATA_DIR / "computed"
HISTORY_DIR = DATA_DIR / "history"
OUTPUT_DIR = DATA_DIR / "output"

COMBINED_CSV = COMPUTED_DIR / "combined.csv"
OUTPUT_PATH = OUTPUT_DIR / "last_year.png"


@dataclass
class Lot:
    portfolio: str
    symbol: str
    trade_date: date | None
    quantity: float


def _parse_trade_date(value: str | None) -> date | None:
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    # The file sometimes encodes yyyymmdd. Could also be empty/floaty.
    # Try yyyymmdd first, then ISO split fallback.
    try:
        # Some inputs look like 20250627.0 → remove decimals
        if "." in s:
            s = s.split(".")[0]
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").date()
    except Exception:
        pass
    # Try ISO date or date part before space
    try:
        s2 = s.split(" ")[0]
        return datetime.strptime(s2, "%Y-%m-%d").date()
    except Exception:
        return None


def read_combined_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_lots_from_combined(path: Path) -> Tuple[List[Lot], float]:
    lots: List[Lot] = []
    total_cash_quantity = 0.0  # CASH is quantity of dollars

    for row in read_combined_rows(path):
        symbol = (row.get("Symbol") or "").strip()
        portfolio = (row.get("Portfolio") or "").strip()
        if not symbol:
            continue
        # Quantity can be float-like string
        try:
            quantity = float(row.get("Quantity") or 0)
        except Exception:
            quantity = 0.0

        if symbol == "CASH":
            total_cash_quantity += quantity
            continue

        trade_date = _parse_trade_date(row.get("Trade Date"))
        if quantity == 0:
            continue
        lots.append(Lot(portfolio=portfolio, symbol=symbol, trade_date=trade_date, quantity=quantity))

    return lots, total_cash_quantity


def load_price_history_close(symbol: str) -> Tuple[List[date], np.ndarray]:
    path = HISTORY_DIR / f"{symbol}.csv"
    if not path.exists():
        return ([], np.array([]))
    dates: List[date] = []
    closes: List[float] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ds = row.get("Date") or ""
            try:
                dt = datetime.fromisoformat(ds.replace("Z", "+00:00"))
            except Exception:
                try:
                    dt = datetime.strptime(ds.split(" ")[0], "%Y-%m-%d")
                except Exception:
                    continue
            try:
                c = float(row.get("Close") or 0)
            except Exception:
                continue
            if c == 0:
                continue
            dates.append(dt.date())
            closes.append(c)
    if not dates:
        return ([], np.array([]))
    # For Python list of date objects, sort with Python sort
    idx = np.argsort(np.array([d.toordinal() for d in dates]))
    dates_sorted = [dates[i] for i in idx]
    closes_arr = np.array([closes[i] for i in idx], dtype=float)
    return (dates_sorted, closes_arr)


def build_trading_calendar(symbols: List[str], start_d: date, end_d: date) -> List[date]:
    # Use SPY as the canonical calendar if available, otherwise union of available dates.
    spy_dates, _ = load_price_history_close("SPY")
    if spy_dates:
        return [d for d in spy_dates if start_d <= d <= end_d]
    # Fallback: union of all dates across symbols
    all_dates: set[date] = set()
    for sym in symbols:
        ds, _ = load_price_history_close(sym)
        for d in ds:
            if start_d <= d <= end_d:
                all_dates.add(d)
    return sorted(all_dates)


def compute_time_series(lots: List[Lot], cash_balance: float, dates: List[date]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Aggregate lots by symbol with earliest trade date per lot; we will include lot quantity only after its trade_date.
    # If trade_date is None, include for all dates.
    # For each date: stocks_value = sum_over_symbols( price(symbol, date) * sum(quantities active on date) )
    symbols = sorted({lot.symbol for lot in lots})

    # Preload price histories
    history: Dict[str, Tuple[List[date], np.ndarray]] = {}
    for sym in symbols:
        history[sym] = load_price_history_close(sym)

    date_to_index_cache: Dict[str, Dict[date, int]] = {}
    for sym in symbols:
        ds, _ = history[sym]
        date_to_index_cache[sym] = {d: i for i, d in enumerate(ds)}

    stocks_series = np.zeros(len(dates), dtype=float)
    cash_series = np.full(len(dates), cash_balance, dtype=float)

    # For efficiency, compute per-symbol active quantity vector across dates
    for sym in symbols:
        ds, closes = history[sym]
        if len(ds) == 0:
            continue
        # Build active quantity per date for this symbol
        active_qty = np.zeros(len(dates), dtype=float)
        for lot in lots:
            if lot.symbol != sym:
                continue
            # Determine from which chart index this lot is active
            if lot.trade_date is None:
                start_index = 0
            else:
                # active on and after trade_date
                # Map to first date >= trade_date on our calendar
                start_index = 0
                for i, d in enumerate(dates):
                    if d >= lot.trade_date:
                        start_index = i
                        break
                    start_index = i + 1
            if start_index < len(dates):
                active_qty[start_index:] += lot.quantity

        # Map close prices to our calendar using last available close up to that date (forward-unavailable → use exact match only)
        # We will use exact-date match; if date missing for symbol, treat price as NaN and contribute 0 for that date.
        price_vec = np.full(len(dates), np.nan, dtype=float)
        date_to_idx = date_to_index_cache[sym]
        for i, d in enumerate(dates):
            j = date_to_idx.get(d)
            if j is not None:
                price_vec[i] = closes[j]

        # value = qty * price; missing price → 0
        contrib = np.nan_to_num(price_vec, nan=0.0) * active_qty
        stocks_series += contrib

    total_series = stocks_series + cash_series
    return total_series, cash_series, stocks_series


def load_spy_series_aligned(dates: List[date]) -> Tuple[np.ndarray, np.ndarray]:
    ds, closes = load_price_history_close("SPY")
    if not ds:
        return (np.array([]), np.array([]))
    idx_map = {d: i for i, d in enumerate(ds)}
    aligned = np.full(len(dates), np.nan, dtype=float)
    for i, d in enumerate(dates):
        j = idx_map.get(d)
        if j is not None:
            aligned[i] = closes[j]
    # Compute % change from first valid point
    if np.all(np.isnan(aligned)):
        return (np.array([]), np.array([]))
    non_nan_indices = np.where(~np.isnan(aligned))[0]
    if non_nan_indices.size == 0:
        return (np.array([]), np.array([]))
    base = aligned[non_nan_indices[0]]
    pct = (aligned / base) - 1.0
    return aligned, pct


def main() -> None:
    if not COMBINED_CSV.exists():
        raise FileNotFoundError(f"{COMBINED_CSV} not found")

    lots, cash_balance = load_lots_from_combined(COMBINED_CSV)

    # Build last-year date range using SPY calendar for consistency
    end_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=365)
    end_d = end_dt.date()
    start_d = start_dt.date()
    symbols = sorted({lot.symbol for lot in lots})
    calendar = build_trading_calendar(symbols, start_d, end_d)
    if not calendar:
        print("No trading dates available in the last year.")
        return

    total, cash, stocks = compute_time_series(lots, cash_balance, calendar)

    # SPY reference normalized to the same starting total
    _, spy_pct = load_spy_series_aligned(calendar)

    # Plot
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax_left = plt.subplots(figsize=(14, 6))
    ax_left.plot(calendar, total, label="Total (Cash+Stocks)", color="#1f77b4", linewidth=2)
    ax_left.plot(calendar, cash, label="Cash", color="#2ca02c", linewidth=1.5, alpha=0.8)
    ax_left.plot(calendar, stocks, label="Stocks", color="#ff7f0e", linewidth=1.5, alpha=0.9)
    ax_left.set_ylabel("Portfolio Value ($)")

    # Right axis for SPY % change, scaled to start at total[0]
    ax_right = ax_left.twinx()
    if spy_pct.size:
        # Convert % to a line that starts at total[0] but shows % on right axis
        base_total = total[0] if total.size and total[0] != 0 else 1.0
        spy_value_like = base_total * (1.0 + np.nan_to_num(spy_pct, nan=0.0))
        ax_left.plot(calendar, spy_value_like, label="S&P 500 (SPY, normalized)", color="#9467bd", linestyle="--", linewidth=1.5)
        ax_right.plot([], [])  # keep twin axis
        ax_right.set_ylabel("S&P 500 Change (%)")
        from matplotlib.ticker import PercentFormatter
        # Synchronize right axis to represent percent relative to base_total
        left_ymin, left_ymax = ax_left.get_ylim()
        ax_right.set_ylim((left_ymin / base_total) - 1.0, (left_ymax / base_total) - 1.0)
        ax_right.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    ax_left.set_title("Last Year: Portfolio Value vs S&P 500")
    ax_left.grid(True, alpha=0.3)
    lines, labels = ax_left.get_legend_handles_labels()
    ax_left.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()


