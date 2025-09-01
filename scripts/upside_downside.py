#!/usr/bin/env python3
"""
Calculate upside capture, downside capture, and spread ratios for stocks.

This script analyzes 1-year daily price history to compute:
- Upside capture: How much the stock gains relative to the market during positive market days
- Downside capture: How much the stock loses relative to the market during negative market days  
- Spread ratio: The difference between upside and downside capture ratios

Results are displayed as a visual heatmap with green-yellow-red coloring.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_combined_data(file_path: str) -> pd.DataFrame:
    """Load the combined portfolio data containing stock symbols and holdings."""
    return pd.read_csv(file_path)


def load_stock_history(symbol: str, data_dir: str) -> pd.DataFrame | None:
    """Load 1-year daily price history for a stock symbol."""
    file_path = Path(data_dir) / f"{symbol}.csv"
    if not file_path.exists():
        return None
    
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # Get last 1 year of data
    end_date = df['Date'].max()
    start_date = end_date - timedelta(days=365)
    df = df[df['Date'] >= start_date].copy()
    
    # Calculate daily returns
    df = df.sort_values('Date')
    df['daily_return'] = df['Close'].pct_change()
    
    return df


def calculate_capture_ratios(stock_df: pd.DataFrame, market_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate upside capture, downside capture, and spread ratios.
    
    Args:
        stock_df: DataFrame with stock price history and daily returns
        market_df: DataFrame with market (SPY) price history and daily returns
        
    Returns:
        Dictionary containing upside_capture, downside_capture, and spread_ratio
    """
    # Merge on date to align returns
    merged = pd.merge(
        stock_df[['Date', 'daily_return']].rename(columns={'daily_return': 'stock_return'}),
        market_df[['Date', 'daily_return']].rename(columns={'daily_return': 'market_return'}),
        on='Date',
        how='inner'
    )
    
    # Remove NaN values
    merged = merged.dropna()
    
    if len(merged) == 0:
        return {'upside_capture': 0.0, 'downside_capture': 0.0, 'spread_ratio': 0.0}
    
    # Separate positive and negative market days
    up_days = merged[merged['market_return'] > 0].copy()
    down_days = merged[merged['market_return'] < 0].copy()
    
    # Calculate capture ratios
    upside_capture = 0.0
    downside_capture = 0.0
    
    if len(up_days) > 0 and up_days['market_return'].sum() != 0:
        upside_capture = up_days['stock_return'].sum() / up_days['market_return'].sum()
    
    if len(down_days) > 0 and down_days['market_return'].sum() != 0:
        downside_capture = down_days['stock_return'].sum() / down_days['market_return'].sum()
    
    spread_ratio = upside_capture - downside_capture
    
    return {
        'upside_capture': upside_capture,
        'downside_capture': downside_capture,
        'spread_ratio': spread_ratio
    }


def get_stock_holdings_value(combined_df: pd.DataFrame, symbol: str) -> float:
    """Get the total holdings value for a stock across all portfolios."""
    stock_rows = combined_df[combined_df['Symbol'] == symbol]
    if len(stock_rows) == 0:
        return 0.0
    
    # Calculate total value: Last_Price * Quantity
    total_value = (stock_rows['Last_Price'] * stock_rows['Quantity']).sum()
    return total_value


def create_heatmap(results: list[dict[str, Any]]) -> None:
    """Create a visual heatmap of upside and downside capture ratios."""
    if not results:
        print("No data to visualize", file=sys.stderr)
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(results)
    df = df.sort_values('Holdings_Value', ascending=False)  # Top to bottom by largest holdings
    
    # Create data matrix for heatmap (stocks x [upside, downside])
    stocks = df['Symbol'].tolist()
    upside_values = df['Upside_Capture'].tolist()
    downside_values = df['Downside_Capture'].tolist()
    
    # Create matrix: each row is a stock, columns are [upside, downside]
    data_matrix = np.array([upside_values, downside_values]).T
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, len(stocks) * 0.6 + 2))
    
    # Create heatmap using matplotlib's imshow
    # Use RdYlGn_r colormap (Red-Yellow-Green reversed, so green=low, red=high)
    im = ax.imshow(data_matrix, cmap='RdYlGn_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Upside Capture', 'Downside Capture'])
    ax.set_yticks(range(len(stocks)))
    ax.set_yticklabels(stocks)
    
    # Add text annotations with values
    for i in range(len(stocks)):
        for j in range(2):
            value = data_matrix[i, j]
            text_color = 'white' if abs(value - 1.0) > 0.3 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                   fontweight='bold', color=text_color)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Capture Ratio (1.0 = matches market)', rotation=270, labelpad=20)
    
    # Add holdings value as secondary information
    for i, (stock, holdings) in enumerate(zip(stocks, df['Holdings_Value'].tolist())):
        ax.text(-0.3, i, f'${holdings:,.0f}', ha='right', va='center', 
               fontsize=9, color='gray')
    
    # Set title and labels
    ax.set_title('Stock Capture Ratios vs Market (SPY)\nOrdered by Holdings Value', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Capture Type', fontsize=12)
    ax.set_ylabel('Stock Symbol', fontsize=12)
    
    # Add subtitle with explanation
    fig.text(0.5, 0.02, 'Green = Low capture, Yellow = Moderate, Red = High capture\n'
                       'Holdings values shown on left', 
             ha='center', fontsize=10, style='italic')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.2)
    
    # Show the plot
    plt.show()


def main() -> None:
    """Main function to calculate and visualize capture ratios."""
    # File paths
    combined_file = "data/computed/combined.csv"
    history_dir = "data/history"
    
    # Load combined data
    try:
        combined_df = load_combined_data(combined_file)
    except Exception as e:
        print(f"Error loading combined data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get unique stocks (excluding CASH and ETFs like SPY)
    unique_stocks = combined_df[
        (combined_df['Symbol'] != 'CASH') & 
        (combined_df['Symbol'] != 'SPY')
    ]['Symbol'].unique()
    
    # Load market data (SPY)
    market_df = load_stock_history('SPY', history_dir)
    if market_df is None:
        print("Error: Could not load SPY market data", file=sys.stderr)
        sys.exit(1)
    
    results: list[dict[str, Any]] = []
    
    # Calculate capture ratios for each unique stock
    for symbol in unique_stocks:
        stock_df = load_stock_history(symbol, history_dir)
        if stock_df is None:
            print(f"Warning: Could not load data for {symbol}", file=sys.stderr)
            continue
        
        # Calculate capture ratios
        ratios = calculate_capture_ratios(stock_df, market_df)
        
        # Get holdings value
        holdings_value = get_stock_holdings_value(combined_df, symbol)
        
        results.append({
            'Symbol': symbol,
            'Holdings_Value': holdings_value,
            'Upside_Capture': ratios['upside_capture'],
            'Downside_Capture': ratios['downside_capture'],
            'Spread_Ratio': ratios['spread_ratio']
        })
    
    # Sort by holdings value (largest to smallest)
    results.sort(key=lambda x: x['Holdings_Value'], reverse=True)
    
    # Create and display heatmap
    create_heatmap(results)


if __name__ == "__main__":
    main()