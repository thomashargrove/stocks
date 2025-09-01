"""
Generate a heatmap of monthly gains/losses in total dollars for each stock over the previous 10 months.

This script reads stock data from data/computed/combined.csv and historical price data from 
data/history/{ticker}.csv files to calculate monthly performance and create a visualization.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


def load_stock_data(combined_csv_path: str) -> pd.DataFrame:
    """
    Load stock data from the combined CSV file.
    
    Args:
        combined_csv_path: Path to the combined.csv file
        
    Returns:
        DataFrame with stock information
    """
    df = pd.read_csv(combined_csv_path)
    # Filter out non-stock entries (CASH, ETFs if needed)
    stock_df = df[~df['Symbol'].isin(['CASH', 'SPY'])].copy()
    return stock_df


def load_price_history(ticker: str, history_dir: str) -> pd.DataFrame | None:
    """
    Load price history for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        history_dir: Directory containing historical price data
        
    Returns:
        DataFrame with price history or None if file doesn't exist
    """
    file_path = Path(history_dir) / f"{ticker}.csv"
    
    if not file_path.exists():
        print(f"Warning: No price history found for {ticker}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df = df.sort_values('Date')
        return df
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None


def calculate_monthly_performance(
    price_df: pd.DataFrame, 
    quantity: float, 
    months_back: int = 10
) -> Dict[str, float]:
    """
    Calculate monthly dollar gains/losses for a stock position.
    
    Args:
        price_df: DataFrame with Date and Close columns
        quantity: Number of shares held
        months_back: Number of months to look back
        
    Returns:
        Dictionary mapping month strings to dollar performance
    """
    if price_df is None or len(price_df) == 0:
        return {}
    
    # Get the end date (most recent data) and start date (12 months back for safety)
    end_date = price_df['Date'].max()
    start_date = end_date - timedelta(days=365)  # Get extra data to ensure we have full months
    
    # Filter to relevant date range
    df_filtered = price_df[price_df['Date'] >= start_date].copy()
    
    if len(df_filtered) == 0:
        return {}
    
    # Resample to monthly data (last business day of each month)
    df_filtered.set_index('Date', inplace=True)
    monthly_prices = df_filtered['Close'].resample('M').last()
    
    # Calculate monthly returns in dollars
    monthly_performance = {}
    
    # Get the last N months
    recent_months = monthly_prices.tail(months_back + 1)  # +1 to calculate returns
    
    for i in range(len(recent_months) - 1):
        start_price = recent_months.iloc[i]
        end_price = recent_months.iloc[i + 1]
        
        # Calculate dollar change for the position
        dollar_change = (end_price - start_price) * quantity
        
        # Create month label (YYYY-MM format)
        month_date = recent_months.index[i + 1]
        month_label = month_date.strftime('%Y-%m')
        
        monthly_performance[month_label] = dollar_change
    
    return monthly_performance


def calculate_overall_performance(monthly_perf: Dict[str, float]) -> float:
    """
    Calculate overall percentage performance across all months.
    
    Args:
        monthly_perf: Dictionary of monthly dollar performance
        
    Returns:
        Overall percentage performance
    """
    if not monthly_perf:
        return 0.0
    
    total_dollar_change = sum(monthly_perf.values())
    # Use absolute values to get a magnitude-based ranking
    return abs(total_dollar_change)


def create_heatmap_data(
    stocks_df: pd.DataFrame, 
    history_dir: str
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create the data matrix for the heatmap.
    
    Args:
        stocks_df: DataFrame with stock information
        history_dir: Directory with price history files
        
    Returns:
        Tuple of (heatmap_df, month_labels)
    """
    all_performance = {}
    all_months = set()
    
    print("Calculating monthly performance for each stock...")
    
    for _, row in stocks_df.iterrows():
        ticker = row['Symbol']
        quantity = row['Quantity']
        
        print(f"Processing {ticker}...")
        
        # Load price history
        price_df = load_price_history(ticker, history_dir)
        
        if price_df is not None:
            # Calculate monthly performance
            monthly_perf = calculate_monthly_performance(price_df, quantity)
            all_performance[ticker] = monthly_perf
            all_months.update(monthly_perf.keys())
    
    # Create sorted list of months (most recent first)
    month_labels = sorted(list(all_months), reverse=True)[:10]  # Last 10 months
    
    # Create matrix for heatmap
    heatmap_data = []
    stock_symbols = []
    overall_performance = []
    
    for ticker, monthly_perf in all_performance.items():
        if monthly_perf:  # Only include stocks with data
            row_data = []
            for month in month_labels:
                row_data.append(monthly_perf.get(month, 0))
            
            heatmap_data.append(row_data)
            stock_symbols.append(ticker)
            overall_performance.append(calculate_overall_performance(monthly_perf))
    
    # Create DataFrame
    heatmap_df = pd.DataFrame(heatmap_data, columns=month_labels, index=stock_symbols)
    
    # Sort by overall performance (largest to smallest)
    performance_series = pd.Series(overall_performance, index=stock_symbols)
    sorted_indices = performance_series.sort_values(ascending=False).index
    heatmap_df = heatmap_df.loc[sorted_indices]
    
    return heatmap_df, month_labels


def create_heatmap_visualization(heatmap_df: pd.DataFrame, month_labels: List[str]) -> None:
    """
    Create and display the heatmap visualization.
    
    Args:
        heatmap_df: DataFrame with monthly performance data
        month_labels: List of month labels for columns
    """
    fig, ax = plt.subplots(figsize=(14, max(8, len(heatmap_df) * 0.6)))
    
    # Create custom colormap (red-yellow-green only)
    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9641']
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list('RdYlGn', colors, N=n_bins)
    
    # Get data values for color scaling
    vmin = heatmap_df.values.min()
    vmax = heatmap_df.values.max()
    
    # Create the heatmap manually
    im = ax.imshow(heatmap_df.values, cmap=cmap, aspect='auto', 
                   vmin=vmin, vmax=vmax)
    
    # Add text annotations
    for i in range(len(heatmap_df)):
        for j in range(len(month_labels)):
            value = heatmap_df.iloc[i, j]
            text = ax.text(j, i, f'${value:.0f}', 
                          ha='center', va='center', 
                          color='white' if abs(value) > (vmax - vmin) * 0.3 else 'black',
                          fontweight='bold', fontsize=9)
    
    # Set ticks and labels
    ax.set_xticks(range(len(month_labels)))
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(heatmap_df)))
    ax.set_yticklabels(heatmap_df.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Monthly Gain/Loss ($)', rotation=270, labelpad=20)
    
    # Add gridlines
    ax.set_xticks(np.arange(len(month_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(heatmap_df)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', size=0)
    
    plt.title('Monthly Stock Performance Heatmap (Dollar Gains/Losses)\nPast 10 Months', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Month (Most Recent → Oldest)', fontsize=12)
    plt.ylabel('Stocks (Ranked by Overall Performance)', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nHeatmap Summary:")
    print(f"Stocks analyzed: {len(heatmap_df)}")
    print(f"Months covered: {len(month_labels)}")
    print(f"Date range: {month_labels[-1]} to {month_labels[0]}")
    
    # Show top and bottom performers
    total_performance = heatmap_df.sum(axis=1)
    print(f"\nTop 3 performers (total $):")
    for i, (stock, perf) in enumerate(total_performance.head(3).items(), 1):
        print(f"  {i}. {stock}: ${perf:,.0f}")
    
    print(f"\nBottom 3 performers (total $):")
    for i, (stock, perf) in enumerate(total_performance.tail(3).items(), 1):
        print(f"  {i}. {stock}: ${perf:,.0f}")


def main() -> None:
    """Main function to run the monthly performance analysis."""
    
    # File paths
    combined_csv = "data/computed/combined.csv"
    history_dir = "data/history"
    
    # Check if files exist
    if not Path(combined_csv).exists():
        print(f"Error: {combined_csv} not found")
        return
    
    if not Path(history_dir).exists():
        print(f"Error: {history_dir} directory not found")
        return
    
    print("Loading stock data...")
    stocks_df = load_stock_data(combined_csv)
    print(f"Found {len(stocks_df)} stocks to analyze")
    
    print("Creating heatmap data...")
    heatmap_df, month_labels = create_heatmap_data(stocks_df, history_dir)
    
    if heatmap_df.empty:
        print("No data available for heatmap generation")
        return
    
    print("Creating visualization...")
    create_heatmap_visualization(heatmap_df, month_labels)


if __name__ == "__main__":
    main()