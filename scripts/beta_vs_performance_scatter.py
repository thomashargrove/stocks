#!/usr/bin/env python3
"""
Script to generate a scatter plot of Beta vs 1-year relative performance to SPY.

Reads data/stock_data/combined.csv and creates a scatter plot showing the relationship
between Beta and Perf_Rel_SPY_12m for all unique symbols.
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# Use non-interactive backend to prevent plot from displaying
matplotlib.use('Agg')


def load_stock_data(csv_path: Path) -> pd.DataFrame:
    """Load stock data from CSV file.
    
    Args:
        csv_path: Path to the combined.csv file
        
    Returns:
        DataFrame containing the stock data
    """
    return pd.read_csv(csv_path)


def prepare_data_for_plotting(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for scatter plot by filtering and cleaning.
    
    Args:
        df: Raw stock data DataFrame
        
    Returns:
        DataFrame with unique symbols and clean Beta/Performance data
    """
    # Filter out CASH entries and rows with missing data
    plot_data = df[
        (df['Symbol'] != 'CASH') & 
        (df['Beta'].notna()) & 
        (df['Perf_Rel_SPY_12m'].notna()) &
        (df['Beta'] != 0.0)  # Remove zero betas which are likely cash equivalents
    ].copy()
    
    # Get unique symbols with their data (take first occurrence)
    unique_symbols = plot_data.groupby('Symbol').first().reset_index()
    
    return unique_symbols


def create_scatter_plot(df: pd.DataFrame) -> None:
    """Create and display scatter plot of Beta vs Performance.
    
    Args:
        df: DataFrame with Beta and Perf_Rel_SPY_12m columns
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot
    scatter = ax.scatter(
        df['Beta'], 
        df['Perf_Rel_SPY_12m'],
        alpha=0.7,
        s=60,
        c='blue',
        edgecolors='black',
        linewidth=0.5
    )
    
    # Add labels for each point
    for idx, row in df.iterrows():
        ax.annotate(
            row['Symbol'], 
            (row['Beta'], row['Perf_Rel_SPY_12m']),
            xytext=(5, 5), 
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    # Add reference lines
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='SPY Performance (1.0)')
    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Market Beta (1.0)')
    
    # Customize the plot
    ax.set_xlabel('Beta', fontsize=12, fontweight='bold')
    ax.set_ylabel('1-Year Performance Relative to SPY', fontsize=12, fontweight='bold')
    ax.set_title('Stock Beta vs 1-Year Relative Performance to SPY', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add some statistics
    correlation = df['Beta'].corr(df['Perf_Rel_SPY_12m'])
    ax.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
    output_path = output_dir / "beta_vs_performance_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Number of unique symbols: {len(df)}")
    print(f"Beta range: {df['Beta'].min():.3f} to {df['Beta'].max():.3f}")
    print(f"Performance range: {df['Perf_Rel_SPY_12m'].min():.3f} to {df['Perf_Rel_SPY_12m'].max():.3f}")
    print(f"Correlation between Beta and Performance: {correlation:.3f}")


def main() -> None:
    """Main function to execute the scatter plot generation."""
    # Define paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    csv_path = project_root / "data" / "computed" / "combined.csv"
    
    # Check if file exists
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    # Load and prepare data
    print(f"Loading data from {csv_path}")
    raw_data = load_stock_data(csv_path)
    plot_data = prepare_data_for_plotting(raw_data)
    
    # Create scatter plot
    create_scatter_plot(plot_data)


if __name__ == "__main__":
    main()