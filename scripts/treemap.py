#!/usr/bin/env python3
"""
Generate a treemap visualization from stock portfolio data.

This script reads data/computed/combined.csv and creates a treemap where:
- Each cell represents a ticker symbol
- Cell size is based on the percentage of overall portfolio (PctOverall)
- Cell color is based on 1-year relative performance to SPY (Perf_Rel_SPY_12m)
  - Red: Underperforming SPY
  - Yellow: Neutral performance
  - Green: Outperforming SPY
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import squarify

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_portfolio_data(csv_path: str) -> pd.DataFrame:
    """Load and process the portfolio data from CSV file.
    
    Args:
        csv_path: Path to the combined.csv file
        
    Returns:
        DataFrame with processed portfolio data
    """
    df = pd.read_csv(csv_path)
    
    # Filter out rows with missing essential data
    df = df.dropna(subset=['Symbol', 'PctOverall', 'Perf_Rel_SPY_12m'])
    
    # Remove CASH entries as they don't have meaningful performance data
    df = df[df['Symbol'] != 'CASH']
    
    # Group by Symbol to aggregate data for multiple entries of same ticker
    aggregated = df.groupby('Symbol').agg({
        'PctOverall': 'sum',
        'Perf_Rel_SPY_12m': 'mean',  # Average performance if multiple entries
        'Last_Price': 'last'  # Take the last price
    }).reset_index()
    
    return aggregated


def create_treemap_data(df: pd.DataFrame) -> tuple[list[str], list[float], list[float]]:
    """Prepare data for treemap visualization.
    
    Args:
        df: Processed portfolio DataFrame
        
    Returns:
        Tuple of (symbols, sizes, performance_values)
    """
    # Sort by percentage to ensure largest holdings are processed first
    df_sorted = df.sort_values('PctOverall', ascending=False)
    
    symbols = df_sorted['Symbol'].tolist()
    sizes = df_sorted['PctOverall'].tolist()
    performance_values = df_sorted['Perf_Rel_SPY_12m'].tolist()
    
    return symbols, sizes, performance_values


def get_color_map(performance_values: list[float]) -> list[str]:
    """Generate color map based on relative performance to SPY.
    
    Args:
        performance_values: List of relative performance values (1.0 = SPY performance)
        
    Returns:
        List of colors in hex format
    """
    colors = []
    
    for perf in performance_values:
        if perf < 0.95:  # Significantly underperforming
            colors.append('#d62728')  # Red
        elif perf < 1.05:  # Close to SPY performance
            colors.append('#ff7f0e')  # Orange/Yellow
        else:  # Outperforming
            colors.append('#2ca02c')  # Green
    
    return colors


def create_treemap_visualization(
    symbols: list[str], 
    sizes: list[float], 
    performance_values: list[float],
    output_path: str
) -> None:
    """Create and save the treemap visualization.
    
    Args:
        symbols: List of ticker symbols
        sizes: List of portfolio percentages
        performance_values: List of relative performance values
        output_path: Path to save the output image
    """
    # Set up the plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    
    # Get colors based on performance
    colors = get_color_map(performance_values)
    
    # Create labels with ticker and performance percentage
    labels = [f"{symbol}\n{int((perf - 1) * 100):+d}%" for symbol, perf in zip(symbols, performance_values)]
    
    # Create the treemap
    squarify.plot(
        sizes=sizes,
        label=labels,
        color=colors,
        alpha=0.8,
        ax=ax,
        linewidth=2,
        edgecolor='black'
    )
    
    # Customize the plot
    ax.set_title(
        'Portfolio Treemap\nSize = Portfolio % | Color = 1-Year Performance vs SPY',
        fontsize=16,
        fontweight='bold',
        pad=20
    )
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#d62728', alpha=0.8, label='Underperforming SPY (<95%)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#ff7f0e', alpha=0.8, label='Neutral (95-105%)'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#2ca02c', alpha=0.8, label='Outperforming SPY (>105%)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Treemap saved to: {output_path}")
    
    # Close the plot to free memory
    plt.close()


def main() -> None:
    """Main function to generate the treemap visualization."""
    # Define paths
    data_dir = project_root / "data"
    csv_path = data_dir / "computed" / "combined.csv"
    output_dir = data_dir / "output"
    output_path = output_dir / "portfolio_treemap.png"
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Check if input file exists
    if not csv_path.exists():
        print(f"Error: Input file not found: {csv_path}")
        sys.exit(1)
    
    try:
        # Load and process data
        print("Loading portfolio data...")
        df = load_portfolio_data(str(csv_path))
        print(f"Loaded {len(df)} tickers")
        
        # Prepare treemap data
        symbols, sizes, performance_values = create_treemap_data(df)
        
        # Create visualization
        print("Generating treemap...")
        create_treemap_visualization(symbols, sizes, performance_values, str(output_path))
        
        print("Treemap generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating treemap: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
