#!/usr/bin/env python3
"""Portfolio performance visualization script.

Reads data/computed/combined.csv and creates a performance chart showing:
- Symbols grouped by portfolio and ordered by total percent of overall
- Relative performance arrows comparing different time periods
- SPY reference line for comparison
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def load_portfolio_data(csv_path: str) -> list[dict[str, Any]]:
    """Load portfolio data from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of portfolio data rows
    """
    data = []
    with open(csv_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert numeric fields
            numeric_fields = ['PctOverall', 'Perf_Rel_SPY_1m', 'Perf_Rel_SPY_3m', 'Perf_Rel_SPY_12m']
            for field in numeric_fields:
                if row[field]:
                    row[field] = float(row[field])
            data.append(row)
    return data


def group_symbols_by_portfolio(data: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group symbols across portfolios and calculate total percentages.
    
    Args:
        data: Portfolio data rows
        
    Returns:
        Dictionary mapping symbol to aggregated data
    """
    symbol_data = defaultdict(lambda: {
        'portfolios': [],
        'total_pct_overall': 0.0,
        'perf_1m': 0.0,
        'perf_3m': 0.0,
        'perf_12m': 0.0,
        'weight_sum': 0.0
    })
    
    for row in data:
        symbol = row['Symbol']
        portfolio = row['Portfolio']
        pct_overall = row['PctOverall'] or 0.0
        
        # Skip CASH entries for the visualization
        if symbol == 'CASH':
            continue
            
        symbol_data[symbol]['portfolios'].append(portfolio)
        symbol_data[symbol]['total_pct_overall'] += pct_overall
        
        # Weight the performance metrics by portfolio percentage
        weight = pct_overall
        symbol_data[symbol]['perf_1m'] += (row['Perf_Rel_SPY_1m'] or 1.0) * weight
        symbol_data[symbol]['perf_3m'] += (row['Perf_Rel_SPY_3m'] or 1.0) * weight
        symbol_data[symbol]['perf_12m'] += (row['Perf_Rel_SPY_12m'] or 1.0) * weight
        symbol_data[symbol]['weight_sum'] += weight
    
    # Calculate weighted averages for performance metrics
    for symbol, data_dict in symbol_data.items():
        if data_dict['weight_sum'] > 0:
            data_dict['perf_1m'] /= data_dict['weight_sum']
            data_dict['perf_3m'] /= data_dict['weight_sum']
            data_dict['perf_12m'] /= data_dict['weight_sum']
        
        # Remove duplicates and sort portfolios
        data_dict['portfolios'] = sorted(list(set(data_dict['portfolios'])))
    
    return dict(symbol_data)


def create_performance_chart(symbol_data: dict[str, dict[str, Any]], output_path: str) -> None:
    """Create the performance visualization chart.
    
    Args:
        symbol_data: Aggregated symbol data
        output_path: Path to save the output image
    """
    # Sort symbols by total percentage (descending)
    sorted_symbols = sorted(
        symbol_data.items(),
        key=lambda x: x[1]['total_pct_overall'],
        reverse=True
    )
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(sorted_symbols) * 0.6)))
    
    # SPY reference line (at 1.0 for relative performance)
    spy_line_x = 1.0
    ax.axvline(x=spy_line_x, color='gray', linestyle='--', alpha=0.7, 
               label='SPY Reference', linewidth=1)
    
    y_positions = []
    labels = []
    
    for i, (symbol, data) in enumerate(sorted_symbols):
        y_pos = len(sorted_symbols) - i - 1
        y_positions.append(y_pos)
        
        # Create label: "SYMBOL (portfolio1, portfolio2) %XX"
        portfolios_str = ', '.join(data['portfolios'])
        pct_str = f"{data['total_pct_overall']:.1f}%"
        label = f"{symbol} ({portfolios_str}) {pct_str}"
        labels.append(label)
        
        # Performance values
        perf_12m = data['perf_12m']
        perf_3m = data['perf_3m']
        perf_1m = data['perf_1m']
        
        # Small arrow: 12m to 3m performance
        ax.annotate('', xy=(perf_3m, y_pos + 0.1), xytext=(perf_12m, y_pos + 0.1),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7, lw=1))
        
        # Big arrow: 3m to 1m performance
        ax.annotate('', xy=(perf_1m, y_pos - 0.1), xytext=(perf_3m, y_pos - 0.1),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.8, lw=2))
        
        # Plot points for reference
        ax.scatter([perf_12m], [y_pos + 0.1], c='blue', s=30, alpha=0.7)
        ax.scatter([perf_3m], [y_pos], c='green', s=40, alpha=0.7)
        ax.scatter([perf_1m], [y_pos - 0.1], c='red', s=30, alpha=0.7)
    
    # Customize the plot
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Performance Relative to S&P 500')
    ax.set_title('Portfolio Performance Analysis\n(Small Arrow: 12m→3m, Large Arrow: 3m→1m)', 
                 fontsize=14, pad=20)
    
    # Add legend for arrows
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=1, label='12m → 3m Performance'),
        Line2D([0], [0], color='red', lw=2, label='3m → 1m Performance'),
        Line2D([0], [0], color='gray', linestyle='--', label='SPY Reference')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Grid and styling
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_axisbelow(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {output_path}")


def main() -> None:
    """Main function to generate portfolio performance chart."""
    # Paths
    csv_path = "data/computed/combined.csv"
    output_path = "data/output/portfolio_performance_chart.png"
    
    # Check if input file exists
    if not Path(csv_path).exists():
        print(f"Error: Input file {csv_path} not found")
        return
    
    # Load and process data
    print("Loading portfolio data...")
    data = load_portfolio_data(csv_path)
    
    print("Grouping symbols by portfolio...")
    symbol_data = group_symbols_by_portfolio(data)
    
    print("Creating performance chart...")
    create_performance_chart(symbol_data, output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()