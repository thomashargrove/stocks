#!/usr/bin/env python3
"""
Portfolio Summary Script

Reads data/stock_data/combined.csv and prints summary statistics for each portfolio:
- Total current value
- Weighted beta
- Total unrealized returns
- Weighted PE ratio
"""

import pandas as pd
import os
import sys

# Configuration: Portfolios to skip in the analysis
SKIP_PORTFOLIOS = {
}


def calculate_portfolio_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate portfolio-level metrics from individual holdings."""
    
    # Calculate current value for each holding
    df['Current_Value'] = df['Last_Price'] * df['Quantity']
    
    # Group by portfolio
    portfolio_metrics = []
    
    for portfolio_name, portfolio_df in df.groupby('Portfolio'):
        # Skip configured portfolios
        if portfolio_name in SKIP_PORTFOLIOS:
            continue
            
        # Total current value
        total_value = portfolio_df['Current_Value'].sum()
        
        # Total unrealized returns - only for positions with known purchase prices
        # Exclude positions where Purchase Price is 0 or NaN (unknown purchase price)
        known_purchases = portfolio_df[
            (portfolio_df['Purchase Price'] > 0) & 
            (portfolio_df['Purchase Price'].notna())
        ]
        total_unrealized = known_purchases['Unrealized_Returns'].sum()
        
        # Calculate weighted beta (exclude NaN values)
        beta_weights = portfolio_df.dropna(subset=['Beta'])
        if len(beta_weights) > 0:
            weighted_beta = (beta_weights['Beta'] * beta_weights['Current_Value']).sum() / beta_weights['Current_Value'].sum()
        else:
            weighted_beta = None
            
        # Calculate weighted PE ratio (exclude NaN values)
        pe_weights = portfolio_df.dropna(subset=['PE_Ratio'])
        if len(pe_weights) > 0:
            weighted_pe = (pe_weights['PE_Ratio'] * pe_weights['Current_Value']).sum() / pe_weights['Current_Value'].sum()
        else:
            weighted_pe = None
        
        portfolio_metrics.append({
            'Portfolio': portfolio_name,
            'Total_Value': total_value,
            'Weighted_Beta': weighted_beta,
            'Total_Unrealized_Returns': total_unrealized,
            'Weighted_PE_Ratio': weighted_pe
        })
    
    return pd.DataFrame(portfolio_metrics)


def main():
    # Check if combined.csv exists
    combined_csv_path = "data/computed/combined.csv"
    
    if not os.path.exists(combined_csv_path):
        print(f"Error: {combined_csv_path} not found. Run 'make' first to generate the combined data.", file=sys.stderr)
        sys.exit(1)
    
    # Read the combined CSV
    try:
        df = pd.read_csv(combined_csv_path)
    except Exception as e:
        print(f"Error reading {combined_csv_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate portfolio metrics
    portfolio_summary = calculate_portfolio_metrics(df)
    
    # Sort by total value (descending)
    portfolio_summary = portfolio_summary.sort_values('Total_Value', ascending=False)
    
    # Print results
    print("PORTFOLIO SUMMARY")
    print("=" * 80)
    print()
    
    for _, row in portfolio_summary.iterrows():
        print(f"Portfolio: {row['Portfolio']}")
        print(f"  Total Current Value:      ${row['Total_Value']:,.2f}")
        print(f"  Total Unrealized Returns: ${row['Total_Unrealized_Returns']:,.2f}")
        
        if pd.notna(row['Weighted_Beta']):
            print(f"  Weighted Beta:            {row['Weighted_Beta']:.3f}")
        else:
            print(f"  Weighted Beta:            N/A")
            
        if pd.notna(row['Weighted_PE_Ratio']):
            print(f"  Weighted PE Ratio:        {row['Weighted_PE_Ratio']:.2f}")
        else:
            print(f"  Weighted PE Ratio:        N/A")
        print()
    
    # Print overall totals
    total_portfolio_value = portfolio_summary['Total_Value'].sum()
    total_unrealized_returns = portfolio_summary['Total_Unrealized_Returns'].sum()
    
    print("OVERALL TOTALS")
    print("=" * 80)
    print(f"Total Portfolio Value:      ${total_portfolio_value:,.2f}")
    print(f"Total Unrealized Returns:   ${total_unrealized_returns:,.2f}")
    
    # Calculate overall weighted averages (excluding skipped portfolios)
    df['Current_Value'] = df['Last_Price'] * df['Quantity']
    active_df = df[~df['Portfolio'].isin(SKIP_PORTFOLIOS)]
    
    # Overall weighted beta
    beta_data = active_df.dropna(subset=['Beta'])
    if len(beta_data) > 0:
        overall_weighted_beta = (beta_data['Beta'] * beta_data['Current_Value']).sum() / beta_data['Current_Value'].sum()
        print(f"Overall Weighted Beta:      {overall_weighted_beta:.3f}")
    else:
        print(f"Overall Weighted Beta:      N/A")
    
    # Overall weighted PE
    pe_data = active_df.dropna(subset=['PE_Ratio'])
    if len(pe_data) > 0:
        overall_weighted_pe = (pe_data['PE_Ratio'] * pe_data['Current_Value']).sum() / pe_data['Current_Value'].sum()
        print(f"Overall Weighted PE Ratio:  {overall_weighted_pe:.2f}")
    else:
        print(f"Overall Weighted PE Ratio:  N/A")
        
    print()
    print(f"Note: Skipped portfolios: {', '.join(sorted(SKIP_PORTFOLIOS))}")


if __name__ == "__main__":
    main()