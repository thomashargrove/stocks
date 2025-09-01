from __future__ import annotations

import yfinance_cache as yfc
import logging
import os
import pandas as pd
import glob
from pathlib import Path
import time
from datetime import datetime, timezone
import sys

# Portfolio filter - if empty ignore, otherwise filter to only these portfolios
PORTFOLIO_FILTER = []

# Cash symbols to normalize in portfolio data
CASH_SYMBOLS = ["CASH", "$$CASH_TX"]

# Beta placeholder values for symbols where Yahoo Finance doesn't provide beta
BETA_PLACEHOLDER = {
    "CASH": 0.0,
    "RDDT": 2.1,
}

# Rate limiting constants
YAHOO_REQUEST_THROTTLE_SECONDS = 2.5
FETCH_DATE_THRESHOLD_SECONDS = 20

def refresh_ticker_history(symbol: str, output_path: str) -> None:
    """Refresh historical data for a single ticker symbol and save to CSV.
    
    Args:
        symbol: Stock ticker symbol to process
        output_path: Full path to the output CSV file
    """
    try:
        dat = yfc.Ticker(symbol)
        history_data = dat.history(period="1y")
    except (TypeError, ValueError, Exception) as e:
        print(f"Error fetching data for {symbol}: {e}")
        print(f"Skipping {symbol} - ticker may be invalid or delisted")
        return
    #print(f"History data:\n{history_data}")
    
    #beta = dat.info.get("beta")
    #print("Beta:", beta)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write to CSV
    history_data.to_csv(output_path)
    print(f"Data saved to {output_path}")
    
    # Check if we hit Yahoo Finance recently and throttle if needed
    if 'FetchDate' in history_data.columns and not history_data.empty:
        # Get the most recent FetchDate
        fetch_date_str = history_data['FetchDate'].iloc[-1]
        if pd.notna(fetch_date_str):
            try:
                # Parse the timestamp string - convert to pandas timestamp then to datetime
                fetch_date = pd.to_datetime(fetch_date_str, utc=True).to_pydatetime()
                current_time = datetime.now(timezone.utc)
                time_diff = (current_time - fetch_date).total_seconds()
                
                if time_diff <= FETCH_DATE_THRESHOLD_SECONDS:
                    print(f"Recent Yahoo Finance request detected (fetch was {time_diff:.1f}s ago), sleeping for {YAHOO_REQUEST_THROTTLE_SECONDS}s to throttle requests...")
                    time.sleep(YAHOO_REQUEST_THROTTLE_SECONDS)
                    
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse FetchDate '{fetch_date_str}': {e}")

def _build_portfolio_dataframe(portfolio_data: dict[str, str]) -> pd.DataFrame:
    """Build combined DataFrame from portfolio file contents.
    
    Args:
        portfolio_data: Dictionary mapping portfolio names to CSV content strings
        
    Returns:
        Combined DataFrame with columns: Portfolio, Symbol, Trade Date, Purchase Price, Quantity, Comment
    """
    if not portfolio_data:
        return pd.DataFrame(columns=["Portfolio", "Symbol", "Trade Date", "Purchase Price", "Quantity", "Comment"])
    
    combined_data: list[pd.DataFrame] = []
    
    for portfolio_name, csv_content in portfolio_data.items():
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(csv_content))
            
            # Normalize cash symbols to the first entry in CASH_SYMBOLS
            df = df.copy()
            df.loc[df['Symbol'].isin(CASH_SYMBOLS), 'Symbol'] = CASH_SYMBOLS[0]
            
            # Add Portfolio column
            df['Portfolio'] = portfolio_name
            
            # Select and rename columns to match desired output
            columns_to_keep = ["Portfolio", "Symbol", "Trade Date", "Purchase Price", "Quantity", "Comment"]
            df = df[columns_to_keep]
            
            combined_data.append(df)
            
        except Exception as e:
            print(f"Error processing portfolio {portfolio_name}: {e}")
            continue
    
    if not combined_data:
        return pd.DataFrame(columns=["Portfolio", "Symbol", "Trade Date", "Purchase Price", "Quantity", "Comment"])
    
    return pd.concat(combined_data, ignore_index=True)

def parse_portfolios(portfolios_dir: str) -> pd.DataFrame:
    """Parse portfolio CSV files and return combined DataFrame with Portfolio column.
    
    Args:
        portfolios_dir: Directory containing portfolio CSV files
        
    Returns:
        Combined DataFrame with columns: Portfolio, Symbol, Trade Date, Purchase Price, Quantity, Comment
    """
    portfolio_files = glob.glob(os.path.join(portfolios_dir, "*.csv"))
    
    portfolio_data: dict[str, str] = {}
    
    for file_path in portfolio_files:
        portfolio_name = Path(file_path).stem
        
        # Apply portfolio filter - if filter is not empty and portfolio not in filter, skip
        if PORTFOLIO_FILTER and portfolio_name not in PORTFOLIO_FILTER:
            continue
            
        try:
            with open(file_path, 'r') as f:
                portfolio_data[portfolio_name] = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    return _build_portfolio_dataframe(portfolio_data)

def refresh_stock_history(portfolios: pd.DataFrame, stock_history_dir: str) -> None:
    """Refresh historical data for all unique stock symbols in portfolios (excluding cash).
    
    Args:
        portfolios: DataFrame containing portfolio data with Symbol column
        stock_history_dir: Directory to save stock history CSV files
    """
    if portfolios.empty:
        return
    
    # Get unique symbols excluding cash
    unique_symbols = portfolios['Symbol'].unique()
    stock_symbols = [symbol for symbol in unique_symbols if symbol not in CASH_SYMBOLS]
    
    for symbol in stock_symbols:
        output_path = os.path.join(stock_history_dir, f"{symbol}.csv")
        refresh_ticker_history(symbol, output_path)

def fetch_beta_from_yahoo(symbol: str):
    """Fetch beta from Yahoo Finance, handling both stocks and ETFs."""
    try:
        t = yfc.Ticker(symbol)
        
        # Try stock-style beta first
        beta = t.info.get("beta")
        if beta is not None:
            return beta

        # For ETFs, try beta3Year
        beta = t.info.get("beta3Year")
        if beta is not None:
            return beta
            
        return None  # Yahoo doesn't have beta data for this symbol
        
    except Exception:
        return None

def compute_relative_performance(symbol1: str, symbol2: str, period: str, stock_history_dir: str) -> float | None:
    """Compute relative performance of symbol1 vs symbol2 over a given time period.
    
    Args:
        symbol1: Primary symbol to compare
        symbol2: Benchmark symbol (e.g., SPY)
        period: Time period ('1m', '3m', '6m', '12m', 'ytd')
        stock_history_dir: Directory containing stock history CSV files
        
    Returns:
        Relative performance ratio (1.0 = same performance, >1.0 = outperformed)
        None if unable to calculate
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Handle cash specially - assume 0% return over all periods
    if symbol1 in CASH_SYMBOLS:
        # Get SPY performance to compute cash underperformance
        spy_file = os.path.join(stock_history_dir, "SPY.csv")
        if not os.path.exists(spy_file):
            return None
        try:
            spy_df = pd.read_csv(spy_file, index_col=0, parse_dates=True)
            spy_df = spy_df.sort_index()
            if len(spy_df) < 2:
                return None
                
            # Calculate periods in days
            period_days = {
                '1m': 30,
                '3m': 90,
                '6m': 180,
                '12m': 365,
                'ytd': (datetime.now() - datetime(datetime.now().year, 1, 1)).days
            }.get(period)
            
            if period_days is None:
                return None
                
            cutoff_date = spy_df.index[-1] - timedelta(days=period_days)
            period_spy = spy_df[spy_df.index >= cutoff_date]
            
            if len(period_spy) < 2:
                return None
                
            spy_return = (period_spy['Close'].iloc[-1] / period_spy['Close'].iloc[0]) - 1
            # Cash return is 0%, so relative performance is 1 / (1 + spy_return)
            return 1.0 / (1.0 + spy_return)
            
        except Exception:
            return None
    
    # For regular stocks, compute actual relative performance
    symbol1_file = os.path.join(stock_history_dir, f"{symbol1}.csv")
    symbol2_file = os.path.join(stock_history_dir, f"{symbol2}.csv")
    
    if not os.path.exists(symbol1_file) or not os.path.exists(symbol2_file):
        return None
        
    try:
        # Read both files
        df1 = pd.read_csv(symbol1_file, index_col=0, parse_dates=True)
        df2 = pd.read_csv(symbol2_file, index_col=0, parse_dates=True)
        
        # Sort by date
        df1 = df1.sort_index()
        df2 = df2.sort_index()
        
        if len(df1) < 2 or len(df2) < 2:
            return None
        
        # Calculate periods in days
        period_days = {
            '1m': 30,
            '3m': 90, 
            '6m': 180,
            '12m': 365,
            'ytd': (datetime.now() - datetime(datetime.now().year, 1, 1)).days
        }.get(period)
        
        if period_days is None:
            return None
            
        # Get cutoff date (use the later of the two latest dates to ensure alignment)
        latest_date = min(df1.index[-1], df2.index[-1])
        cutoff_date = latest_date - timedelta(days=period_days)
        
        # Filter to period
        period_df1 = df1[df1.index >= cutoff_date]
        period_df2 = df2[df2.index >= cutoff_date]
        
        if len(period_df1) < 2 or len(period_df2) < 2:
            return None
            
        # Calculate returns
        return1 = (period_df1['Close'].iloc[-1] / period_df1['Close'].iloc[0]) - 1
        return2 = (period_df2['Close'].iloc[-1] / period_df2['Close'].iloc[0]) - 1
        
        # Relative performance: (1 + return1) / (1 + return2)
        return (1.0 + return1) / (1.0 + return2)
        
    except Exception:
        return None

def write_summary_csv(portfolios: pd.DataFrame, filename: str, stock_history_dir: str) -> None:
    """Write summary CSV with portfolio data plus beta and last price for each stock.
    
    Args:
        portfolios: DataFrame containing portfolio data
        filename: Full path to output CSV file
        stock_history_dir: Directory containing stock history CSV files
    """
    if portfolios.empty:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Start with a copy of portfolios
    summary_df = portfolios.copy()
    
    # Add columns for beta, last price, and computed metrics
    summary_df['Beta'] = None
    summary_df['Last_Price'] = None
    summary_df['PctOverall'] = None
    summary_df['PctPortfolio'] = None
    summary_df['Unrealized_Returns'] = None
    summary_df['PE_Ratio'] = None
    summary_df['Perf_Rel_SPY_1m'] = None
    summary_df['Perf_Rel_SPY_3m'] = None
    summary_df['Perf_Rel_SPY_6m'] = None
    summary_df['Perf_Rel_SPY_12m'] = None
    summary_df['Perf_Rel_SPY_YTD'] = None
    
    # Process all symbols (including cash for beta assignment)
    all_symbols = portfolios['Symbol'].unique()
    
    for symbol in all_symbols:
        # Skip history file processing for cash symbols
        if symbol in CASH_SYMBOLS:
            # For cash symbols, only assign beta from placeholder
            if symbol in BETA_PLACEHOLDER:
                # Calculate cash relative performance (underperformance vs SPY)
                perf_1m = compute_relative_performance(symbol, 'SPY', '1m', stock_history_dir)
                perf_3m = compute_relative_performance(symbol, 'SPY', '3m', stock_history_dir)
                perf_6m = compute_relative_performance(symbol, 'SPY', '6m', stock_history_dir)
                perf_12m = compute_relative_performance(symbol, 'SPY', '12m', stock_history_dir)
                perf_ytd = compute_relative_performance(symbol, 'SPY', 'ytd', stock_history_dir)
                
                # Round performance ratios to 3 decimal places
                if perf_1m is not None:
                    perf_1m = round(perf_1m, 3)
                if perf_3m is not None:
                    perf_3m = round(perf_3m, 3)
                if perf_6m is not None:
                    perf_6m = round(perf_6m, 3)
                if perf_12m is not None:
                    perf_12m = round(perf_12m, 3)
                if perf_ytd is not None:
                    perf_ytd = round(perf_ytd, 3)
                
                symbol_mask = summary_df['Symbol'] == symbol
                summary_df.loc[symbol_mask, 'Beta'] = BETA_PLACEHOLDER[symbol]
                summary_df.loc[symbol_mask, 'Last_Price'] = 1.0  # Cash is always worth 1.0
                summary_df.loc[symbol_mask, 'Perf_Rel_SPY_1m'] = perf_1m
                summary_df.loc[symbol_mask, 'Perf_Rel_SPY_3m'] = perf_3m
                summary_df.loc[symbol_mask, 'Perf_Rel_SPY_6m'] = perf_6m
                summary_df.loc[symbol_mask, 'Perf_Rel_SPY_12m'] = perf_12m
                summary_df.loc[symbol_mask, 'Perf_Rel_SPY_YTD'] = perf_ytd
            continue
            
        # Process stock symbols with history files
        history_file = os.path.join(stock_history_dir, f"{symbol}.csv")
        
        if os.path.exists(history_file):
            try:
                # Read the history data for this symbol
                history_df = pd.read_csv(history_file, index_col=0)
                
                if not history_df.empty:
                    # Get the most recent Close price (last price)
                    last_price = history_df['Close'].iloc[-1] if 'Close' in history_df.columns else None
                    if last_price is not None:
                        last_price = round(last_price, 2)
                    
                    # Extract beta and PE ratio from ticker info
                    try:
                        ticker = yfc.Ticker(symbol)
                        beta = fetch_beta_from_yahoo(symbol)
                        pe_ratio = ticker.info.get("trailingPE")
                    except Exception as ex:
                        print(f"Failed to get ticker info, err: {ex}")
                        beta = None
                        pe_ratio = None
                    
                    # Use placeholder beta if Yahoo Finance doesn't provide one
                    if beta is None and symbol in BETA_PLACEHOLDER:
                        beta = BETA_PLACEHOLDER[symbol]
                    
                    # Round PE ratio if available
                    if pe_ratio is not None:
                        pe_ratio = round(pe_ratio, 2)
                    
                    # Calculate relative performance vs SPY for different periods
                    perf_1m = compute_relative_performance(symbol, 'SPY', '1m', stock_history_dir)
                    perf_3m = compute_relative_performance(symbol, 'SPY', '3m', stock_history_dir)
                    perf_6m = compute_relative_performance(symbol, 'SPY', '6m', stock_history_dir)
                    perf_12m = compute_relative_performance(symbol, 'SPY', '12m', stock_history_dir)
                    perf_ytd = compute_relative_performance(symbol, 'SPY', 'ytd', stock_history_dir)
                    
                    # Round performance ratios to 3 decimal places
                    if perf_1m is not None:
                        perf_1m = round(perf_1m, 3)
                    if perf_3m is not None:
                        perf_3m = round(perf_3m, 3)
                    if perf_6m is not None:
                        perf_6m = round(perf_6m, 3)
                    if perf_12m is not None:
                        perf_12m = round(perf_12m, 3)
                    if perf_ytd is not None:
                        perf_ytd = round(perf_ytd, 3)
                    
                    # Update all rows for this symbol
                    symbol_mask = summary_df['Symbol'] == symbol
                    summary_df.loc[symbol_mask, 'Beta'] = beta
                    summary_df.loc[symbol_mask, 'Last_Price'] = last_price
                    summary_df.loc[symbol_mask, 'PE_Ratio'] = pe_ratio
                    summary_df.loc[symbol_mask, 'Perf_Rel_SPY_1m'] = perf_1m
                    summary_df.loc[symbol_mask, 'Perf_Rel_SPY_3m'] = perf_3m
                    summary_df.loc[symbol_mask, 'Perf_Rel_SPY_6m'] = perf_6m
                    summary_df.loc[symbol_mask, 'Perf_Rel_SPY_12m'] = perf_12m
                    summary_df.loc[symbol_mask, 'Perf_Rel_SPY_YTD'] = perf_ytd
                    
            except Exception as e:
                print(f"Warning: Could not process history data for {symbol}: {e}")
    
    # Calculate computed columns
    # Convert numeric columns to proper types, filling NaN with 0 for calculations
    summary_df['Last_Price'] = pd.to_numeric(summary_df['Last_Price'], errors='coerce').fillna(0)
    summary_df['Quantity'] = pd.to_numeric(summary_df['Quantity'], errors='coerce').fillna(0)
    summary_df['Purchase Price'] = pd.to_numeric(summary_df['Purchase Price'], errors='coerce').fillna(0)
    
    # Calculate current market values
    summary_df['Current_Value'] = summary_df['Last_Price'] * summary_df['Quantity']
    
    # Calculate portfolio totals (per portfolio)
    portfolio_totals = summary_df.groupby('Portfolio')['Current_Value'].sum()
    summary_df['Portfolio_Total'] = summary_df['Portfolio'].map(portfolio_totals)
    
    # Calculate overall total (across all portfolios)
    overall_total = summary_df['Current_Value'].sum()
    
    # Calculate percentages (avoid division by zero)
    summary_df['PctPortfolio'] = (summary_df['Current_Value'] / summary_df['Portfolio_Total'] * 100).round(2)
    summary_df['PctOverall'] = (summary_df['Current_Value'] / overall_total * 100).round(2) if overall_total > 0 else 0
    
    # Calculate unrealized returns (current value - initial investment)
    summary_df['Initial_Investment'] = summary_df['Purchase Price'] * summary_df['Quantity'] 
    summary_df['Unrealized_Returns'] = (summary_df['Current_Value'] - summary_df['Initial_Investment']).round(2)
    
    # Drop temporary columns
    summary_df = summary_df.drop(columns=['Current_Value', 'Portfolio_Total', 'Initial_Investment'])
    
    # Write the summary CSV
    summary_df.to_csv(filename, index=False)
    print(f"Summary CSV saved to {filename}")

def main() -> None:
    """Main entry point for the application."""
    
    # Parse command line arguments for data directory
    if len(sys.argv) < 2:
        data_dir = "data"
        print(f"No data directory specified, using default: {data_dir}")
    else:
        data_dir = sys.argv[1]
    
    # Compute directory paths
    portfolios_dir = os.path.join(data_dir, "input")
    stock_history_dir = os.path.join(data_dir, "history")
    computed_dir = os.path.join(data_dir, "computed")
    combined_csv = os.path.join(computed_dir, "combined.csv")
    cache_dir = os.path.join(data_dir, "cache")

    portfolios = parse_portfolios(portfolios_dir)
    #print(portfolios)

    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    yfc.EnableLogging(logging.INFO)    
    # Set cache directory to local data folder so it gets checked in and you can run this on different machines
    yfc.yfc_cache_manager.SetCacheDirpath(cache_dir)
    
    refresh_stock_history(portfolios, stock_history_dir)
    write_summary_csv(portfolios, combined_csv, stock_history_dir)

if __name__ == "__main__":
    main()
