import pytest
import pandas as pd
from main import _build_portfolio_dataframe


def test_build_portfolio_dataframe_two_stocks():
    """Test _build_portfolio_dataframe with portfolio containing 2 stocks."""
    portfolio_data = {
        "test_portfolio": "Symbol,Trade Date,Purchase Price,Quantity,Comment\nAAPL,20250106,185.20,10,Test Apple\nMSFT,20250107,420.50,5,Test Microsoft"
    }
    
    result = _build_portfolio_dataframe(portfolio_data)
    
    assert len(result) == 2
    assert list(result.columns) == ["Portfolio", "Symbol", "Trade Date", "Purchase Price", "Quantity", "Comment"]
    assert result["Portfolio"].tolist() == ["test_portfolio", "test_portfolio"]
    assert result["Symbol"].tolist() == ["AAPL", "MSFT"]
    assert result["Purchase Price"].tolist() == [185.20, 420.50]
    assert result["Quantity"].tolist() == [10, 5]


def test_build_portfolio_dataframe_stock_and_cash():
    """Test _build_portfolio_dataframe with 1 stock and 1 cash position (cash should be normalized to CASH)."""
    portfolio_data = {
        "mixed_portfolio": "Symbol,Current Price,Date,Time,Change,Open,High,Low,Volume,Trade Date,Purchase Price,Quantity,Commission,High Limit,Low Limit,Comment,Transaction Type\n$$CASH_TX,,,,,,,,,20250829,,83053.0,,,,,DEPOSIT\nNVDA,174.18,2025/08/29,16:00 EDT,-5.93901,178.11,178.11,173.145,241551243,20250627,157.21,46.0,0.0,,,,BUY"
    }
    
    result = _build_portfolio_dataframe(portfolio_data)
    
    assert len(result) == 2
    assert list(result.columns) == ["Portfolio", "Symbol", "Trade Date", "Purchase Price", "Quantity", "Comment"]
    assert result["Portfolio"].tolist() == ["mixed_portfolio", "mixed_portfolio"]
    assert result["Symbol"].tolist() == ["CASH", "NVDA"]  # $$CASH_TX normalized to CASH
    assert result["Trade Date"].tolist() == [20250829, 20250627]
    assert result["Quantity"].tolist() == [83053.0, 46.0]


def test_build_portfolio_dataframe_empty_input():
    """Test _build_portfolio_dataframe with empty input."""
    result = _build_portfolio_dataframe({})
    
    assert len(result) == 0
    assert list(result.columns) == ["Portfolio", "Symbol", "Trade Date", "Purchase Price", "Quantity", "Comment"]