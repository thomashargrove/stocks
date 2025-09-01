# Stock Portfolio Analysis

A simple Python program that retrieves stock market history for one or more portfolios, creating an output dataset that an AI coding agent can use for custom analysis and visulizations.

## Quick Start - Define Portfolios

Define your portfolios in `data/input/{name}.csv` using CSV format.  The filename will be used as the portfolio name, and use the following schema.  This format is compatible with Yahoo Finance Portfolio CSV download.


| Symbol | Trade Date | Purchase Price | Quantity | Comment |
|--------|------------|----------------|----------|---------|
| AAPL   | 20250106   | 185.20         | 10       | Tech position |
| MSFT   | 20250107   | 420.50         | 5        | Cloud play |
| NVDA   | 20250627   | 157.21         | 46       | AI exposure |

## Run Refresh.py to Download and Prepare Data

There are 3 options for running the python scrupt.  If you don't care about installing system-wide python libraries do the following:

```bash
pip install -r requirements.txt
python refresh.py
```

To use a virtual environmebnt

```bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
python refresh.py
```

If you are comfortable with UV then:

```bash
uv sync
uv run refresh.py
```

This script will produce 2 outputs.  The first is a file `data/computed/combined.csv`.  This is the main input you will use for custom analysis.  It has one entry for every stock you hold with computed data.  For example:

|Portfolio|Symbol|Trade Date|Purchase Price|Quantity|Comment     |Beta |Last_Price|PctOverall|PctPortfolio|Unrealized_Returns|PE_Ratio|Perf_Rel_SPY_1m|Perf_Rel_SPY_3m|Perf_Rel_SPY_6m|Perf_Rel_SPY_12m|Perf_Rel_SPY_YTD|
|---------|------|----------|--------------|--------|------------|-----|----------|----------|------------|------------------|--------|---------------|---------------|---------------|----------------|----------------|
|example  |AAPL  |20221115.0|148.72        |15.0    |            |1.165|232.14    |3.9       |3.9         |1251.3            |35.23   |1.093          |1.056          |0.88           |0.879           |0.837           |
|example  |MSFT  |20240507.0|412.2         |8.0     |            |1.055|506.69    |4.54      |4.54        |755.92            |37.17   |0.973          |1.006          |1.177          |1.056           |1.087           |
|example  |GOOGL |20230823.0|133.58        |12.0    |            |1.014|212.91    |2.86      |2.86        |951.96            |22.67   |1.066          |1.155          |1.15           |1.13            |1.012           |
|example  |AMZN  |20221017.0|112.47        |20.0    |            |1.314|229.0     |5.13      |5.13        |2330.6            |34.96   |0.978          |1.015          |1.005          |1.107           |0.938           |
|example  |NVDA  |20241003.0|118.93        |10.0    |            |2.145|174.18    |1.95      |1.95        |552.5             |49.48   |0.956          |1.162          |1.374          |1.26            |1.149           |
|example  |META  |20250312.0|485.1         |6.0     |            |1.273|738.7     |4.96      |4.96        |1521.6            |26.81   |1.045          |1.01           |1.016          |1.227           |1.134           |
|example  |TSLA  |20230127.0|173.25        |9.0     |meme stock  |2.331|333.87    |3.36      |3.36        |1445.58           |199.92  |1.029          |0.893          |1.055          |1.346           |0.725           |
|example  |SPY   |20240101.0|600.0         |100.0   |example ETF |1.0  |645.05    |72.19     |72.19       |4505.0            |27.39   |1.0            |1.0            |1.0            |1.0             |1.0             |
|example  |CASH  |20240101.0|1.0           |1000.0  |cash example|0.0  |1.0       |1.12      |1.12        |0.0               |        |0.984          |0.916          |0.9            |0.863           |0.906           |

The other output is data/history/{ticker}.csv.  This will contain the last year of daily prices for the given stock.  For example:

|Date                     |Open             |High              |Low               |Close            |Volume  |Dividends|Stock Splits|Repaired?|Final?|FetchDate                       |
|-------------------------|-----------------|------------------|------------------|-----------------|--------|---------|------------|---------|------|--------------------------------|
|2024-08-30 00:00:00-04:00|229.123652992766 |229.33267166766268|226.42620031745173|227.9391632080078|52990800|0.0      |0.0         |False    |True  |2025-08-31 03:25:36.486229+00:00|
|2024-09-03 00:00:00-04:00|227.4912604304481|227.9391727978887 |220.1454429272487 |221.738037109375 |50190600|0.0      |0.0         |False    |True  |2025-08-31 03:25:36.486229+00:00|

# Example AI Queries

Now that your computed data is ready, you can have an AI coding agent create custom visulizations for you, or you can try out these pre-build scripts (or use them for inspiration).

### scripts/beta_vs_perfoemance.py

```
Write me a python script at scripts/beta_vs_perfoemance.py that reads data/computed/combined.csv and generates a 
scatter plot for all unique symbols with one axis as beta and the other axis as the column 
Perf_Rel_SPY_12m.  Write the output image to data/output and don't pop anything up.
```

Generates:

<img width="3564" height="2364" alt="scatter" src="https://github.com/user-attachments/assets/2bb04681-667f-4f6f-8fcb-5ee944d48b2d" />

### scripts/portfolio_summary.py

```
write me a python script scripts/portfolio_summary.py that reads data/computed/combined.csv and generates a 
summary for each portfolio, including total current value, unrealized returns, weighted beta,
and weighted PE ratio.  Sort portfolios by total value descending.  At the end include an 
overall total and weighted averages across all portfolios.  Handle missing data gracefully 
(NaN values in Beta and PE_Ratio).  Only include positions with known purchase price when 
calculating unrealized returns.
```

Outputs:

```
PORTFOLIO SUMMARY
================================================================================

Portfolio: example
  Total Current Value:      $31,299.87
  Total Unrealized Returns: $9,259.96
  Weighted Beta:            1.271
  Weighted PE Ratio:        48.64

OVERALL TOTALS
================================================================================
Total Portfolio Value:      $31,299.87
Total Unrealized Returns:   $9,259.96
Overall Weighted Beta:      1.271
Overall Weighted PE Ratio:  48.64
```

### scripts/portfolio_performance_chart.py

Prompt:

```
write me a python script portfolio_performance_chart.py that reads data/computed/combined.csv.  
I want it to group together each symbol across portfolios and render a graph where symbols are 
ordered top to bottom based on total percent of overall.  The text on the left should be "{symbol} 
(portfolio1, portfolio2) %25".  To the right of that should be a display  of relative 
performance to S&P500.  Each entry will get a small arrow from 12mon perf to 3mon perf, and 
a big arrow from 3mon relative perf to 1mon relative perf.  Show a dotted line for SPY500 
reference.  Write the output image to data/output/
```

Output:

<img width="4171" height="2370" alt="perf" src="https://github.com/user-attachments/assets/9e694b01-4138-40d5-b0e2-b66e6d2f9a58" />

### scripts/upside_downside.py

```
Write me a python script scripts/upside_downside.py that calculates upside capture, downside 
capture, and spread ratios for unique stocks across all portfolios using 1-year of daily price 
history.  List of stocks can be found in data/computed/combined.csv and price history for all 
symbols are in data/history/{symbol}.csv.  Order outputs from largest to smallest holdings.
Output a heatmap with up/down cells collored from red-yellow-green based on how small-large
the values are.
```

Output:

<img width="800" height="616" alt="image" src="https://github.com/user-attachments/assets/0e99e99c-07c7-4cd1-978c-a236a34a0120" />
