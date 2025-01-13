import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Creating Synethetic Data
def fetch_stock_data(ticker, start_date, end_date, interval="1m"):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    
    df.index = pd.to_datetime(df.index)
    np.random.seed(42)
    df['bid_fill'] = np.random.randint(100, 2000, size=len(df))
    df['ask_fill'] = np.random.randint(100, 2000, size=len(df))
    df['Signed Volume'] = df['bid_fill'] - df['ask_fill']
    df['price'] = df['Close']
    df['best_bid'] = df['Low']
    df['best_ask'] = df['High']
    df['mid_price'] = (df['best_bid'] + df['best_ask']) / 2
    
    df = df[['bid_fill', 'ask_fill', 'Signed Volume', 'price', 'best_bid', 'best_ask', 'mid_price']]
    
    return df


def graph_data(data):
    # Some visualiation graphs with the synthetic data
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['price'], label="Price")
    plt.plot(data.index, data['mid_price'], label="Mid Price")
    plt.title("Price vs Mid Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['price'], label="Price")
    plt.plot(data.index, data['mid_price'], label="Mid Price")
    plt.plot(data.index, data['best_bid'], label="Best Bid")
    plt.plot(data.index, data['best_ask'], label="Best Ask")
    plt.title("Price vs Mid Price vs Best Bid vs Best Ask")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    
# TWAP strategy
def twap(data):
    total_volume = abs(data['Signed Volume']).sum()
    num_intervals = len(data)
    volume_p_interval = total_volume / num_intervals
    
    executed_volume = 0
    twap_price = 0
    execution_prices = data['price']
    
    for i, price in enumerate(execution_prices):
        executed_volume += volume_p_interval
        twap_price += price * volume_p_interval
        
    twap_price = twap_price / total_volume
        
    return twap_price
    
def vwap(data):
    execution_prices = data['price']
    volumes = data['Signed Volume']

    # Calculate VWAP
    vwap = np.cumsum(execution_prices * volumes) / np.cumsum(volumes)
    
    return vwap


# Calculate Performance Metrics
def execution_cost(data, vwap):
    execution_prices = data['price']
    volumes = data['Signed Volume']
    execution_costs = (execution_prices - vwap) * volumes
    total_execution_costs = sum(execution_costs) /sum(volumes)
    return total_execution_costs
    
def slippage(data, prices):
    execution_prices = data['price']
    expected_price = data['mid_price']
    volumes = data['Signed Volume']
    slippage = ((execution_prices - expected_price) / expected_price) * volumes
    total_slippage = sum(slippage) / sum(volumes)
    return total_slippage

def fill_rate(data):
    bid_fill = data['bid_fill']
    ask_fill = data['ask_fill']
    signed_volume = data['Signed Volume']
    
    total_executed_volume = bid_fill.sum() + ask_fill.sum()
    total_intended_volume = abs(signed_volume).sum()

    if total_intended_volume != 0:
        fill_rate = (total_executed_volume / total_intended_volume) 
    else:
        fill_rate = 0
    return fill_rate

def plot_results(data, vwap_series):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['price'], label="Price", marker='o')
    plt.plot(data.index, vwap_series, label="VWAP", color='r', linestyle='-')
    plt.title("Price vs VWAP")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def backtesting(ticker, start_date, end_date):
    data = fetch_stock_data(ticker, start_date, end_date)
    twap_price = twap(data)
    vwap_series = vwap(data)
    total_execution_cost = execution_cost(data, vwap_series)
    total_slippage = slippage(data, data['price'])
    total_fill_rate = fill_rate(data)
    
    
    # plot_results(data, vwap_series)
    
    print(f"TWAP Price: {twap_price}")
    print(f"Execution Cost: {total_execution_cost}")
    print(f"Slippage: {total_slippage}")
    print(f"Fill Rate: {total_fill_rate}")

# Use Case
ticker = "MSFT" 
start_date = "2024-12-10"
end_date = "2024-12-11"
backtesting(ticker, start_date, end_date)