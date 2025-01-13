import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# data fetcher
def fetch_stock_data(ticker, period='1y'): 
    stock = yf.Ticker(ticker)
    historical_data = stock.history(period=period)
    
    data = historical_data[['Low', 'High', 'Close', 'Open']].copy()
    data.reset_index(inplace=True)
    data['Date'] = data['Date'].dt.date
    
    data[['Low', 'High', 'Close', 'Open']] = data[['Low', 'High', 'Close', 'Open']].round(2)
    
    data.to_csv(f'./stock_data/{ticker}.csv', index=False)
    return data


def prepare_data_for_model(data, window_size=252):
    #create lagged features (e.g., previous 'window_size' days)
    #lagged features = past values of the target variable (in this case, the Close price) that can be used to predict future values.     
    lagged_data = [data['Close'].shift(i) for i in range(window_size, 0, -1)]
    lagged_data.append(data['Close'])  # Adding the current price as the last column
    
    # Concatenate all lagged columns together
    data_lagged = pd.concat(lagged_data, axis=1)
    
    # Drop rows with NaN values (which are the result of shifting)
    data_lagged = data_lagged.dropna()
    
    # Renaming columns
    data_lagged.columns = [f'Lag_{i}' for i in range(window_size, 0, -1)] + ['Close']
    
    # X and y split
    X = data_lagged.drop(columns=['Close'])
    y = data_lagged['Close']
    
    return X, y

def plot_stock_data(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'], label='Closing Price')
    plt.title("Stock Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.show()
    
def save_data_for_company(X, y, ticker):
    folder_path = f'./stock_data/{ticker}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Save the X and y data as CSV files inside the company's folder
    X.to_csv(f'{folder_path}/{ticker}_X.csv', index=False)
    y.to_csv(f'{folder_path}/{ticker}_y.csv', index=False)

data = fetch_stock_data('CM', period='5y')  