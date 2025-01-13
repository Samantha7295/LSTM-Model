import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#reference https://neptune.ai/blog/predicting-stock-prices-using-machine-learning

# train-test split for time-series
stockprices = pd.read_csv("./stock_data/CM.csv", index_col="Date")
stockprices.index = pd.to_datetime(stockprices.index)

test_ratio = 0.2
training_ratio = 1 - test_ratio

train_size = int(training_ratio * len (stockprices))
test_size = int (test_ratio * len (stockprices))
print(f"train_size: {train_size}")
print(f"test_size: {test_size}")

train = stockprices[:train_size]
test = stockprices[train_size:]

train_close = train["Close"]
test_close = test["Close"]

train.index = pd.to_datetime(train.index)
test.index = pd.to_datetime(test.index)


# will use RMSE(Root Mean Squared Error) and MAPE(Mean Absolute Percentage Error) as model evaluation metrics
# RMSE gives the differences between predicted and true values, whereas MAPE (%) measures this difference relative to the true values. 
# For example, a MAPE value of 12% indicates that the mean difference between the predicted stock price and the actual stock price is 12%.

def extract_seqX_outcomeY(data, N, offset):
    """ Split time-series into training sequence X and outcome value Y

    Args: 
        data - dataset
        N - window size, e.g., 50 for 50 days of historical stock prices
        offset - position to start the split
    """
    X, y = [], []
    
    for i in range(offset + N, len(data)):
        X.append(data[i - N:i])
        y.append(data[i])
        
    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.float32)
    
    if X_array.shape[1] < N:  # Check if padding is needed
        padding = np.zeros((X_array.shape[0], N - X_array.shape[1]))  # Create padding
        X_array = np.hstack((padding, X_array))  # Pad sequences at the beginning
        
    X_tensor = torch.tensor(X_array, dtype=torch.float32)
    y_tensor = torch.tensor(y_array, dtype=torch.float32)
    
    return X_tensor, y_tensor



def calculate_rmse(y_true, y_pred):
    """ Calculate Root Mean Squared Error (RMSE)"""
    y_true, y_pred = torch.tensor(y_true, dtype=torch.float32), torch.tensor(y_pred, dtype=torch.float32)
    rmse = torch.sqrt(torch.mean((y_true - y_pred)**2))
    return rmse

def calculate_mape(y_true, y_pred):
    """ Calculate Mean Absolute Percentage Error (MAPE)"""
    y_true, y_pred = torch.tensor(y_true, dtype=torch.float32), torch.tensor(y_pred, dtype=torch.float32)
    mape = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
    return mape

def calculate_perf_metrics(var):
    #convert stock price data to PyTorch tensors
    true_values = torch.tensor(stockprices[train_size:]["Close"].values, dtype=torch.float32)
    predicted_values = torch.tensor(stockprices[train_size:][var].values, dtype=torch.float32)
    
    #calculate RMSE and MAPE (assuming these functions are defined elsewhere)
    rmse = calculate_rmse(true_values, predicted_values)
    mape = calculate_mape(true_values, predicted_values)
    
    return rmse, mape

def plot_stock_trend(var, cur_title):
    ax = stockprices[["Close", var, "200day"]].plot(figsize=(20, 10))
    plt.grid(False)
    plt.title(cur_title)
    plt.axis("tight")
    plt.ylabel("Stock Price ($)")
    plt.legend()
    plt.show()

def calculate_sma(data, window_size):
    """ Calculate Simple Moving Average (SMA)"""
    
    sma = data['Close'].rolling(window=window_size, min_periods=1).mean()  # Adjust to handle initial NaNs
    
    return sma

window_size = 50
stockprices['SMA'] = calculate_sma(stockprices, window_size)
stockprices['200day'] = calculate_sma(stockprices, 200)
# plot_stock_trend("SMA", "Simple Moving Average")
rmse_sma, mape_sma = calculate_perf_metrics(var="SMA")

# print(f"RMSE for SMA: {rmse_sma}") 
# print(f"MAPE for SMA: {mape_sma}%")
# Of course 50-day SMA is better trend indicator than 200-day SMA in terms of short-to-medium movements

#EMA - Exponential Moving Average: applies higher weights to recent data points, more responsive to price changes than SMA
def calculate_ema(data, window_size):
    """ Calculate Exponential Moving Average (EMA)"""
    
    alpha = 2/(window_size + 1)
    ema = torch.zeros_like(data, dtype=torch.float32)
    ema[0] = data[0]
    
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
        
    return ema

window_ema_var = f"{window_size}_EMA"
close_prices = torch.tensor(stockprices["Close"].values, dtype=torch.float32)
ema_values = calculate_ema(close_prices, window_size)

stockprices[window_ema_var] = ema_values.numpy()
stockprices["200day"] = stockprices["Close"].rolling(200).mean()
# plot_stock_trend(var=window_ema_var, cur_title="Exponential Moving Average")
rmse_ema, mape_ema = calculate_perf_metrics(var=window_ema_var)
# print(f"RMSE for EMA: {rmse_ema}")
# print(f"MAPE for EMA: {mape_ema}%")



#LSTM = long short-term memory - algorithm for time series, captures historical trend patterns and predicts future values with high accuracy
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(StockPriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        #LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        #Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)
        
        
    def forward(self, x):
        #initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        #pass through LSTM layer
        out, _ = self.lstm(x , (h0, c0))
        #take only last output for prediction
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 50
num_layers = 1
batch_size = 20
num_epochs = 50
learning_rate = 0.01

# Model, loss, and optimizer
model = StockPriceLSTM(input_size, hidden_size, num_layers)
criterion = nn.MSELoss() # Mean Squared Error Loss ffor regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#scale dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(stockprices[["Close"]].values)
scaled_data_train = scaled_data[: train.shape[0]]

#prepare training and validation data
X_train, y_train = extract_seqX_outcomeY(scaled_data_train, window_size, 0)

#split into train and validation datasets(80/20)
train_size = int(0.8 * len(X_train))
val_size = len(X_train) - train_size
X_train, X_val = X_train[:train_size], X_train[train_size:]
y_train, y_val = y_train[:train_size], y_train[train_size:]

#convert to pyTorch tensors
X_train_tensor = X_train.clone().detach().to(torch.float32)
y_train_tensor = y_train.clone().detach().to(torch.float32)
X_val_tensor = X_val.clone().detach().to(torch.float32)
y_val_tensor = y_val.clone().detach().to(torch.float32)

train_data = torch.randn(100, 10, input_size)
train_labels = torch.randn(100, 1)


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()
    
    model.eval()
    
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        
    # scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Learning Rate: {current_lr}")
    

        
def preprocess_testdat(data=stockprices, scaler=scaler, window_size=window_size, test=test):
   
    raw = data["Close"][len(data)-len(test)-window_size:].values
    raw = raw.reshape(-1, 1)
    raw = scaler.transform(raw)
    
    # first i was 1
    X_test = [raw[i-window_size:i, 0] for i in range(window_size, raw.shape[0])]
    
    padded_X_test = [np.pad(seq, (0, window_size - len(seq)), mode='constant', constant_values=0) for seq in X_test]
    
    X_test = np.array(padded_X_test)
    
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    X_test = torch.tensor(X_test, dtype=torch.float32)
    return X_test



X_test = preprocess_testdat()

predicted_price_ = model(X_test)
predicted_price_ = predicted_price_.detach().numpy()
predicted_price = scaler.inverse_transform(predicted_price_)


test_copy = test.copy()
test_copy.loc[:, "Predictions_lstm"] = predicted_price


rmse_lstm = calculate_rmse(np.array(test_copy["Close"]), np.array(test_copy["Predictions_lstm"]))
mape_lstm = calculate_mape(np.array(test_copy["Close"]), np.array(test_copy["Predictions_lstm"]))

print(f"RMSE: {rmse_lstm}")
print(f"MAPE: {mape_lstm}%")

def plot_stock_trend_lstm(train, test):
    fig, ax = plt.subplots(figsize = (20,10))
    ax.plot(train.index, train["Close"], label = "Train Closing Price")
    ax.plot(test.index, test["Close"], label = "Test Closing Price")
    ax.plot(test.index, test["Predictions_lstm"], label = "Predicted Closing Price")
    
    ax.set_title("LSTM Model")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price ($)")
    ax.legend(loc="upper left")
    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation = 45)
    
    plt.show()

plot_stock_trend_lstm(train, test_copy)








