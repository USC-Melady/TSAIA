import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# ----------------------
# 1. Buy and Hold
# ----------------------
def buy_and_hold(data, begin_trade_index=0):
    data = data[begin_trade_index:].copy()
    data["Position"] = 0
    data["Position"].iloc[0] = 1
    return data["Position"].values

# ----------------------
# 2. SMA Crossover
# ----------------------
def sma_strategy(data, short_window=20, long_window=100,begin_trade_index=0):
    if long_window > begin_trade_index:
        long_window = begin_trade_index
    if short_window >= long_window:
        short_window = long_window//2
    data = data.copy()
    data['Short_SMA'] = data['Close'].rolling(short_window).mean()
    data['Long_SMA'] = data['Close'].rolling(long_window).mean()
    data['Signal'] = 0
    data.loc[data['Short_SMA'] > data['Long_SMA'], 'Signal'] = 1
    data.loc[data['Short_SMA'] < data['Long_SMA'], 'Signal'] = -1
    data['Position'] = data['Signal'].shift().fillna(0)
    return data[begin_trade_index:].copy()["Position"].values#simulate_trades(data[begin_trade_index:].copy(), initial_cash)

# ----------------------
# 3. MACD
# ----------------------
def macd_strategy(data,begin_trade_index=0):
    data = data.copy()
    data['EMA12'] = data['Close'].ewm(span=12).mean()
    data['EMA26'] = data['Close'].ewm(span=26).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    data['Signal'] = 0
    data.loc[data['MACD'] > data['Signal_Line'], 'Signal'] = 1
    data.loc[data['MACD'] < data['Signal_Line'], 'Signal'] = -1
    data['Position'] = data['Signal'].shift().fillna(0)
    return data[begin_trade_index:].copy()["Position"].values#simulate_trades(data[begin_trade_index:].copy(), initial_cash)

# ----------------------
# 4. KDJ + RSI
# ----------------------
def kdj_rsi_strategy(data,begin_trade_index=0):
    data = data.copy()
    low_min = data['Low'].rolling(window=9).min()
    high_max = data['High'].rolling(window=9).max()
    data['RSV'] = (data['Close'] - low_min) / (high_max - low_min) * 100
    data['K'] = data['RSV'].ewm(com=2).mean()
    data['D'] = data['K'].ewm(com=2).mean()
    data['J'] = 3 * data['K'] - 2 * data['D']
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['Signal'] = 0
    data.loc[(data['J'] > 80) & (data['RSI'] > 70), 'Signal'] = -1
    data.loc[(data['J'] < 20) & (data['RSI'] < 30), 'Signal'] = 1
    data['Position'] = data['Signal'].shift().fillna(0)
    return data[begin_trade_index:].copy()["Position"].values#simulate_trades(data[begin_trade_index:].copy(), initial_cash)

# ----------------------
# 5. ZMR (Zero Mean Reversion)
# ----------------------
def zmr_strategy(data, window=20, threshold=2,begin_trade_index=0):
    data = data.copy()
    data['Rolling_Mean'] = data['Close'].rolling(window).mean()
    data['Rolling_Std'] = data['Close'].rolling(window).std()
    data['Z_Score'] = (data['Close'] - data['Rolling_Mean']) / data['Rolling_Std']
    data['Signal'] = 0
    data.loc[data['Z_Score'] > threshold, 'Signal'] = -1
    data.loc[data['Z_Score'] < -threshold, 'Signal'] = 1
    data['Position'] = data['Signal'].shift().fillna(0)
    return data[begin_trade_index:].copy()["Position"].values#simulate_trades(data[begin_trade_index:].copy(), initial_cash)

# ----------------------
# Simulate trades helper
# ----------------------
def simulate_trades(data, initial_cash):
    cash = initial_cash
    shares = 0
    portfolio_value = []
    for i in range(len(data)):
        price = data['Close'].iloc[i]
        position = data['Position'].iloc[i]
        if position == 1 and cash > 0:
            shares = cash / price
            cash = 0
        elif position == -1 and shares > 0:
            cash = shares * price
            shares = 0
        portfolio_value.append(cash + shares * price)
    data['Portfolio_Value'] = portfolio_value
    return data

# ----------------------
# Evaluate performance
# ----------------------
def evaluate_investment(data, initial_cash):
    risk_free = pd.read_csv("market_indices/risk_free_rate.csv", parse_dates=True, index_col=0)
    risk_free.index = pd.to_datetime(risk_free.index).date
    rf_series = risk_free["Close"]
    data = simulate_trades(data, initial_cash)
    returns = data['Portfolio_Value'].pct_change().dropna()

    rf_daily = rf_series.reindex(returns.index).fillna(method='ffill') / 100 / 252

    excess_returns = returns - rf_daily
    cumulative_return = data['Portfolio_Value'].iloc[-1] / data['Portfolio_Value'].iloc[0] - 1
    annualized_return = (1 + cumulative_return) ** (252 / len(data)) - 1
    sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    drawdown = data['Portfolio_Value'] / data['Portfolio_Value'].cummax() - 1
    max_drawdown = -drawdown.min()
    return {
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }

# ----------------------
# Plot function
# ----------------------
def plot_portfolio(data, title="Portfolio Value"):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Portfolio_Value'], label="Portfolio Value")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
    # plt.savefig(f"{title}.png")
