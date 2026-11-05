import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# ==========================================
# 1. Data Fetching
# ==========================================
ticker = "NOK"
df = yf.download(ticker, start="2022-01-01", end="2026-11-05")
df = df[['Open', 'High', 'Low', 'Close']].copy()

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ==========================================
# 2. Indicator Calculation
# ==========================================
df['SMA_Short'] = df['Close'].rolling(window=20).mean()
df['SMA_Long'] = df['Close'].rolling(window=60).mean()

# RSI Momentum
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# ATR Volatility
df['High-Low'] = df['High'] - df['Low']
df['High-PrevClose'] = (df['High'] - df['Close'].shift(1)).abs()
df['Low-PrevClose'] = (df['Low'] - df['Close'].shift(1)).abs()
df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
df['ATR'] = df['TR'].rolling(window=14).mean()
df['ATR_MA'] = df['ATR'].rolling(window=20).mean()

# ==========================================
# 3. Fixed Step Dynamic Position (No ATR Penalty)
# ==========================================
cond_trend = df['SMA_Short'] > df['SMA_Long']
cond_momentum = df['RSI'] > 50
cond_volatility = df['ATR'] > df['ATR_MA']

score = cond_trend.astype(int) + cond_momentum.astype(int) + cond_volatility.astype(int)

# Dynamic allocation without reducing position size due to high ATR
raw_position = np.select(
    [score == 3, score == 2, score == 1],
    [1.0, 0.6, 0.3],
    default=0.0
)
df['Target_Position'] = np.where(cond_trend, raw_position, 0.0)

# ==========================================
# 4. Realistic Risk & Cost Engine (Wider Trailing Stop)
# ==========================================
buy_cost = 0.0005  # 0.05%
sell_cost = 0.0020  # 0.20%

stop_loss_pct = 0.08  # 8% Hard Stop Loss
trailing_stop_pct = 0.20  # 放寬至 20% Trailing Stop (適應高波動飆股)

final_position = np.zeros(len(df))
cost_series = np.zeros(len(df))

in_position = False
entry_price = 0.0
peak_price = 0.0
current_pos_size = 0.0

close_prices = df['Close'].values
target_positions = df['Target_Position'].values

for i in range(1, len(df)):
    target_pos = target_positions[i - 1]
    price = close_prices[i]
    prev_pos = current_pos_size

    if target_pos > 0:
        if not in_position:
            in_position = True
            entry_price = price
            peak_price = price
            current_pos_size = target_pos
            cost_series[i] = current_pos_size * buy_cost
        else:
            peak_price = max(peak_price, price)

            hit_hard_stop = price < entry_price * (1 - stop_loss_pct)
            hit_trailing_stop = price < peak_price * (1 - trailing_stop_pct)

            if hit_hard_stop or hit_trailing_stop:
                in_position = False
                current_pos_size = 0.0
                cost_series[i] = prev_pos * sell_cost
            else:
                pos_diff = target_pos - prev_pos
                if pos_diff > 0:
                    cost_series[i] = pos_diff * buy_cost
                elif pos_diff < 0:
                    cost_series[i] = abs(pos_diff) * sell_cost
                current_pos_size = target_pos
    else:
        if in_position:
            in_position = False
            cost_series[i] = prev_pos * sell_cost
            current_pos_size = 0.0
        else:
            current_pos_size = 0.0

    final_position[i] = current_pos_size

df['Final_Position'] = final_position
df['Transaction_Cost'] = cost_series

# ==========================================
# 5. Performance Calculation
# ==========================================
df['Market_Return'] = df['Close'].pct_change()
df['Strategy_Net_Return'] = (df['Market_Return'] * df['Final_Position']) - df['Transaction_Cost']

df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
df['Cumulative_Net'] = (1 + df['Strategy_Net_Return']).cumprod()

# ==========================================
# 6. Visualization
# ==========================================
plt.figure(figsize=(12, 6))
plt.plot(df['Cumulative_Market'], label='Buy & Hold (SK Hynix)', color='gray', alpha=0.4)
plt.plot(df['Cumulative_Net'], label='Fixed-Step Realistic Strategy', color='green', linewidth=1.8)

plt.title(f'{ticker} Fixed-Step Quant Performance (Up to Nov 2026)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return Multiplier')
plt.legend()
plt.grid(True)
plt.show()

print(f"Buy & Hold Final Return: {(df['Cumulative_Market'].iloc[-1] - 1) * 100:.2f}%")
print(f"Fixed-Step Net Strategy Return: {(df['Cumulative_Net'].iloc[-1] - 1) * 100:.2f}%")