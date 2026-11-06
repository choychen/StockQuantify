import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="動態量化策略回測系統", layout="wide")
st.title("📈 多因子量化趨勢與風控系統")

# ==========================================
# 1. 側邊欄設定
# ==========================================
st.sidebar.header("策略參數設定")
ticker = st.sidebar.text_input("股票代碼 (Yahoo Finance)", value="000660.KS")
start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("2026-11-05"))

st.sidebar.subheader("風控與手續費")
stop_loss_pct = st.sidebar.slider("固定止損 (%)", 3.0, 15.0, 8.0) / 100
trailing_stop_pct = st.sidebar.slider("移動追蹤止盈 (%)", 5.0, 30.0, 20.0) / 100
buy_cost = st.sidebar.number_input("買入費率 (%)", value=0.05) / 100
sell_cost = st.sidebar.number_input("賣出費率 (%)", value=0.20) / 100

# ==========================================
# 2. 加載數據 (使用 @st.cache_data 避免請求被鎖 IP)
# ==========================================
@st.cache_data(ttl=3600)  # 快取 1 小時，避免頻繁請求 Yahoo Finance
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

if ticker:
    with st.spinner("正在讀取市場數據..."):
        try:
            df = load_data(ticker, start_date, end_date)
            
            if df is None or df.empty:
                st.warning("⚠️ 無法取得資料，請確認股票代碼是否正確，或稍後再試。")
            else:
                df = df[['Open', 'High', 'Low', 'Close']].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 計算指標
                df['SMA_Short'] = df['Close'].rolling(window=20).mean()
                df['SMA_Long'] = df['Close'].rolling(window=60).mean()

                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))

                df['High-Low'] = df['High'] - df['Low']
                df['High-PrevClose'] = (df['High'] - df['Close'].shift(1)).abs()
                df['Low-PrevClose'] = (df['Low'] - df['Close'].shift(1)).abs()
                df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
                df['ATR'] = df['TR'].rolling(window=14).mean()
                df['ATR_MA'] = df['ATR'].rolling(window=20).mean()

                # 信號與動態倉位
                cond_trend = df['SMA_Short'] > df['SMA_Long']
                cond_momentum = df['RSI'] > 50
                cond_volatility = df['ATR'] > df['ATR_MA']
                score = cond_trend.astype(int) + cond_momentum.astype(int) + cond_volatility.astype(int)

                raw_position = np.select([score == 3, score == 2, score == 1], [1.0, 0.6, 0.3], default=0.0)
                df['Target_Position'] = np.where(cond_trend, raw_position, 0.0)

                # 模擬風控迴圈
                final_position = np.zeros(len(df))
                cost_series = np.zeros(len(df))
                in_position, entry_price, peak_price, current_pos_size = False, 0.0, 0.0, 0.0

                close_prices = df['Close'].values
                target_positions = df['Target_Position'].values

                for i in range(1, len(df)):
                    target_pos = target_positions[i-1]
                    price = close_prices[i]
                    prev_pos = current_pos_size

                    if target_pos > 0:
                        if not in_position:
                            in_position, entry_price, peak_price, current_pos_size = True, price, price, target_pos
                            cost_series[i] = current_pos_size * buy_cost
                        else:
                            peak_price = max(peak_price, price)
                            hit_hard_stop = price < entry_price * (1 - stop_loss_pct)
                            hit_trailing_stop = price < peak_price * (1 - trailing_stop_pct)

                            if hit_hard_stop or hit_trailing_stop:
                                in_position, current_pos_size = False, 0.0
                                cost_series[i] = prev_pos * sell_cost
                            else:
                                pos_diff = target_pos - prev_pos
                                if pos_diff > 0: cost_series[i] = pos_diff * buy_cost
                                elif pos_diff < 0: cost_series[i] = abs(pos_diff) * sell_cost
                                current_pos_size = target_pos
                    else:
                        if in_position:
                            in_position, current_pos_size = False, 0.0
                            cost_series[i] = prev_pos * sell_cost
                        else:
                            current_pos_size = 0.0

                    final_position[i] = current_pos_size

                df['Final_Position'] = final_position
                df['Transaction_Cost'] = cost_series
                df['Market_Return'] = df['Close'].pct_change()
                df['Strategy_Net_Return'] = (df['Market_Return'] * df['Final_Position']) - df['Transaction_Cost']

                df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
                df['Cumulative_Net'] = (1 + df['Strategy_Net_Return']).cumprod()

                # 結果呈現
                mkt_ret = (df['Cumulative_Market'].iloc[-1] - 1) * 100
                strat_ret = (df['Cumulative_Net'].iloc[-1] - 1) * 100

                col1, col2 = st.columns(2)
                col1.metric("買入持有報酬率", f"{mkt_ret:.2f}%")
                col2.metric("量化策略淨報酬率", f"{strat_ret:.2f}%", delta=f"{strat_ret - mkt_ret:.2f}%")

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df['Cumulative_Market'], label='Buy & Hold', color='gray', alpha=0.5)
                ax.plot(df['Cumulative_Net'], label='Quant Strategy', color='green', linewidth=2)
                ax.set_title(f"{ticker} Performance")
                ax.grid(True)
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"連線或計算時發生錯誤: {e}")
