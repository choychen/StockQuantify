import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 頁面配置
st.set_page_config(page_title="動態量化策略回測系統", layout="wide")

# 2. 注入手機手勢解鎖與觸控優化 CSS
st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <style>
        /* 解除手機瀏覽器對 Plotly 圖表的拉動攔截，允許雙指捏合與單指滑動 */
        .stPlotlyChart > div {
            touch-action: pan-x pan-y !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📈 多因子量化趨勢與風控系統")

# ==========================================
# 側邊欄設定
# ==========================================
st.sidebar.header("策略參數設定")
ticker = st.sidebar.text_input("股票代碼 (Yahoo Finance)", value="000660.KS")
start_date = st.sidebar.date_input("開始日期", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("結束日期", pd.to_datetime("2026-11-06"))

st.sidebar.subheader("風控與手續費")
stop_loss_pct = st.sidebar.slider("固定止損 (%)", 3.0, 20.0, 12.4) / 100
trailing_stop_pct = st.sidebar.slider("移動追蹤止盈 (%)", 5.0, 30.0, 15.2) / 100
buy_cost = st.sidebar.number_input("買入費率 (%)", value=0.05) / 100
sell_cost = st.sidebar.number_input("賣出費率 (%)", value=0.20) / 100

# ==========================================
# 數據加載與策略計算
# ==========================================
@st.cache_data(ttl=3600)
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

if ticker:
    with st.spinner("正在加載數據並配置手機原生手勢..."):
        try:
            df = load_data(ticker, start_date, end_date)
            
            if df is None or df.empty:
                st.warning("⚠️ 無法取得資料，請確認股票代碼與時間設定。")
            else:
                df = df[['Open', 'High', 'Low', 'Close']].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # 技術指標計算
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

                # 信號與倉位
                cond_trend = df['SMA_Short'] > df['SMA_Long']
                cond_momentum = df['RSI'] > 50
                cond_volatility = df['ATR'] > df['ATR_MA']
                score = cond_trend.astype(int) + cond_momentum.astype(int) + cond_volatility.astype(int)

                raw_position = np.select([score == 3, score == 2, score == 1], [1.0, 0.6, 0.3], default=0.0)
                df['Target_Position'] = np.where(cond_trend, raw_position, 0.0)

                # 風控迴圈
                final_position = np.zeros(len(df))
                cost_series = np.zeros(len(df))
                trade_signals = [''] * len(df)
                
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
                            trade_signals[i] = f'買入建倉 ({int(target_pos*100)}% 倉位)'
                        else:
                            peak_price = max(peak_price, price)
                            hit_hard_stop = price < entry_price * (1 - stop_loss_pct)
                            hit_trailing_stop = price < peak_price * (1 - trailing_stop_pct)

                            if hit_hard_stop or hit_trailing_stop:
                                in_position, current_pos_size = False, 0.0
                                cost_series[i] = prev_pos * sell_cost
                                trade_signals[i] = '觸發固定止損清倉' if hit_hard_stop else '觸發追蹤止盈清倉'
                            else:
                                pos_diff = target_pos - prev_pos
                                if pos_diff > 0: cost_series[i] = pos_diff * buy_cost
                                elif pos_diff < 0: cost_series[i] = abs(pos_diff) * sell_cost
                                current_pos_size = target_pos
                    else:
                        if in_position:
                            in_position, current_pos_size = False, 0.0
                            cost_series[i] = prev_pos * sell_cost
                            trade_signals[i] = '趨勢轉空清倉'
                        else:
                            current_pos_size = 0.0

                    final_position[i] = current_pos_size

                df['Final_Position'] = final_position
                df['Transaction_Cost'] = cost_series
                df['Trade_Signal'] = trade_signals
                df['Market_Return'] = df['Close'].pct_change()
                df['Strategy_Net_Return'] = (df['Market_Return'] * df['Final_Position']) - df['Transaction_Cost']

                df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
                df['Cumulative_Net'] = (1 + df['Strategy_Net_Return']).cumprod()

                mkt_ret = (df['Cumulative_Market'].iloc[-1] - 1) * 100
                strat_ret = (df['Cumulative_Net'].iloc[-1] - 1) * 100

                col1, col2 = st.columns(2)
                col1.metric("買入持有報酬率", f"{mkt_ret:.2f}%")
                col2.metric("量化策略淨報酬率", f"{strat_ret:.2f}%", delta=f"{strat_ret - mkt_ret:.2f}%")

                # ==========================================
                # 圖表繪製：強制切換為 Pan（手勢平移/雙指縮放）
                # ==========================================
                fig = make_subplots(
                    rows=2, cols=1, 
                    shared_xaxes=True, 
                    vertical_spacing=0.12,
                    row_heights=[0.7, 0.3],
                    subplot_titles=(f"📈 {ticker} 策略與進出場點", "📊 資金持倉水位 (0% ~ 100%)")
                )

                # 買入持有
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Cumulative_Market'],
                    mode='lines', name='Buy & Hold (死抱)',
                    line=dict(color='#cccccc', width=1.5, dash='dot'),
                    hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>買入持有倍數</b>: %{y:.2f}x<extra></extra>'
                ), row=1, col=1)

                # 量化策略
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Cumulative_Net'],
                    mode='lines', name='Quant Strategy (量化)',
                    line=dict(color='#1f77b4', width=2.5),
                    hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>策略累積倍數</b>: %{y:.2f}x<extra></extra>'
                ), row=1, col=1)

                # 🟢 買入點
                buy_mask = df['Trade_Signal'].str.contains('買入建倉')
                fig.add_trace(go.Scatter(
                    x=df.index[buy_mask], y=df['Cumulative_Net'][buy_mask],
                    mode='markers', name='🟢 關鍵買入點',
                    marker=dict(symbol='triangle-up', size=12, color='#2ca02c', line=dict(width=1, color='black')),
                    customdata=df['Trade_Signal'][buy_mask],
                    hovertemplate='<b>🟢 買入日期</b>: %{x|%Y-%m-%d}<br><b>說明</b>: %{customdata}<extra></extra>'
                ), row=1, col=1)

                # 🔴 清倉點
                sell_mask = df['Trade_Signal'].str.contains('清倉')
                fig.add_trace(go.Scatter(
                    x=df.index[sell_mask], y=df['Cumulative_Net'][sell_mask],
                    mode='markers', name='🔴 風控/清倉點',
                    marker=dict(symbol='triangle-down', size=12, color='#d62728', line=dict(width=1, color='black')),
                    customdata=df['Trade_Signal'][sell_mask],
                    hovertemplate='<b>🔴 出場日期</b>: %{x|%Y-%m-%d}<br><b>原因</b>: %{customdata}<extra></extra>'
                ), row=1, col=1)

                # 持倉水位
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['Final_Position'] * 100,
                    mode='lines', name='持倉水位(%)',
                    line=dict(color='#2ca02c', width=2, shape='hv'),
                    fill='tozeroy', fillcolor='rgba(44, 160, 44, 0.15)',
                    hovertemplate='<b>日期</b>: %{x|%Y-%m-%d}<br><b>當前持倉</b>: %{y:.0f}%<extra></extra>'
                ), row=2, col=1)

                # 關鍵修改：dragmode="pan" 徹底關閉單指畫框放大，改為手勢滑動/縮放
                fig.update_layout(
                    hovermode="x unified",
                    dragmode="pan", 
                    template="plotly_white",
                    height=680,
                    margin=dict(t=80, b=40, l=40, r=30),
                    legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5),
                    xaxis2=dict(rangeslider=dict(visible=True), type="date")
                )

                # Streamlit 手機組態
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    config={
                        'scrollZoom': True,           # 開啟縮放
                        'doubleClick': 'reset',        # 雙擊還原
                        'displayModeBar': True,
                        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'zoom2d'], # 移除單指畫框按鈕
                        'responsive': True
                    }
                )

        except Exception as e:
            st.error(f"連線或計算時發生錯誤: {e}")
