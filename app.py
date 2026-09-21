import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 頁面設定 ---
st.set_page_config(
    page_title="台股盤後與技術指標分析系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 初始化 Session State (記憶自選股與目前主股票) ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["2330.TW", "2317.TW", "2454.TW", "00878.TW", "00881.TW"]

if 'primary_ticker' not in st.session_state:
    st.session_state.primary_ticker = "2330.TW"

# --- 側邊欄：功能選單與設定 ---
st.sidebar.header("🔍 股票查詢與自選股")

# 1. 獨立的代號新增輸入框與按鈕
col_input, col_btn = st.sidebar.columns([3, 1])
with col_input:
    new_ticker_input = st.text_input("新增自選代號", placeholder="例如: 5351, 2603", label_visibility="collapsed")
with col_btn:
    add_clicked = st.button("➕ 新增", use_container_width=True)

# 當按下新增按鈕時處理
if add_clicked and new_ticker_input:
    clean_input = new_ticker_input.strip().upper()
    if not clean_input.endswith((".TW", ".TWO")):
        clean_input = f"{clean_input}.TW"
    
    if clean_input not in st.session_state.watchlist:
        st.session_state.watchlist.append(clean_input)
    
    # 自動切換到剛新增的股票
    st.session_state.primary_ticker = clean_input
    st.rerun()

# 2. 自選股清單下拉選單（直接切換檢視）
primary_ticker = st.sidebar.selectbox(
    "選擇要檢視的股票", 
    options=st.session_state.watchlist, 
    index=st.session_state.watchlist.index(st.session_state.primary_ticker) if st.session_state.primary_ticker in st.session_state.watchlist else 0
)
st.session_state.primary_ticker = primary_ticker

# 時間區間選項
period_option = st.sidebar.selectbox(
    "歷史K線區間",
    options=["1個月", "3個月", "6個月", "1年", "2年", "5年"],
    index=2
)

period_map = {
    "1個月": "1mo",
    "3個月": "3mo",
    "6個月": "6mo",
    "1年": "1y",
    "2年": "2y",
    "5年": "5y"
}

# --- 資料抓取與技術指標計算函數 ---
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(symbol, period):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        info = stock.info
        if df is None or df.empty:
            return None, None
        
        # --- 計算技術指標 ---
        # 1. 價格均線 (MA10, MA20) 與 布林通道
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['STD'] = df['Close'].rolling(window=20).std()
        df['Upper'] = df['MA20'] + (df['STD'] * 2)
        df['Lower'] = df['MA20'] - (df['STD'] * 2)

        # 2. OBV 能量潮指標與 OBV 均線 (MAOBV，預設 9 日)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_MA9'] = df['OBV'].rolling(window=9).mean()

        # 3. RSV 未成熟隨機值 (9日)
        n = 9
        low_min = df['Low'].rolling(window=n, min_periods=1).min()
        high_max = df['High'].rolling(window=n, min_periods=1).max()
        denominator = high_max - low_min
        df['RSV'] = np.where(denominator == 0, 50, (df['Close'] - low_min) / denominator * 100)

        return df, info
    except Exception as e:
        return None, None

# 載入股票資料
with st.spinner(f"正在從 Yahoo 股市載入 {primary_ticker} 資料與計算指標..."):
    df, stock_info = load_stock_data(primary_ticker, period_map[period_option])

# --- 主畫面標題 ---
st.title("📊 台股盤後分析與技術指標系統")

if df is None or df.empty:
    st.error(f"找不到代號 `{primary_ticker}` 的資料或 Yahoo 股市連線逾時。請確認台股代號是否正確（例如上櫃請輸入帶 `.TWO` 的完整代號，如 `5351.TWO`）。")
else:
    # 1. 即時摘要看板
    latest = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest['Open']
    change = latest['Close'] - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0

    name = stock_info.get('longName', primary_ticker) if stock_info else primary_ticker
    
    st.markdown(f"### 目前檢視：`{name}` ({primary_ticker})")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新收盤價", f"{latest['Close']:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    col2.metric("成交量", f"{int(latest['Volume']):,}")
    col3.metric("RSV (9日)", f"{latest['RSV']:.2f}%")
    col4.metric("OBV / OBV_MA9", f"{int(latest['OBV']):,} / {int(latest['OBV_MA9']) if not pd.isna(latest['OBV_MA9']) else 0:,}")

    # --- 技術線圖繪製 ---
    st.subheader(f"📈 技術線圖 (K線 + 10日均線 + 布林軌道 + 成交量 + OBV 雙線)")
    
    fig = make_subplots(
        rows=3, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.55, 0.22, 0.22]
    )

    # 1. K線與 10日均線、布林通道
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)

    # 新增 10日均線 (MA10)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], name='10日均線 (MA10)', line=dict(color='orange', width=1.5)), row=1, col=1)
    # 布林通道上下軌與中軌
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='布林上軌', line=dict(color='rgba(250, 128, 114, 0.8)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='布林中軌 (MA20)', line=dict(color='rgba(30, 144, 255, 0.8)', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='布林下軌', line=dict(color='rgba(144, 238, 144, 0.8)', width=1), fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)'), row=1, col=1)

    # 2. 成交量
    colors = ['#ef5350' if row['Close'] >= row['Open'] else '#26a69a' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)

    # 3. OBV 雙線指標
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name='OBV 能量潮', line=dict(color='#ab63fa', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV_MA9'], name='OBV 9日均線', line=dict(color='#ffa15a', width=1.2, dash='dot')), row=3, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=700,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 數據明細表 ---
    st.subheader(f"📋 近期技術指標與交易數據明細")
    display_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'Upper', 'Lower', 'OBV', 'OBV_MA9', 'RSV'] if c in df.columns]
    df_display = df[display_cols].copy()
    if 'RSV' in df_display.columns:
        df_display['RSV'] = df_display['RSV'].round(2)
    if 'OBV_MA9' in df_display.columns:
        df_display['OBV_MA9'] = df_display['OBV_MA9'].round(0)
    if 'MA10' in df_display.columns:
        df_display['MA10'] = df_display['MA10'].round(2)
        
    st.dataframe(df_display.tail(20).sort_index(ascending=False), use_container_width=True)
