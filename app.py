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

# --- 初始化 Session State (記憶自選股清單) ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["2330.TW", "2317.TW", "2454.TW", "00878.TW", "00881.TW"]

if 'primary_ticker' not in st.session_state:
    st.session_state.primary_ticker = "2330.TW"

# --- 側邊欄：功能選單與設定 ---
st.sidebar.header("🔍 股票查詢與自選股")

# 1. 獨立的代號新增輸入框與按鈕
col_input, col_btn = st.sidebar.columns([3, 1])
with col_input:
    new_ticker_input = st.text_input("新增自選代號", placeholder="例如: 2308, 5351, 2603", label_visibility="collapsed")
with col_btn:
    add_clicked = st.button("➕ 新增", use_container_width=True)

# 當按下新增按鈕時處理
if add_clicked and new_ticker_input:
    clean_input = new_ticker_input.strip().upper()
    
    # 智慧判斷後綴：若沒輸入後綴，預設先補 .TW
    if not clean_input.endswith((".TW", ".TWO", ".US")):
        # 簡單防呆：台股上市通常是 .TW，上櫃可能是 .TWO。若使用者直接輸入代號，先預設加 .TW
        formatted_ticker = f"{clean_input}.TW"
    else:
        formatted_ticker = clean_input

    if formatted_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(formatted_ticker)
    
    # 自動切換到剛新增的股票
    st.session_state.primary_ticker = formatted_ticker
    st.rerun()

# 2. 自選股清單下拉選單（支援點擊切換，並可隨時從清單中選擇）
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

# --- 資料抓取與技術指標計算函數 (加入雙後綴自動備援機制) ---
@st.cache_data(ttl=300, show_spinner=False)
def load_stock_data(symbol, period):
    # 建立嘗試清單：如果使用者輸入的代號抓不到，自動幫忙切換 .TW 或 .TWO 測試
    symbols_to_try = [symbol]
    if not symbol.endswith((".TW", ".TWO")):
        symbols_to_try = [f"{symbol}.TW", f"{symbol}.TWO"]
    elif symbol.endswith(".TW"):
        symbols_to_try = [symbol, symbol.replace(".TW", ".TWO")]
    elif symbol.endswith(".TWO"):
        symbols_to_try = [symbol, symbol.replace(".TWO", ".TW")]

    df = None
    info = {}
    valid_symbol = symbol

    for s in symbols_to_try:
        try:
            stock = yf.Ticker(s)
            temp_df = stock.history(period=period)
            if temp_df is not None and not temp_df.empty:
                df = temp_df
                info = stock.info
                valid_symbol = s
                break
        except Exception:
            continue

    if df is None or df.empty:
        return None, None, valid_symbol

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

    return df, info, valid_symbol

# 載入股票資料
with st.spinner(f"正在從 Yahoo 股市載入 {primary_ticker} 資料與計算指標..."):
    df, stock_info, actual_symbol = load_stock_data(primary_ticker, period_map[period_option])

# --- 主畫面標題 ---
st.title("📊 台股盤後分析與技術指標系統")

if df is None or df.empty:
    st.error(f"找不到代號 `{primary_ticker}` 的資料。請確認台股代號是否正確（例如：上市請輸入 `2330` 或 `2330.TW`，上櫃請輸入 `5351` 或 `5351.TWO`）。")
else:
    # 1. 即時摘要看板
    latest = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest['Open']
    change = latest['Close'] - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0

    name = stock_info.get('longName', actual_symbol) if stock_info else actual_symbol
    
    st.markdown(f"### 目前檢視：`{name}` ({actual_symbol})")
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

    # 10日均線 (MA10)
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
