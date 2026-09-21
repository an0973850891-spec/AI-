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

# --- 常見台股中文名稱對照表 ---
STOCK_NAME_MAP = {
    "2330.TW": "台積電",
    "2317.TW": "鴻海",
    "2454.TW": "聯發科",
    "00878.TW": "國泰永續高股息",
    "00881.TW": "國泰台灣5G+",
    "2603.TW": "長榮",
    "2308.TW": "台達電",
    "2382.TW": "廣達",
    "2881.TW": "富邦金",
    "2882.TW": "國泰金",
    "5351.TWO": "鈺創",
}

# --- 初始化 Session State (記憶自選股清單) ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["2330.TW", "2317.TW", "2454.TW", "00878.TW", "00881.TW"]

if 'primary_ticker' not in st.session_state:
    st.session_state.primary_ticker = "2330.TW"

# --- 側邊欄：功能選單與設定 ---
st.sidebar.header("🔍 股票查詢與自選股")

col_input, col_btn = st.sidebar.columns([3, 1])
with col_input:
    new_ticker_input = st.text_input("新增自選代號", placeholder="例如: 2308, 5351, 2603", label_visibility="collapsed")
with col_btn:
    add_clicked = st.button("➕ 新增", use_container_width=True)

if add_clicked and new_ticker_input:
    clean_input = new_ticker_input.strip().upper()
    if not clean_input.endswith((".TW", ".TWO", ".US")):
        formatted_ticker = f"{clean_input}.TW"
    else:
        formatted_ticker = clean_input

    if formatted_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(formatted_ticker)
    
    st.session_state.primary_ticker = formatted_ticker
    st.rerun()

primary_ticker = st.sidebar.selectbox(
    "選擇要檢視的股票", 
    options=st.session_state.watchlist, 
    index=st.session_state.watchlist.index(st.session_state.primary_ticker) if st.session_state.primary_ticker in st.session_state.watchlist else 0
)
st.session_state.primary_ticker = primary_ticker

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

    # 技術指標計算
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['STD'] * 2)
    df['Lower'] = df['MA20'] - (df['STD'] * 2)

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['OBV_MA9'] = df['OBV'].rolling(window=9).mean()

    n = 9
    low_min = df['Low'].rolling(window=n, min_periods=1).min()
    high_max = df['High'].rolling(window=n, min_periods=1).max()
    denominator = high_max - low_min
    df['RSV'] = np.where(denominator == 0, 50, (df['Close'] - low_min) / denominator * 100)

    return df, info, valid_symbol

# 載入股票資料
with st.spinner(f"正在從 Yahoo 股市載入 {primary_ticker} 資料與計算指標..."):
    df, stock_info, actual_symbol = load_stock_data(primary_ticker, period_map[period_option])

# --- 主畫面標題與中文名稱解析 ---
st.title("📊 台股盤後分析與技術指標系統")

if df is None or df.empty:
    st.error(f"找不到代號 `{primary_ticker}` 的資料。請確認台股代號是否正確（例如：上市請輸入 `2330` 或 `2330.TW`，上櫃請輸入 `5351` 或 `5351.TWO`）。")
else:
    chinese_name = STOCK_NAME_MAP.get(actual_symbol)
    if not chinese_name and stock_info:
        chinese_name = stock_info.get('chineseName', stock_info.get('shortName', actual_symbol))
    if not chinese_name:
        chinese_name = actual_symbol

    latest = df.iloc[-1]
    prev_close = df.iloc[-2]['Close'] if len(df) > 1 else latest['Open']
    change = latest['Close'] - prev_close
    pct_change = (change / prev_close) * 100 if prev_close != 0 else 0

    # --- AI 燈號短中長線判讀邏輯 ---
    short_signal = "🟡 中立盤整"
    if latest['Close'] > latest['MA10'] and latest['RSV'] > 50:
        short_signal = "🟢 短線偏多"
    elif latest['Close'] < latest['MA10'] and latest['RSV'] < 50:
        short_signal = "🔴 短線偏空"

    long_signal = "🟡 盤整觀望"
    obv_val = latest['OBV']
    obv_ma9 = latest['OBV_MA9'] if not pd.isna(latest['OBV_MA9']) else obv_val
    if latest['Close'] > latest['MA20'] and obv_val >= obv_ma9:
        long_signal = "🟢 中長線多頭"
    elif latest['Close'] < latest['MA20'] and obv_val < obv_ma9:
        long_signal = "🔴 中長線空頭"

    # 顯示主標題與名稱
    st.markdown(f"### 🎯 **{chinese_name}** `({actual_symbol})`")
    
    # 修正欄位寬度與排版，避免文字被擠壓換行
    col_light1, col_light2, col_space = st.columns([3, 3, 4])
    col_light1.markdown(f"**短線燈號：** {short_signal}")
    col_light2.markdown(f"**中長線燈號：** {long_signal}")

    st.markdown("---")

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

    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='K線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], name='10日均線 (MA10)', line=dict(color='orange', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], name='布林上軌', line=dict(color='rgba(250, 128, 114, 0.8)', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='布林中軌 (MA20)', line=dict(color='rgba(30, 144, 255, 0.8)', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], name='布林下軌', line=dict(color='rgba(144, 238, 144, 0.8)', width=1), fill='tonexty', fillcolor='rgba(200, 200, 200, 0.1)'), row=1, col=1)

    colors = ['#ef5350' if row['Close'] >= row['Open'] else '#26a69a' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker_color=colors), row=2, col=1)

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
    
    rename_dict = {
        'Open': '開盤',
        'High': '最高',
        'Low': '最低',
        'Close': '收盤',
        'Volume': '成交量',
        'MA10': '10日均線',
        'MA20': '布林中軌(MA20)',
        'Upper': '布林上軌',
        'Lower': '布林下軌',
        'OBV': 'OBV能量潮',
        'OBV_MA9': 'OBV9日均線',
        'RSV': 'RSV(9日)'
    }
    
    display_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'Upper', 'Lower', 'OBV', 'OBV_MA9', 'RSV'] if c in df.columns]
    df_display = df[display_cols].copy()
    
    if 'RSV' in df_display.columns:
        df_display['RSV'] = df_display['RSV'].round(2)
    if 'OBV_MA9' in df_display.columns:
        df_display['OBV_MA9'] = df_display['OBV_MA9'].round(0)
    if 'MA10' in df_display.columns:
        df_display['MA10'] = df_display['MA10'].round(2)
    if 'MA20' in df_display.columns:
        df_display['MA20'] = df_display['MA20'].round(2)
    if 'Upper' in df_display.columns:
        df_display['Upper'] = df_display['Upper'].round(2)
    if 'Lower' in df_display.columns:
        df_display['Lower'] = df_display['Lower'].round(2)

    df_display = df_display.rename(columns=rename_dict)
    df_display.index = pd.to_datetime(df_display.index).strftime('%Y-%m-%d')
    df_display.index.name = '日期'
        
    st.dataframe(df_display.tail(20).sort_index(ascending=False), use_container_width=True)
