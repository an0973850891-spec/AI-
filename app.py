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

# --- 常見台股中文名稱與代號對照池 ---
STOCK_POOL = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2382.TW": "廣達", "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2884.TW": "玉山金",
    "3037.TW": "欣興", "2379.TW": "瑞昱", "3711.TW": "日月光投控", "2303.TW": "聯電",
    "1301.TW": "台塑", "1303.TW": "南亞", "1326.TW": "化纖", "2002.TW": "中鋼",
    "00878.TW": "國泰永續高股息", "0050.TW": "元大台灣50", "0056.TW": "元大高股息", "00881.TW": "國泰台灣5G+",
    "5351.TWO": "鈺創", "3264.TWO": "欣銓", "6182.TWO": "合晶", "5483.TWO": "中美晶"
}

# --- 初始化 Session State (記憶自選股與篩選結果) ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = list(STOCK_POOL.keys())

if 'primary_ticker' not in st.session_state:
    st.session_state.primary_ticker = "2330.TW"

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- 側邊欄：功能選單與設定 ---
st.sidebar.header("🔍 股票查詢與智慧篩選")

# 1. 獨立的代號新增輸入框與按鈕
col_input, col_btn = st.sidebar.columns([3, 1])
with col_input:
    new_ticker_input = st.text_input("新增自選代號", placeholder="例如: 2308, 5351", label_visibility="collapsed")
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

st.sidebar.markdown("---")
st.sidebar.subheader("🚀 智慧飆股掃描器")
st.sidebar.markdown("尋找 **OBV向上突破OBV9日均線** 且 **收盤價站上10日均線** 的個股。")
scan_button = st.sidebar.button("開始掃描強勢成交量個股", use_container_width=True)

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

    # 將成交量從「股」轉換為「張」 (1張 = 1000股)
    df['Volume_Zhang'] = df['Volume'] / 1000

    # 技術指標計算
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['STD'] * 2)
    df['Lower'] = df['MA20'] - (df['STD'] * 2)

    # OBV 改以張數的累積計算
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume_Zhang']).fillna(0).cumsum()
    df['OBV_MA9'] = df['OBV'].rolling(window=9).mean()

    n = 9
    low_min = df['Low'].rolling(window=n, min_periods=1).min()
    high_max = df['High'].rolling(window=n, min_periods=1).max()
    denominator = high_max - low_min
    df['RSV'] = np.where(denominator == 0, 50, (df['Close'] - low_min) / denominator * 100)

    return df, info, valid_symbol

# --- 執行掃描並存入 session_state ---
if scan_button:
    with st.spinner("正在掃描股票池中的 OBV 與 10日均線交疊向上狀態..."):
        matched_stocks = []
        progress_bar = st.progress(0)
        total_stocks = len(STOCK_POOL)
        
        for idx, (sym, cname) in enumerate(STOCK_POOL.items()):
            try:
                temp_df, _, _ = load_stock_data(sym, "3mo")
                if temp_df is not None and len(temp_df) > 10:
                    last = temp_df.iloc[-1]
                    if last['Close'] > last['MA10'] && last['OBV'] >= last['OBV_MA9']:
                        matched_stocks.append({"代號": sym, "名稱": cname, "收盤價": round(last['Close'], 2), "成交量(張)": int(last['Volume_Zhang']), "RSV": round(last['RSV'], 1)})
            except:
                pass
            progress_bar.progress((idx + 1) / total_stocks)
        
        progress_bar.empty()
        st.session_state.scan_results = matched_stocks

# --- 主畫面：顯示持久化的掃描結果 ---
if st.session_state.scan_results is not None:
    st.markdown("## 🔍 盤後強勢量價齊揚股票掃描結果")
    if len(st.session_state.scan_results) > 0:
        st.success(f"共找到 {len(st.session_state.scan_results)} 檔符合「量價齊揚 (OBV向上交叉 + 站上10日線)」的潛力股：")
        df_matched = pd.DataFrame(st.session_state.scan_results)
        st.dataframe(df_matched, use_container_width=True)
        st.info("💡 提示：掃描結果已固定保留，您可以隨時從左側選單切換代號來查看這些潛力股的詳細 K 線圖與技術指標！")
    else:
        st.warning("目前清單中沒有完全符合條件的股票。")
    st.markdown("---")

# 載入當前選擇的股票資料
with st.spinner(f"正在從 Yahoo 股市載入 {primary_ticker} 資料與計算指標..."):
    df, stock_info, actual_symbol = load_stock_data(primary_ticker, period_map[period_option])

# --- 主畫面標題與中文名稱解析 ---
st.title("📊 台股盤後分析與技術指標系統")

if df is None or df.empty:
    st.error(f"找不到代號 `{primary_ticker}` 的資料。請確認台股代號是否正確。")
else:
    chinese_name = STOCK_POOL.get(actual_symbol)
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

    # --- AI 操作建議生成邏輯 ---
    if "偏多" in short_signal and "多頭" in long_signal:
        advice = "🚀 **強勢多頭格局**：短中線動能皆強，資金持續流入（OBV向上）。操作上可沿 10 日均線偏多操作，若回測布林中軌不破可視為尋找買點的時機，注意追高風險。"
    elif "偏空" in short_signal and "空頭" in long_signal:
        advice = "⚠️ **弱勢空頭格局**：短中線均呈現回檔，賣壓較重且成交量能偏向流出。建議暫時多看少動、嚴守停損，避免過早逢低承接搶反彈。"
    elif "偏空" in short_signal or "空頭" in long_signal:
        advice = "防守為主：目前技術面呈現拉回或震盪偏空走勢，短期上檔逢壓。建議保持觀望，等待量能回穩、OBV突破均線後再行尋找介入機會。"
    elif "偏多" in short_signal:
        advice = "短線彈升：短線雖有買盤回溫跡象，但中長線仍在打底或盤整。操作上宜短打因應，嚴設停利停損，不宜過度重倉。"
    else:
        advice = "🔍 **盤整觀望格局**：目前短中線指標交錯、方向不明確。建議靜待突破訊號（如帶量突破布林上軌或 OBV 翻揚向上）再擬定進場策略。"

    # 顯示主標題與名稱
    st.markdown(f"### 🎯 **{chinese_name}** `({actual_symbol})`")
    
    col_light1, col_light2, col_space = st.columns([3, 3, 4])
    col_light1.markdown(f"**短線燈號：** {short_signal}")
    col_light2.markdown(f"**中長線燈號：** {long_signal}")

    st.info(f"💡 **AI 操作建議**：{advice}")

    st.markdown("---")

    obv_str = f"{latest['OBV'] / 10000:,.1f}萬張"
    obv_ma9_val = latest['OBV_MA9']
    obv_ma9_str = f"{obv_ma9_val / 10000:,.1f}萬張" if not pd.isna(obv_ma9_val) else "0萬張"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新收盤價", f"{latest['Close']:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
    col2.metric("成交量", f"{int(latest['Volume_Zhang']):,} 張")
    col3.metric("RSV (9日)", f"{latest['RSV']:.2f}%")
    col4.metric("OBV / OBV_MA9", f"{obv_str} / {obv_ma9_str}")

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
    fig.add_trace(go.Bar(x=df.index, y=df['Volume_Zhang'], name='成交量(張)', marker_color=colors), row=2, col=1)

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
    
    df_table = df.copy()
    df_table['Volume_Zhang_Int'] = df_table['Volume_Zhang'].round(0).astype(int)
    df_table['OBV_Wan'] = (df_table['OBV'] / 10000).round(1)
    df_table['OBV_MA9_Wan'] = (df_table['OBV_MA9'] / 10000).round(1)

    rename_dict = {
        'Open': '開盤',
        'High': '最高',
        'Low': '最低',
        'Close': '收盤',
        'Volume_Zhang_Int': '成交量(張)',
        'MA10': '10日均線',
        'MA20': '布林中軌(MA20)',
        'Upper': '布林上軌',
        'Lower': '布林下軌',
        'OBV_Wan': 'OBV能量潮(萬張)',
        'OBV_MA9_Wan': 'OBV9日均線(萬張)',
        'RSV': 'RSV(9日)'
    }
    
    display_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume_Zhang_Int', 'MA10', 'MA20', 'Upper', 'Lower', 'OBV_Wan', 'OBV_MA9_Wan', 'RSV'] if c in df_table.columns]
    df_display = df_table[display_cols].copy()
    
    if 'RSV' in df_display.columns:
        df_display['RSV'] = df_display['RSV'].round(2)
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
