import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 頁面設定 ---
st.set_page_config(
    page_title="台股盤後與智慧量價掃描系統",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 擴充台股熱門掃描標的池 ---
STOCK_POOL = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2382.TW": "廣達", "3231.TW": "緯創", "2356.TW": "英業達", "6669.TW": "緯穎",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2618.TW": "長榮航",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2884.TW": "玉山金",
    "3037.TW": "欣興", "2379.TW": "瑞昱", "3711.TW": "日月光投控", "2303.TW": "聯電",
    "3017.TW": "奇鋐", "8210.TW": "勤誠", "2421.TW": "建準", "3661.TW": "世芯-KY",
    "3529.TW": "力旺", "1513.TW": "中興電", "1519.TW": "華城", "1605.TW": "華新",
    "1301.TW": "台塑", "1303.TW": "南亞", "2002.TW": "中鋼", "00878.TW": "國泰永續高股息",
    "0050.TW": "元大台灣50", "0056.TW": "元大高股息", "00881.TW": "國泰台灣5G+",
    "5351.TWO": "鈺創", "3264.TWO": "欣銓", "6182.TWO": "合晶", "5483.TWO": "中美晶"
}

# --- 初始化 Session State ---
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = list(STOCK_POOL.keys())

if 'primary_ticker' not in st.session_state:
    st.session_state.primary_ticker = "2330.TW"

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- 側邊欄：功能選單與設定 ---
st.sidebar.header("🔍 股票查詢與進階篩選")

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
st.sidebar.subheader("🚀 專業進階選股掃描器")
st.sidebar.markdown("""
**篩選條件：**
1. 🌊 **流動性**：成交量 $\ge$ 1,000 張
2. 📈 **動能**：OBV向上交叉 + 站上10日線
3. 🎯 **型態**：當日收紅K 突破近20日平台前高
""")
scan_button = st.sidebar.button("開始多條件智慧掃描", use_container_width=True)

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

    df['Volume_Zhang'] = df['Volume'] / 1000

    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD'] = df['Close'].rolling(window=20).std()
    df['Upper'] = df['MA20'] + (df['STD'] * 2)
    df['Lower'] = df['MA20'] - (df['STD'] * 2)

    df['High_20'] = df['High'].shift(1).rolling(window=20).max()

    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume_Zhang']).fillna(0).cumsum()
    df['OBV_MA9'] = df['OBV'].rolling(window=9).mean()

    n = 9
    low_min = df['Low'].rolling(window=n, min_periods=1).min()
    high_max = df['High'].rolling(window=n, min_periods=1).max()
    denominator = high_max - low_min
    df['RSV'] = np.where(denominator == 0, 50, (df['Close'] - low_min) / denominator * 100)

    return df, info, valid_symbol

# --- 執行多條件智慧掃描 ---
if scan_button:
    with st.spinner("正在執行多條件深度掃描（流動性、OBV、10日線、紅K突破前高）..."):
        matched_stocks = []
        progress_bar = st.progress(0)
        total_stocks = len(STOCK_POOL)
        
        for idx, (sym, cname) in enumerate(STOCK_POOL.items()):
            try:
                temp_df, _, _ = load_stock_data(sym, "3mo")
                if temp_df is not None and len(temp_df) > 20:
                    last = temp_df.iloc[-1]
                    
                    cond_volume = last['Volume_Zhang'] >= 1000
                    cond_obv_ma = (last['Close'] > last['MA10']) and (last['OBV'] >= last['OBV_MA9'])
                    is_red_k = last['Close'] > last['Open']
                    is_breakout = last['Close'] >= last['High_20']
                    
                    if cond_volume and cond_obv_ma and is_red_k and is_breakout:
                        matched_stocks.append({
                            "代號": sym, 
                            "名稱": cname, 
                            "收盤價": round(last['Close'], 2), 
                            "成交量(張)": int(last['Volume_Zhang']), 
                            "RSV": round(last['RSV'], 1),
                            "突破型態": "🔥 紅K突破前高"
                        })
            except:
                pass
            progress_bar.progress((idx + 1) / total_stocks)
        
        progress_bar.empty()
        st.session_state.scan_results = matched_stocks

# --- 主畫面：顯示掃描結果 ---
if st.session_state.scan_results is not None:
    st.markdown("## 🚀 專業多條件選股掃描結果（流動性 + OBV + 10日線 + 突破前高）")
    if len(st.session_state.scan_results) > 0:
        st.success(f"共找到 {len(st.session_state.scan_results)} 檔符合嚴格多頭突破條件的強勢標的：")
        df_matched = pd.DataFrame(st.session_state.scan_results)
        st.dataframe(df_matched, use_container_width=True)
        st.info("💡 提示：點擊左側選單切換代號，即可直接檢視這些突破股的支撐壓力與技術線圖！")
    else:
        st.warning("目前清單中沒有完全符合「成交量1000張以上 + 站上10日線 + OBV向上 + 紅K突破20日高點」的個股。")
    st.markdown("---")

# 載入當前選擇的股票資料
with st.spinner(f"正在從 Yahoo 股市載入 {primary_ticker} 資料與計算指標..."):
    df, stock_info, actual_symbol = load_stock_data(primary_ticker, period_map[period_option])

# --- 主畫面標題與技術分析呈現 ---
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

    # 走勢支撐與壓力計算
    support_1 = latest['Lower']
    support_2 = latest['MA10']
    resistance_1 = latest['Upper']
    resistance_2 = df['High'].rolling(20).max().iloc[-1]
    current_price = latest['Close']

    # 燈號判讀
    short_signal = "🟡 中立盤整"
    if current_price > latest['MA10'] and latest['RSV'] > 50:
        short_signal = "🟢 短線偏多"
    elif current_price < latest['MA10'] and latest['RSV'] < 50:
        short_signal = "🔴 短線偏空"

    long_signal = "🟡 盤整觀望"
    obv_val = latest['OBV']
    obv_ma9 = latest['OBV_MA9'] if not pd.isna(latest['OBV_MA9']) else obv_val
    if current_price > latest['MA20'] and obv_val >= obv_ma9:
        long_signal = "🟢 中長線多頭"
    elif current_price < latest['MA20'] and obv_val < obv_ma9:
        long_signal = "🔴 中長線空頭"

    # --- 操盤解說：入手價與短中長期策略運算 ---
    if "偏多" in short_signal and "多頭" in long_signal:
        entry_price = f"約 **{support_2:.2f} ~ {current_price:.2f} 元**（貼近 10 日均線附近或強勢整理區間分批承接）"
        short_term_strategy = f"短線動能強勢，上檔直指前高壓力 **{resistance_2:.2f} 元**。若量能續強可續抱，守穩 10 日線不破短多格局不變。"
        long_term_strategy = f"中長線資金持續流入（OBV向上且站穩布林中軌）。波段目標可看布林上軌 **{resistance_1:.2f} 元** 以上，中期防守點可設在布林中軌（MA20）。"
    elif "偏空" in short_signal and "空頭" in long_signal:
        entry_price = f"暫不建議積極進場，若欲搶短需等待量縮止跌或回測強支撐 **{support_1:.2f} 元** 附近再觀察。"
        short_term_strategy = f"短線賣壓重，反彈若無法站回 10 日線 **({support_2:.2f})** 易受壓回測，短打宜嚴守停損。"
        long_term_strategy = f"中長線偏空，能量潮（OBV）呈現流出。建議空手觀望或進行保守資產配置，避免躁進摸底。"
    else:
        entry_price = f"建議於區間下緣 **{support_2:.2f} ~ {support_1:.2f} 元** 尋找低接機會，或等待突破再順勢操作。"
        short_term_strategy = f"短線處於盤整震盪，上下空間有限，應避免追高殺低。"
        long_term_strategy = f"中長線待成交量與方向明確化（帶量突破布林上軌或跌破支撐），再調整部位大小。"

    # 顯示主標題與名稱
    st.markdown(f"### 🎯 **{chinese_name}** `({actual_symbol})`")
    
    col_light1, col_light2, col_space = st.columns([3, 3, 4])
    col_light1.markdown(f"**短線燈號：** {short_signal}")
    col_light2.markdown(f"**中長線燈號：** {long_signal}")

    # 新增：操盤解說專屬區塊
    with st.expander("💡 **【AI 專業操盤解說與進出策略】**", expanded=True):
        st.markdown(f"""
- 📍 **建議入手價**：{entry_price}
- ⚡ **短期策略**：{short_term_strategy}
- 🌊 **中長期策略**：{long_term_strategy}
        """)

    st.markdown(f"📍 **當前關鍵價位** | 短線支撐：`{support_2:.2f}`元 | 強力支撐(布林下軌)：`{support_1:.2f}`元 | 上檔壓力(布林上軌)：`{resistance_1:.2f}`元 | 前高壓力：`{resistance_2:.2f}`元")

    st.markdown("---")

    obv_str = f"{latest['OBV'] / 10000:,.1f}萬張"
    obv_ma9_val = latest['OBV_MA9']
    obv_ma9_str = f"{obv_ma9_val / 10000:,.1f}萬張" if not pd.isna(obv_ma9_val) else "0萬張"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新收盤價", f"{current_price:.2f}", f"{change:+.2f} ({pct_change:+.2f}%)")
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
