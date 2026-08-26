import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. 基本設定 (移除可能導致隱形或排版衝突的自訂 CSS)
st.title("📈 股票技術決策與AI風險分析系統")

# ==============================================================================
# 🗂️ 自選股管理機制 (使用原生元件，保證 100% 正常顯示與記憶)
# ==============================================================================
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ["5351.TWO", "2330.TW", "00881.TW", "00878.TW"]
st.sidebar.header("⭐ 自選股組合")
add_ticker = st.sidebar.text_input("➕ 輸入 4 碼代碼新增 (如: 2454.TW)").strip()
if st.sidebar.button("確認新增股票", use_container_width=True):
    if add_ticker and add_ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(add_ticker)
        st.rerun()

st.sidebar.write("📌 快速切換分析標的：")
selected_ticker = st.session_state.watchlist[0] if st.session_state.watchlist else "5351.TWO"

for t in st.session_state.watchlist:
    col_select, col_del = st.sidebar.columns(2)
    if col_select.button(f"📊 {t}", key=f"sel_{t}", use_container_width=True):
        selected_ticker = t
    if col_del.button("❌ 刪除", key=f"del_{t}", use_container_width=True):
        st.session_state.watchlist.remove(t)
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 參數設定")
ticker = st.sidebar.text_input("當前分析股票代碼", value=selected_ticker)
# 自動計算今天的 6 個月前作為預設起點
default_start = pd.to_datetime("today") - pd.DateOffset(months=6)
start_date = st.sidebar.date_input("歷史資料起點", value=default_start.date())
# 自動將結束日期鎖定在昨天以前，100% 阻絕未收盤當天無數據錯誤
end_date = st.sidebar.date_input("歷史資料終點", value=pd.to_datetime("today") - pd.Timedelta(days=1))

# ==============================================================================
# 🎯 核心日 K 數據抓取與計算區 (導入防多層索引防禦機制)
# ==============================================================================
# 強制加入 threads=False 防範 Python 3.14 線程鎖死
df_raw = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False, progress=False, threads=False)

if df_raw.empty or len(df_raw) < 20:
    st.error("❌ 無法獲取該股票的歷史日 K 線。台股上市請補 .TW (如 2330.TW)；上櫃請補 .TWO (如 5351.TWO)。")
else:
    # 雙重清洗：將回傳數據強制扁平化為標準一維 DataFrame
    df = pd.DataFrame(index=df_raw.index)
    df['Open'] = df_raw['Open'].squeeze()
    df['High'] = df_raw['High'].squeeze()
    df['Low'] = df_raw['Low'].squeeze()
    df['Close'] = df_raw['Close'].squeeze()
    df['Volume'] = df_raw['Volume'].squeeze()
    
    # 技術指標運算
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    # === 原本的均線與指標計算 ===
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # 👇 新增：布林通道 (Bollinger Bands) 計算 👇
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['BB_Up'] = df['MA20'] + (2 * df['STD20'])  # 上軌 (壓力線)
    df['BB_Low'] = df['MA20'] - (2 * df['STD20']) # 下軌 (支撐線)
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    low_9 = df['Low'].rolling(window=9).min()
    high_9 = df['High'].rolling(window=9).max()
    df['RSV'] = 100 * ((df['Close'] - low_9) / (high_9 - low_9))
    
    k_list, d_list = [50.0] * len(df), [50.0] * len(df)
    for i in range(9, len(df)):
        if not np.isnan(df['RSV'].iloc[i]):
            k_list[i] = (2/3) * k_list[i-1] + (1/3) * df['RSV'].iloc[i]
            d_list[i] = (2/3) * d_list[i-1] + (1/3) * k_list[i]
    df['K'], df['D'] = k_list, d_list

    # ==============================================================================
    # 🧠 AI 量化風險診斷大腦 (加入布林通道突破/跌破判定)
    # ==============================================================================
    latest = df.iloc[-1]
    c_close = float(latest['Close'])
    ma20 = float(latest['MA20'])
    bb_up = float(latest['BB_Up'])
    bb_low = float(latest['BB_Low'])
    k_val = float(latest['K'])
    rsi_val = float(latest['RSI'])
    
    # 1. 判定過熱與風險狀態 (KD & RSI)
    status_heat = "🟠 短線過熱" if (k_val > 78 and rsi_val > 68) else ("🟡 過熱觀察" if (k_val > 72 or rsi_val > 62) else ("🔵 低檔超賣" if (k_val < 22 and rsi_val < 32) else "⚪ 走勢平穩"))
    bias_ma20 = ((c_close - ma20) / ma20) * 100
    
    # 2. 布林通道突破與跌破判定
    if c_close > bb_up:
        status_bb = "🔥 突破上軌 (強勢/過熱)"
        bb_msg = "股價已突破布林上軌，短線動能極強，但也伴隨過熱乖離的風險，請留意高檔爆量反轉或獲利了結賣壓。"
    elif c_close < bb_low:
        status_bb = "❄️ 跌破下軌 (弱勢/超賣)"
        bb_msg = "股價已跌破布林下軌，短線表現弱勢，但也可能隨時醞釀跌深反彈，建議觀察下檔支撐與量能變化。"
    else:
        status_bb = "🔹 通道內震盪"
        bb_msg = "股價目前運行於布林通道內部，未出現極端偏離，屬於正常震盪區間。"

    # 3. 綜合風險判定與 AI 建議生成
    if bias_ma20 > 7 or c_close > bb_up:
        # 只要乖離過大 或 突破上軌，都視為高風險
        status_risk = "風險 🔴 高"
        ai_suggestion = f"🚨 **高檔風險警示**：目前技術指標處於超買區。\n\n📌 **布林通道分析**：{bb_msg}\n\n👉 **AI 綜合建議**：強烈建議分批減碼或靜待拉回，不要在此時盲目追高！"
    elif bias_ma20 < -7 or c_close < bb_low:
        # 只要乖離負值過大 或 跌破下軌，都視為低風險(超跌)
        status_risk = "風險 🟢 低"
        ai_suggestion = f"🍏 **低檔反彈契機**：股價已落入超賣低點，下檔具備超跌撐托。\n\n📌 **布林通道分析**：{bb_msg}\n\n👉 **AI 綜合建議**：可留意日 K 線帶量重新站回短期均線的落底訊號，適合尋找買點。"
    else:
        # 正常震盪
        status_risk = "風險 🟡 中"
        ai_suggestion = f"ℹ️ **中性震盪階段**：目前指標位於合理中性區。\n\n📌 **布林通道分析**：{bb_msg}\n\n👉 **AI 綜合建議**：操作上建議維持原有的投資紀律，無明確帶量突破前不用頻繁追價。"

    st.markdown(f"### 🧠 {ticker} AI 量化操作與綜合風險建議")
    
    # 原生警告艙塊
    if "🔴" in status_risk:
        st.error(ai_suggestion)
    elif "🟢" in status_risk:
        st.success(ai_suggestion)
    else:
        st.info(ai_suggestion)

    # 輸出彩色核心健康燈號 (加入布林狀態顯示)
    st.write(f"📊 **指標診斷**： {status_heat} ｜ ⚠️ **風險評級**： {status_risk} ｜ 🌀 **布林狀態**： {status_bb} ｜ 📦 **資料**： 🟢 完整 ({len(df)}天)")

    st.markdown("---")

    # ==============================================================================
    # 🗓️ 歷史技術圖表與模擬回測數據明細 (改為上下滿版排版)
    # ==============================================================================
    
    # 1. 上方：技術分析圖表
    st.markdown("##### 📊 技術分析 K 線與量化指標圖")
    # 建立專業紅綠三合一日 K 線蠟燭圖
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.55, 0.15, 0.3])
    
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='日 K 線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20月線', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker=dict(color='#ef5350')), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K值', line=dict(color='#ff5252', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D值', line=dict(color='#00e676', width=1.5)), row=3, col=1)
    
    # 稍微增加圖表高度 (從 520 加大到 650)，因為現在是全寬了，拉高會更好看
    fig.update_layout(height=650, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # 加上一條視覺分隔線
    st.markdown("---")

    # 2. 下方：量化數據明細
    st.markdown("##### 📋 最新量化數據明細")
    df_display = df[['Close', 'MA20', 'K', 'D', 'RSI']].copy().sort_index(ascending=False)
    st.dataframe(df_display.head(8), use_container_width=True, column_config={"Close": "最新收盤價", "MA20": "月線位置", "K": "K值", "D": "D值", "RSI": "RSI(14)"})
# ... (前面的 yfinance 下載與 KD 計算保留不變) ...
    
    # ==============================================================================
    # 📊 籌碼面數據擴充區 (由於 yfinance 無此數據，此處先建立 DataFrame 欄位與模擬資料)
    # 💡 建議：未來可使用 FinMind API 或 Fugle API 來取代下方的隨機模擬數據
    # ==============================================================================
    # 模擬：三大法人買賣超 (單位: 張)
    df['Inst_Net_Buy'] = np.random.randint(-5000, 5000, size=len(df)) 
    # 模擬：買賣家數差 (大於 0 代表買進家數多於賣出家數，籌碼發散；小於 0 代表籌碼集中)
    df['Broker_Diff'] = np.random.randint(-150, 150, size=len(df))

    # ... (保留原有的 AI 量化風險診斷大腦區塊不變) ...

   # ==============================================================================
    # 🗓️ 歷史技術圖表與模擬回測數據明細 (含布林通道)
    # ==============================================================================
    st.markdown("##### 📊 技術分析與籌碼面綜合 K 線圖")
    
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03, 
        row_heights=[0.45, 0.12, 0.15, 0.14, 0.14],
        subplot_titles=("", "", "", "三大法人買賣超", "買賣家數差")
    )
    
    # --- Row 1: K 線、MA20 與 布林通道 ---
    # 布林上軌 (設定為虛線或淺色)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_Up'], name='布林上軌', 
        line=dict(color='rgba(135, 206, 235, 0.6)', width=1, dash='dot')
    ), row=1, col=1)
    
    # 布林下軌 (連同填滿顏色至上軌)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['BB_Low'], name='布林下軌', 
        line=dict(color='rgba(135, 206, 235, 0.6)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(135, 206, 235, 0.1)' # 淺藍色半透明通道
    ), row=1, col=1)

    # 原始的 MA20 與 K 線
    fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20月線/中軌', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='日 K 線', increasing_line_color='#ef5350', decreasing_line_color='#26a69a'
    ), row=1, col=1)
    
    # --- Row 2: 成交量 ---
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='成交量', marker=dict(color='#ef5350')), row=2, col=1)
    
    # --- Row 3: KD 指標 ---
    fig.add_trace(go.Scatter(x=df.index, y=df['K'], name='K值', line=dict(color='#ff5252', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['D'], name='D值', line=dict(color='#00e676', width=1.5)), row=3, col=1)

    # --- Row 4: 三大法人買賣超 ---
    colors_inst = ['#ef5350' if val > 0 else '#26a69a' for val in df['Inst_Net_Buy']]
    fig.add_trace(go.Bar(x=df.index, y=df['Inst_Net_Buy'], name='法人買賣超', marker_color=colors_inst), row=4, col=1)

    # --- Row 5: 買賣家數差 ---
    colors_broker = ['#ef5350' if val > 0 else '#26a69a' for val in df['Broker_Diff']]
    fig.add_trace(go.Bar(x=df.index, y=df['Broker_Diff'], name='家數差', marker_color=colors_broker), row=5, col=1)
    
    # 圖表版面設定
    fig.update_layout(height=950, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified", xaxis_rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.2)')
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 更新量化數據明細表格，加入布林通道上下軌數值
    st.markdown("##### 📋 最新量化與籌碼數據明細")
    df_display = df[['Close', 'BB_Up', 'MA20', 'BB_Low', 'K', 'Inst_Net_Buy']].copy().sort_index(ascending=False)
    
    # 數值四捨五入，讓表格看起來更乾淨
    df_display[['BB_Up', 'MA20', 'BB_Low']] = df_display[['BB_Up', 'MA20', 'BB_Low']].round(2)

    st.dataframe(
        df_display.head(8), 
        use_container_width=True, 
        column_config={
            "Close": "最新收盤價", 
            "BB_Up": "布林上軌",
            "MA20": "月線(中軌)", 
            "BB_Low": "布林下軌",
            "K": "K值", 
            "Inst_Net_Buy": "法人買賣超(張)",
        }
    )
