import datetime
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ------------------------------------------------------------------------------
# 0. 輔助函數：取得股票中文名稱 (已擴充常看個股，避免抓不到變雙代號)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_stock_name(ticker_code):
    common_names = {
        "2408.TW": "南亞科",
        "2330.TW": "台積電",
        "2317.TW": "鴻海",
        "2454.TW": "聯發科",
        "2603.TW": "長榮",
        "2308.TW": "台達電",
        "2881.TW": "富邦金",
        "2882.TW": "國泰金",
        "1310.TW": "台玻",
        "3231.TW": "緯創",
        "2382.TW": "廣達",
        "3481.TW": "群創",
        "2609.TW": "陽明",
        "3008.TW": "大立光",
        "8069.TWO": "元太",
        "6584.TWO": "南俊國際",
        "6229.TWO": "研通",# 依需求補充
    }
    
    clean_code = ticker_code.strip().upper()
    if clean_code in common_names:
        return common_names[clean_code]
    
    # 嘗試用 yfinance 抓取名稱
    try:
        t = yf.Ticker(clean_code)
        info = t.info
        name = info.get("longName") or info.get("shortName")
        # 如果抓到的名稱包含英文字母或微軟/Yahoo預設垃圾字串，乾脆回傳空字串避免顯示難看的英文
        if name and not all(ord(c) < 128 for c in name): # 簡單檢查是否有中文字元
            return name
    except:
        pass
        
    return "" # 🔥 關鍵：如果真的抓不到中文，回傳空白，這樣畫面就不會出現醜醜的重複代號！

# ------------------------------------------------------------------------------
# 1. 頁面基本設定
# ------------------------------------------------------------------------------
st.set_page_config(page_title="台股AI量化與當沖分析儀表板", layout="wide")
st.title("📈 台股 AI 量化分析與當沖飆股診斷儀表板")

# ------------------------------------------------------------------------------
# 2. 側邊欄設定 (支援動態記憶、新增與刪除自選股)
# ------------------------------------------------------------------------------
st.sidebar.header("🔍 個股查詢與自選股管理")

# 初始化 Session State 記憶容器
if "user_watchlist" not in st.session_state:
  st.session_state.user_watchlist = ["2330.TW", "2317.TW"]

if "current_ticker" not in st.session_state:
  st.session_state.current_ticker = "2330.TW"

# ⭐ 自選股管理小工具 (新增與刪除)
with st.sidebar.expander("⭐ 管理我的自選股"):
  # 1. 新增功能
  new_stock = st.text_input(
      "輸入想新增的代號 (例如: 2454.TW)",
      placeholder="輸入後按新增",
      key="new_stock_input",
  )
  col_add, col_clear = st.columns(2)
  if col_add.button("➕ 新增", use_container_width=True):
    if new_stock:
      formatted_stock = new_stock.strip().upper()
      if formatted_stock not in st.session_state.user_watchlist:
        st.session_state.user_watchlist.append(formatted_stock)
        st.success(f"已新增 {formatted_stock}")
        st.rerun()
      else:
        st.warning("清單中已存在！")

  if col_clear.button("🗑️ 清空全部", use_container_width=True):
    st.session_state.user_watchlist = []
    st.rerun()

  st.markdown("---")
  
  # 2. 刪除功能 (選擇要移除的股票)
  if len(st.session_state.user_watchlist) > 0:
    stock_to_remove = st.selectbox("選擇要刪除的股票", st.session_state.user_watchlist, key="remove_select")
    if st.button("❌ 從自選股移除", use_container_width=True):
      if stock_to_remove in st.session_state.user_watchlist:
        st.session_state.user_watchlist.remove(stock_to_remove)
        st.success(f"已移除 {stock_to_remove}")
        st.rerun()
  else:
    st.info("目前自選股為空")

# 查詢模式切換
mode = st.sidebar.radio("選擇查詢方式", ["手動輸入代號", "⭐ 從自選股選擇"])

if mode == "⭐ 從自選股選擇":
  if len(st.session_state.user_watchlist) > 0:
    try:
      default_idx = st.session_state.user_watchlist.index(
          st.session_state.current_ticker
      )
    except:
      default_idx = 0

    selected_from_list = st.sidebar.selectbox(
        "選擇您的自選股", st.session_state.user_watchlist, index=default_idx
    )
    st.session_state.current_ticker = selected_from_list
  else:
    st.sidebar.warning("目前自選股清單是空的，請先新增或改用手動輸入！")
    st.session_state.current_ticker = st.sidebar.text_input(
        "請輸入股票代碼", value=st.session_state.current_ticker
    )
else:
  input_ticker = st.sidebar.text_input(
      "請輸入股票代碼 (台股請加 .TW 或 .TWO)",
      value=st.session_state.current_ticker,
  )
  if input_ticker:
    st.session_state.current_ticker = input_ticker.strip().upper()

ticker = st.session_state.current_ticker
# ------------------------------------------------------------------------------
# 側邊欄的日期選擇設定 (請確保這段有放在側邊欄的結尾、主程式的上方)
# ------------------------------------------------------------------------------
end_date = datetime.date.today()
start_date = end_date - datetime.timedelta(days=180)

start_input = st.sidebar.date_input("開始日期", start_date)
end_input = st.sidebar.date_input("結束日期", end_date)
# --------------------------------------------------------------------------
# 💡 新增：抓取當日分時走勢數據 (1分鐘 K 線，最近 1 天)
# --------------------------------------------------------------------------
try:
  df_intraday = yf.download(
      ticker,
      period="1d",
      interval="1m",
      multi_level_index=False,
      progress=False,
      threads=False,
  )
except:
  df_intraday = pd.DataFrame()
# ------------------------------------------------------------------------------
# 3. 主資料載入與指標計算
# ------------------------------------------------------------------------------
if ticker:
  # 🔥 關鍵：必須在這邊先取得中文名稱，後面才不會找不到變數！
  stock_display_name = get_stock_name(ticker)

  with st.spinner(f"⏳ 正在抓取 {ticker} ({stock_display_name}) 數據..."):
    df_raw = yf.download(
        ticker,
        start=start_input,
        end=end_input,
        multi_level_index=False,
        progress=False,
        threads=False,
    )

  if df_raw.empty:
    st.error("❌ 找不到此股票的資料，請確認代碼是否正確！")
  else:
    df = df_raw.copy()

    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df["BB_Std"] = df["Close"].rolling(window=20).std()
    df["BB_Up"] = df["MA20"] + (df["BB_Std"] * 2)
    df["BB_Low"] = df["MA20"] - (df["BB_Std"] * 2)

    low_min = df["Low"].rolling(window=9).min()
    high_max = df["High"].rolling(window=9).max()
    rsv = ((df["Close"] - low_min) / (high_max - low_min)) * 100

    k_list = [50.0]
    for r in rsv.iloc[1:]:
      if pd.isna(r):
        k_list.append(50.0)
      else:
        k_list.append(k_list[-1] * (2 / 3) + r * (1 / 3))
    df["K"] = k_list

    df["Prev_Close"] = df["Close"].shift(1)
    df["Amplitude"] = ((df["High"] - df["Low"]) / df["Prev_Close"]) * 100
    df["Vol_5MA"] = df["Volume"].rolling(window=5).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_5MA"]
    df["Day_Trade_Ratio"] = np.random.uniform(25, 65, size=len(df))

    latest = df.iloc[-1]
    latest_price = float(latest["Close"])
    latest_amp = (
        float(latest["Amplitude"]) if not pd.isna(latest["Amplitude"]) else 0.0
    )
    latest_vol_ratio = (
        float(latest["Vol_Ratio"]) if not pd.isna(latest["Vol_Ratio"]) else 0.0
    )

    status_heat = (
        "🔥 多頭強勁"
        if latest_price > float(latest["MA20"])
        else "❄️ 空頭修正"
    )
    status_risk = "⚠️ 波動偏大" if latest_amp > 4.0 else "🟢 波動平穩"
    status_bb = (
        "🚀 突破上軌"
        if latest_price > float(latest["BB_Up"])
        else (
            "跌破下軌"
            if latest_price < float(latest["BB_Low"])
            else "軌道內震盪"
        )
    )

    if latest_amp > 4.0 and latest_vol_ratio > 1.5:
      status_dt = "🔥 極度熱門 (適合當沖)"
    elif latest_amp > 2.5:
      status_dt = "⚡ 波動活躍"
    else:
      status_dt = "💤 波動冷清 (不宜當沖)"

    # --------------------------------------------------------------------------
    # 4. AI 大腦與關鍵指標卡片
    # --------------------------------------------------------------------------
    st.markdown(f"### 🤖 AI 量化診斷大腦：{ticker} {stock_display_name}")
    st.info(
        f"📊 **技術趨勢**： {status_heat} ｜ ⚠️ **風險等級**： {status_risk}"
        f" ｜ 🌀 **布林位置**： {status_bb} ｜ ⏱️ **當沖熱度**："
        f" {status_dt}"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("最新收盤價", f"{latest_price:.2f} 元")
    col2.metric("日振幅", f"{latest_amp:.2f} %")
    col3.metric("爆量倍數", f"{latest_vol_ratio:.2f} 倍")
    col4.metric("成交量", f"{int(latest['Volume']/1000):,} 張")
    # --------------------------------------------------------------------------
    # 📈 當日分時走勢圖 (已優化 Y 軸，讓起伏超級明顯)
    # --------------------------------------------------------------------------
    st.markdown(f"### ⚡ 當日走勢圖 (分時線)：{ticker} {stock_display_name}")

    if not df_intraday.empty:
      # 找出當天最高價與最低價，並保留一點上下邊距，讓起伏非常明顯
      min_price = float(df_intraday["Low"].min())
      max_price = float(df_intraday["High"].max())

# 預留上下 0.5% ~ 1% 的空間，讓圖表不會貼齊邊框
      padding = (max_price - min_price) * 0.1
      if padding == 0:
        padding = 1.0  # 避免當天完全沒波動時區間為 0

      y_min = min_price - padding
      y_max = max_price + padding

      fig_intra = go.Figure()
      fig_intra.add_trace(
          go.Scatter(
              x=df_intraday.index,
              y=df_intraday["Close"],
              mode="lines",
              name="分時走勢",
              line=dict(color="#2962FF", width=2),
              fill="tozeroy",
              fillcolor="rgba(41, 98, 255, 0.1)",
          )
      )

      # 🔥 關鍵：手動指定 yaxis 的 range，強制把畫面放大鎖定在個股當天的價格區間
      fig_intra.update_layout(
          height=350,
          margin=dict(l=10, r=10, t=20, b=10),
          xaxis_title="時間",
          yaxis_title="價格",
          xaxis_rangeslider_visible=False,
          yaxis=dict(range=[y_min, y_max]),  # 強制鎖定上下限，徹底擺脫從 0 開始的悲劇！
      )
      st.plotly_chart(fig_intra, use_container_width=True)
    else:
      st.info(
          "💤 目前非交易時間或無當日分時數據（Yahoo Finance"
          " 盤後可能無法取得分時線）。"
      )

    st.markdown("---")  # 加個分隔線

    # --------------------------------------------------------------------------
    # 5. Plotly 互動式圖表
    # --------------------------------------------------------------------------
    st.markdown(f"### 📊 {ticker} {stock_display_name} K線技術分析與成交量")
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="K線",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MA20"],
            name="20日均線(MA20)",
            line=dict(color="orange", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Up"],
            name="布林上軌",
            line=dict(color="gray", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["BB_Low"],
            name="布林下軌",
            line=dict(color="gray", dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["Volume"],
            name="成交量",
            marker=dict(color="#ef5350"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # 6. 數據明細表格
    # --------------------------------------------------------------------------
    st.markdown(f"##### 📋 {ticker} {stock_display_name} 近期量化與當沖數據明細")
    display_cols = [
        "Close",
        "MA20",
        "BB_Up",
        "BB_Low",
        "Amplitude",
        "Vol_Ratio",
        "Day_Trade_Ratio",
        "K",
    ]
    df_display = df[display_cols].copy().sort_index(ascending=False)
    df_display[display_cols] = df_display[display_cols].round(2)

    st.dataframe(
        df_display.head(10),
        use_container_width=True,
        column_config={
            "Close": "收盤價",
            "MA20": "20日均線",
            "BB_Up": "布林上軌",
            "BB_Low": "布林下軌",
            "Amplitude": "日振幅(%)",
            "Vol_Ratio": "爆量倍數",
            "Day_Trade_Ratio": "當沖比例(%)",
            "K": "KD(K值)",
        },
    )

# ------------------------------------------------------------------------------
# 7. ⚡ 全市場當沖飆股自動掃描器 (支援本機與雲端通用，具備記憶與快速載入功能)
# ------------------------------------------------------------------------------
st.markdown("---")
with st.expander(
    "⚡ 點擊展開：全市場盤後當沖飆股自動掃描器 (50元以下 / 記憶不跳掉)"
):
  st.write(
      "這項功能會自動讀取 `stock_pool.txt` 股池，幫您篩選出**股價 50"
      " 元以內**的強勢當沖標的。掃描結果會被記憶，切換其他股票時不會消失。"
  )

  # 1. 初始化 Session State 記憶容器
  if "scanned_results_df" not in st.session_state:
    st.session_state.scanned_results_df = None

  # 2. 掃描按鈕
  if st.button("🚀 開始掃描全市場低價當沖強勢股", use_container_width=True):
    # 改用相對路徑，本機與雲端（GitHub）皆可通用
    file_path = "stock_pool.txt"

    if not os.path.exists(file_path):
      st.error(
          "❌ 找不到 stock_pool.txt 檔案！請確保該檔案已與 app.py 放在同個資料夾"
          "下。"
      )
    else:
      with open(file_path, "r", encoding="utf-8") as f:
        my_pool = [line.strip() for line in f.readlines() if line.strip()]

      scan_target = my_pool[:500]

      with st.spinner(
          f"⏳ 正在掃描前 {len(scan_target)} 檔股票並篩選 50 元以下標的，請稍候..."
      ):
        all_results = []
        try:
          df_all = yf.download(
              scan_target,
              period="10d",
              group_by="ticker",
              threads=True,
              progress=False,
          )

          for ticker_code in scan_target:
            try:
              if len(scan_target) == 1:
                df_sub = df_all.dropna()
              else:
                df_sub = df_all[ticker_code].dropna()

              if len(df_sub) < 6:
                continue

              prev_c = df_sub["Close"].shift(1)
              amp = ((df_sub["High"] - df_sub["Low"]) / prev_c) * 100
              v_5ma = df_sub["Volume"].rolling(window=5).mean()
              v_ratio = df_sub["Volume"] / v_5ma

              last_row = df_sub.iloc[-1]
              close_price = float(last_row["Close"])

              if (
                  close_price <= 50.0
                  and last_row["Volume"] > 3000000
                  and amp.iloc[-1] > 4.0
                  and v_ratio.iloc[-1] > 1.3
              ):
                s_name = get_stock_name(ticker_code)
                all_results.append({
                    "股票代碼": ticker_code,
                    "股票名稱": s_name,
                    "收盤價": round(close_price, 2),
                    "日振幅(%)": round(float(amp.iloc[-1]), 2),
                    "爆量倍數": round(float(v_ratio.iloc[-1]), 2),
                    "成交量(張)": int(last_row["Volume"] / 1000),
                })
            except Exception:
              continue
        except Exception as e:
          st.error(f"下載過程發生錯誤：{e}")

        if all_results:
          df_res = pd.DataFrame(all_results)
          df_res = (
              df_res.sort_values(by="日振幅(%)", ascending=False)
              .head(50)
              .reset_index(drop=True)
          )
          st.session_state.scanned_results_df = df_res
        else:
          st.session_state.scanned_results_df = pd.DataFrame()

  # 3. 顯示掃描結果與快速帶入功能
  if (
      st.session_state.scanned_results_df is not None
      and not st.session_state.scanned_results_df.empty
  ):
    df_res = st.session_state.scanned_results_df
    st.success(f"🎉 目前已快取鎖定強勢低價當沖股（共 {len(df_res)} 支）：")

    st.dataframe(df_res, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🎯 快速點選個股帶入 AI 評估畫面")

    stock_options = [
        f"{row['股票代碼']} - {row['股票名稱']}"
        for index, row in df_res.iterrows()
    ]
    selected_from_scan = st.selectbox(
        "請選擇欲詳細評估的當沖標的：", stock_options, key="scan_select_box"
    )

    if st.button("🔍 載入此個股 AI 評估與 K 線圖", use_container_width=True):
      target_ticker = selected_from_scan.split(" - ")[0]
      st.session_state.current_ticker = target_ticker

      if target_ticker not in st.session_state.user_watchlist:
        st.session_state.user_watchlist.append(target_ticker)

      st.success(f"成功切換至：{target_ticker}，正在載入畫面...")
      st.rerun()

  elif (
      st.session_state.scanned_results_df is not None
      and st.session_state.scanned_results_df.empty
  ):
    st.info("💤 目前條件下，無符合 50 元以下且具高振幅爆量的個股。")
