import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Data Overview",
    page_icon="📊",
    layout="wide"
)

st.title("ภาพรวมข้อมูล")

# Sidebar filters
st.sidebar.header("ตัวกรองข้อมูล")
symbol = st.sidebar.selectbox(
    "เลือกคู่เหรียญ",
    ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
)

timeframe = st.sidebar.selectbox(
    "เลือกช่วงเวลา",
    ["1m", "5m", "15m", "1h", "4h", "1d"]
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("กราฟราคา")
    # ตัวอย่างข้อมูล
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    prices = pd.Series(range(100), index=dates)
    
    fig = go.Figure(data=[go.Candlestick(
        x=dates,
        open=prices,
        high=prices + 2,
        low=prices - 2,
        close=prices + 1
    )])
    
    fig.update_layout(
        title=f"{symbol} Price Chart",
        yaxis_title="Price (USDT)",
        xaxis_title="Date"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("สถิติพื้นฐาน")
    # ตัวอย่างข้อมูลสถิติ
    stats = {
        "ราคาปัจจุบัน": "45,000 USDT",
        "การเปลี่ยนแปลง 24h": "+2.5%",
        "ปริมาณการเทรด 24h": "1.2B USDT",
        "ราคาสูงสุด 24h": "46,000 USDT",
        "ราคาต่ำสุด 24h": "44,000 USDT"
    }
    
    for key, value in stats.items():
        st.metric(key, value)

# Technical Indicators
st.subheader("ตัวชี้วัดทางเทคนิค")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("RSI (14)", "65.4")
    st.metric("MACD", "Bullish")

with col2:
    st.metric("EMA (20)", "44,500")
    st.metric("SMA (50)", "44,200")

with col3:
    st.metric("Bollinger Bands", "Upper: 46,000")
    st.metric("", "Lower: 43,000")

# Market Data
st.subheader("ข้อมูลตลาด")
market_data = pd.DataFrame({
    "Exchange": ["Binance", "Coinbase", "Kraken"],
    "Price": ["45,000", "45,100", "44,900"],
    "Volume": ["500M", "300M", "200M"],
    "Spread": ["0.1%", "0.2%", "0.15%"]
})

st.dataframe(market_data, use_container_width=True) 