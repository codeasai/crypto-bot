import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
from datetime import datetime

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Live Trading",
    page_icon="💹",
    layout="wide"
)

st.title("ซื้อขายเรียลไทม์")

# Sidebar configuration
st.sidebar.header("การตั้งค่าการเทรด")

# เลือกโมเดล
model = st.sidebar.selectbox(
    "โมเดล",
    ["DQN_Model_001", "PG_Model_002", "AC_Model_003"]
)

# ตั้งค่าพารามิเตอร์
st.sidebar.subheader("พารามิเตอร์")
initial_capital = st.sidebar.number_input("เงินทุนเริ่มต้น (USDT)", 1000, 100000, 10000)
max_position = st.sidebar.slider("ตำแหน่งสูงสุด (%)", 1, 100, 10)
stop_loss = st.sidebar.slider("Stop Loss (%)", 1, 20, 5)
take_profit = st.sidebar.slider("Take Profit (%)", 1, 50, 10)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("กราฟราคาเรียลไทม์")
    
    # แสดงกราฟราคา
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
        title="BTC/USDT Price",
        yaxis_title="Price (USDT)",
        xaxis_title="Time"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("สถานะปัจจุบัน")
    
    # แสดงสถานะ
    status = {
        "สถานะ": "กำลังทำงาน",
        "ตำแหน่งปัจจุบัน": "Long",
        "ขนาดตำแหน่ง": "0.1 BTC",
        "ราคาเข้า": "45,000 USDT",
        "กำไร/ขาดทุน": "+500 USDT",
        "ROI": "+5%"
    }
    
    for key, value in status.items():
        st.metric(key, value)

# Active Orders
st.subheader("คำสั่งที่กำลังทำงาน")
orders = pd.DataFrame({
    "Time": [datetime.now()] * 3,
    "Type": ["BUY", "SELL", "BUY"],
    "Price": [45000, 46000, 44000],
    "Amount": [0.1, 0.1, 0.1],
    "Status": ["FILLED", "OPEN", "OPEN"]
})

st.dataframe(orders, use_container_width=True)

# Performance Metrics
st.subheader("เมตริกประสิทธิภาพ")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Win Rate", "65%")
    st.metric("Profit Factor", "2.5")

with col2:
    st.metric("Total Trades", "10")
    st.metric("Active Trades", "2")

with col3:
    st.metric("Daily P/L", "+1,000 USDT")
    st.metric("Monthly P/L", "+20,000 USDT")

# Control Panel
st.subheader("แผงควบคุม")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("เริ่มการเทรด", type="primary"):
        st.info("เริ่มการเทรด...")
with col2:
    if st.button("หยุดการเทรด"):
        st.warning("หยุดการเทรด")
with col3:
    if st.button("รีเซ็ต"):
        st.success("รีเซ็ตระบบ")

# System Status
st.subheader("สถานะระบบ")
status_data = pd.DataFrame({
    "Component": ["API Connection", "Data Feed", "Model", "Order Execution"],
    "Status": ["Connected", "Running", "Active", "Ready"],
    "Last Update": [datetime.now()] * 4,
    "Health": ["Good", "Good", "Good", "Good"]
})

st.dataframe(status_data, use_container_width=True) 