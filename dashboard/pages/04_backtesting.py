import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Backtesting",
    page_icon="🔍",
    layout="wide"
)

st.title("ทดสอบย้อนหลัง")

# Sidebar configuration
st.sidebar.header("การตั้งค่าการทดสอบ")

# เลือกโมเดล
model = st.sidebar.selectbox(
    "โมเดล",
    ["DQN_Model_001", "PG_Model_002", "AC_Model_003"]
)

# ตั้งค่าช่วงเวลา
start_date = st.sidebar.date_input("วันที่เริ่มต้น", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("วันที่สิ้นสุด", pd.to_datetime("2024-03-01"))

# ตั้งค่าพารามิเตอร์
st.sidebar.subheader("พารามิเตอร์")
initial_capital = st.sidebar.number_input("เงินทุนเริ่มต้น (USDT)", 1000, 100000, 10000)
commission = st.sidebar.slider("ค่าธรรมเนียม (%)", 0.0, 1.0, 0.1, 0.01)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("ผลการทดสอบ")
    
    # แสดงกราฟ equity curve
    dates = pd.date_range(start=start_date, end=end_date)
    equity = pd.Series(range(len(dates)), index=dates) * 100 + initial_capital
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=equity, name="Equity"))
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Date",
        yaxis_title="Equity (USDT)"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("สรุปผล")
    
    # แสดงสถิติ
    stats = {
        "กำไรสุทธิ": "+25,000 USDT",
        "ROI": "+250%",
        "Win Rate": "65%",
        "Profit Factor": "2.5",
        "Max Drawdown": "-15%",
        "Sharpe Ratio": "1.8"
    }
    
    for key, value in stats.items():
        st.metric(key, value)

# Trade History
st.subheader("ประวัติการเทรด")
trades = pd.DataFrame({
    "Date": pd.date_range(start=start_date, periods=10),
    "Type": ["BUY", "SELL"] * 5,
    "Price": [45000, 46000, 44000, 47000, 43000, 48000, 42000, 49000, 41000, 50000],
    "Amount": [0.1] * 10,
    "P/L": [100, -50, 200, -100, 300, -150, 400, -200, 500, -250]
})

st.dataframe(trades, use_container_width=True)

# Performance Analysis
st.subheader("การวิเคราะห์ประสิทธิภาพ")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Average Win", "+200 USDT")
    st.metric("Average Loss", "-100 USDT")

with col2:
    st.metric("Largest Win", "+500 USDT")
    st.metric("Largest Loss", "-250 USDT")

with col3:
    st.metric("Total Trades", "100")
    st.metric("Average Trade", "+50 USDT")

# Risk Metrics
st.subheader("เมตริกความเสี่ยง")
risk_data = pd.DataFrame({
    "Metric": ["Value at Risk (95%)", "Expected Shortfall", "Beta", "Alpha"],
    "Value": ["-5%", "-7%", "1.2", "0.8"],
    "Benchmark": ["-4%", "-6%", "1.0", "0.0"],
    "Status": ["Good", "Good", "Good", "Good"]
})

st.dataframe(risk_data, use_container_width=True) 