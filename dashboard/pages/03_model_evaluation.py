import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

st.title("ประเมินผลโมเดล")

# Sidebar
st.sidebar.header("เลือกโมเดล")
model_name = st.sidebar.selectbox(
    "โมเดล",
    ["DQN_Model_001", "PG_Model_002", "AC_Model_003"]
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("ผลการทำนาย")
    
    # แสดงกราฟผลการทำนาย
    dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
    actual = pd.Series(range(100), index=dates)
    predicted = actual + pd.Series(range(-5, 95), index=dates)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, name="Actual"))
    fig.add_trace(go.Scatter(x=dates, y=predicted, name="Predicted"))
    
    fig.update_layout(
        title="Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title="Price"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("เมตริกการประเมิน")
    
    # แสดงเมตริก
    metrics = {
        "Accuracy": "85%",
        "Precision": "0.82",
        "Recall": "0.78",
        "F1 Score": "0.80",
        "MSE": "0.15",
        "MAE": "0.12"
    }
    
    for key, value in metrics.items():
        st.metric(key, value)

# Confusion Matrix
st.subheader("Confusion Matrix")
confusion_matrix = pd.DataFrame({
    "Predicted Buy": [45, 15],
    "Predicted Sell": [10, 30]
}, index=["Actual Buy", "Actual Sell"])

st.dataframe(confusion_matrix)

# Performance Metrics
st.subheader("Performance Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Win Rate", "75%")
    st.metric("Profit Factor", "2.5")

with col2:
    st.metric("Sharpe Ratio", "1.8")
    st.metric("Max Drawdown", "-15%")

with col3:
    st.metric("Total Trades", "100")
    st.metric("Average Trade", "+2.5%")

# Detailed Analysis
st.subheader("การวิเคราะห์เชิงลึก")
analysis_data = pd.DataFrame({
    "Metric": ["Win Rate", "Profit Factor", "Sharpe Ratio", "Max Drawdown"],
    "Value": ["75%", "2.5", "1.8", "-15%"],
    "Benchmark": ["70%", "2.0", "1.5", "-20%"],
    "Status": ["Good", "Good", "Good", "Good"]
})

st.dataframe(analysis_data, use_container_width=True) 