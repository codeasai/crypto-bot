import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Model Training",
    page_icon="🤖",
    layout="wide"
)

st.title("ฝึกสอนโมเดล")

# Sidebar configuration
st.sidebar.header("การตั้งค่าโมเดล")

# เลือกประเภทโมเดล
model_type = st.sidebar.selectbox(
    "ประเภทโมเดล",
    ["Deep Q-Learning", "Policy Gradient", "Actor-Critic"]
)

# ตั้งค่าพารามิเตอร์
st.sidebar.subheader("พารามิเตอร์")
learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
batch_size = st.sidebar.slider("Batch Size", 32, 512, 64, 32)
epochs = st.sidebar.slider("Epochs", 10, 1000, 100, 10)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("ข้อมูลการฝึก")
    
    # แสดงกราฟการเรียนรู้
    st.line_chart({
        "Loss": [0.5, 0.4, 0.3, 0.2, 0.1],
        "Reward": [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    
    # แสดงความคืบหน้า
    progress = st.progress(0)
    for i in range(100):
        progress.progress(i + 1)
    
    # แสดงข้อมูลการฝึก
    training_data = pd.DataFrame({
        "Epoch": range(1, 6),
        "Loss": [0.5, 0.4, 0.3, 0.2, 0.1],
        "Reward": [0.1, 0.2, 0.3, 0.4, 0.5],
        "Accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
    })
    st.dataframe(training_data)

with col2:
    st.subheader("สถิติโมเดล")
    
    # แสดงสถิติ
    st.metric("Loss", "0.1")
    st.metric("Reward", "0.5")
    st.metric("Accuracy", "90%")
    
    # แสดงพารามิเตอร์ปัจจุบัน
    st.subheader("พารามิเตอร์ปัจจุบัน")
    st.json({
        "model_type": model_type,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    })

# ปุ่มควบคุม
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("เริ่มการฝึก", type="primary"):
        st.info("เริ่มการฝึกโมเดล...")
with col2:
    if st.button("หยุดการฝึก"):
        st.warning("หยุดการฝึกโมเดล")
with col3:
    if st.button("บันทึกโมเดล"):
        st.success("บันทึกโมเดลสำเร็จ") 