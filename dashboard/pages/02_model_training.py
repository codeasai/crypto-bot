import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import glob
import json
import os
import shutil

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Model Training",
    page_icon="🤖",
    layout="wide"
)

st.title("ฝึกสอนโมเดล")

# ฟังก์ชันสำหรับโหลดรายการโมเดล
@st.cache_data
def load_models():
    models_path = project_root / "models" / "saved_models"
    model_info = []
    
    # ตรวจสอบโฟลเดอร์ dqn_btcusdt_5m
    dqn_path = models_path / "dqn_btcusdt_5m"
    if dqn_path.exists():
        for model_dir in dqn_path.iterdir():
            if model_dir.is_dir():
                # อ่านข้อมูลโมเดลจากไฟล์ config (ถ้ามี)
                config_file = model_dir / "config.json"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                else:
                    config = {
                        "model_type": "DQN",
                        "symbol": "BTCUSDT",
                        "timeframe": "5m",
                        "created_at": model_dir.name
                    }
                
                model_info.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "config": config
                })
    
    return model_info

# ฟังก์ชันสำหรับลบโมเดล
def delete_model(model_path: str):
    try:
        shutil.rmtree(model_path)
        st.success(f"ลบโมเดล {Path(model_path).name} สำเร็จ")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการลบโมเดล: {str(e)}")

# ฟังก์ชันสำหรับ backtest
def run_backtest(model_path: str):
    try:
        # TODO: เพิ่มโค้ดสำหรับ backtest
        st.info(f"กำลังทำ backtest โมเดล {Path(model_path).name}...")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการทำ backtest: {str(e)}")

# ฟังก์ชันสำหรับฝึกเพิ่ม
def continue_training(model_path: str, episodes: int):
    try:
        # TODO: เพิ่มโค้ดสำหรับฝึกเพิ่ม
        st.info(f"กำลังฝึกเพิ่มโมเดล {Path(model_path).name} อีก {episodes} รอบ...")
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการฝึกเพิ่ม: {str(e)}")

# โหลดรายการโมเดล
models = load_models()

# แสดงรายการโมเดลที่มีอยู่
st.subheader("โมเดลที่มีอยู่")
if models:
    for model in models:
        with st.expander(f"📁 {model['name']}"):
            st.json(model['config'])
            
            # สร้าง columns สำหรับปุ่มต่างๆ
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button(f"โหลดโมเดล", key=f"load_{model['name']}"):
                    st.success(f"โหลดโมเดล {model['name']} สำเร็จ")
            
            with col2:
                if st.button(f"ฝึกเพิ่ม", key=f"continue_{model['name']}"):
                    episodes = st.number_input("จำนวนรอบที่ต้องการฝึกเพิ่ม", min_value=1, max_value=1000, value=100, key=f"episodes_{model['name']}")
                    if st.button("ยืนยัน", key=f"confirm_continue_{model['name']}"):
                        continue_training(model['path'], episodes)
            
            with col3:
                if st.button(f"Backtest", key=f"backtest_{model['name']}"):
                    run_backtest(model['path'])
            
            with col4:
                if st.button(f"ลบโมเดล", key=f"delete_{model['name']}"):
                    if st.button("ยืนยันการลบ", key=f"confirm_delete_{model['name']}"):
                        delete_model(model['path'])
else:
    st.warning("ไม่พบโมเดลที่บันทึกไว้")

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