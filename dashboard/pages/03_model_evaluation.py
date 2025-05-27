import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import os
from datetime import datetime

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.evaluate_agent import EvaluationPipeline
from core.evaluator import ModelEvaluator
from evaluation.validation import ValidationMetrics

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

# ล้างแคชทั้งหมด
st.cache_data.clear()

st.write("start page")
project_root = Path(__file__).parent.parent.parent
st.write(f"project_root: {project_root}")

models_path = project_root / "models" / "saved_models"
dqn_path = models_path / "dqn_btcusdt_5m"
st.write(f"dqn_path: {dqn_path}")
st.write(f"dqn_path exists: {dqn_path.exists()}")
if dqn_path.exists():
    for model_dir in dqn_path.iterdir():
        st.write(f"model_dir: {model_dir}, is_dir: {model_dir.is_dir()}")
        st.write(f"best_model.keras exists: {(model_dir / 'best_model.keras').exists()}")

def get_trained_models():
    """ดึงรายการโมเดลที่เทรนไว้แล้ว"""
    models_path = project_root / "models" / "saved_models"
    model_info = []
    
    # Debug: แสดง path ที่กำลังค้นหา
    st.write(f"models_path: {models_path}")
    
    # ตรวจสอบโฟลเดอร์ dqn_btcusdt_5m
    dqn_path = models_path / "dqn_btcusdt_5m"
    st.write(f"dqn_path: {dqn_path}")
    if dqn_path.exists():
        st.write("dqn_path exists")
        for model_dir in dqn_path.iterdir():
            st.write(f"model_dir: {model_dir}")
            if (model_dir / 'best_model.keras').exists():
                st.write(f"พบโมเดล: {model_dir / 'best_model.keras'}")
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
                    "path": str(model_dir / "best_model.keras"),
                    "config": config,
                    "date": datetime.fromtimestamp(model_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                })
    else:
        st.write("dqn_path does not exist")
    
    # Debug: แสดงจำนวนโมเดลที่พบ
    st.write(f"พบโมเดลทั้งหมด: {len(model_info)} รายการ")
    
    return sorted(model_info, key=lambda x: x["date"], reverse=True)

def load_evaluation_results(model_name):
    """โหลดผลการประเมินของโมเดล"""
    eval_dir = project_root / "logs" / "evaluation" / model_name
    if not eval_dir.exists():
        return None
    
    metrics_file = eval_dir / "validation_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            return json.load(f)
    return None

def plot_metrics(metrics):
    """สร้างกราฟแสดง metrics"""
    fig = go.Figure()
    
    # Performance metrics
    perf_metrics = metrics["validation_metrics"]
    fig.add_trace(go.Bar(
        name="Performance",
        x=list(perf_metrics.keys()),
        y=list(perf_metrics.values()),
        text=[f"{v:.2f}" for v in perf_metrics.values()],
        textposition="auto",
    ))
    
    fig.update_layout(
        title="Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig

def main():
    st.title("ประเมินผลโมเดล")
    models = get_trained_models()
    st.write("DEBUG: models =", models)
    if not models:
        st.warning("ไม่พบโมเดลที่เทรนไว้")
        return
    st.subheader("รายการโมเดลที่มีอยู่")
    for model in models:
        st.write(f"DEBUG: model name = {model['name']}")
        st.json(model)

    # Sidebar
    st.sidebar.header("เลือกโมเดล")
    
    if not models:
        st.warning("ไม่พบโมเดลที่เทรนไว้")
        return
    
    # แสดงรายการโมเดลที่มีอยู่
    st.subheader("รายการโมเดลที่มีอยู่")
    for model in models:
        with st.expander(f"📁 {model['name']} ({model['date']})"):
            # แสดงรายละเอียดโมเดล
            st.json(model['config'])
            
            # สร้าง columns สำหรับปุ่มต่างๆ
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"เลือกโมเดลนี้", key=f"select_{model['name']}"):
                    st.session_state.selected_model = model
                    st.experimental_rerun()
            
            with col2:
                if st.button(f"ประเมินผล", key=f"eval_{model['name']}"):
                    with st.spinner("กำลังประเมินผล..."):
                        try:
                            pipeline = EvaluationPipeline(
                                model_path=model["path"],
                                validation_data_path=str(project_root / "data" / "datasets" / "train_test_split" / "validation_data.csv"),
                                output_dir=str(project_root / "logs" / "evaluation")
                            )
                            pipeline.run_evaluation()
                            st.success("ประเมินผลเสร็จสิ้น")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"เกิดข้อผิดพลาด: {str(e)}")
    
    # แสดงรายละเอียดโมเดลที่เลือกใน sidebar
    if "selected_model" in st.session_state:
        st.sidebar.subheader("รายละเอียดโมเดลที่เลือก")
        st.sidebar.json(st.session_state.selected_model["config"])
    
    st.subheader("ผลการทำนาย")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "selected_model" in st.session_state:
            # โหลดและแสดงผลการประเมิน
            metrics = load_evaluation_results(st.session_state.selected_model["name"])
            if metrics:
                fig = plot_metrics(metrics)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("ยังไม่มีผลการประเมิน")
        else:
            st.info("กรุณาเลือกโมเดลที่ต้องการประเมินผล")
    
    with col2:
        st.subheader("เมตริกการประเมิน")
        
        if "selected_model" in st.session_state:
            metrics = load_evaluation_results(st.session_state.selected_model["name"])
            if metrics:
                # แสดง metrics หลัก
                st.metric("Total Return", f"{metrics['validation_metrics']['total_return']:.2f}%")
                st.metric("Sharpe Ratio", f"{metrics['validation_metrics']['sharpe_ratio']:.2f}")
                st.metric("Max Drawdown", f"{metrics['validation_metrics']['max_drawdown']:.2f}%")
                st.metric("Win Rate", f"{metrics['validation_metrics']['win_rate']*100:.1f}%")
                
                # แสดง metrics เพิ่มเติม
                st.metric("Profit Factor", f"{metrics['validation_metrics']['profit_factor']:.2f}")
                st.metric("Total Trades", str(metrics['validation_metrics']['total_trades']))
                st.metric("Winning Trades", str(metrics['validation_metrics']['winning_trades']))
                st.metric("Losing Trades", str(metrics['validation_metrics']['losing_trades']))
            else:
                st.info("ยังไม่มีผลการประเมิน")
        else:
            st.info("กรุณาเลือกโมเดลที่ต้องการประเมินผล")
    
    # ปุ่มประเมินผลใหม่
    if "selected_model" in st.session_state:
        if st.button("ประเมินผลใหม่"):
            with st.spinner("กำลังประเมินผล..."):
                try:
                    pipeline = EvaluationPipeline(
                        model_path=st.session_state.selected_model["path"],
                        validation_data_path=str(project_root / "data" / "datasets" / "train_test_split" / "validation_data.csv"),
                        output_dir=str(project_root / "logs" / "evaluation")
                    )
                    pipeline.run_evaluation()
                    st.success("ประเมินผลเสร็จสิ้น")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    main() 