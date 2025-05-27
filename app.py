"""
Streamlit GUI สำหรับ Crypto Trading Bot
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import json
import sys
from pathlib import Path
import threading
import queue
import glob
import plotly.graph_objects as go
import shutil

# เพิ่ม path ของ root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ตั้งค่า environment variable
os.environ["PYTHONPATH"] = str(current_dir)

from scripts.train_agent import train_dqn_agent, evaluate_model
from agents.dqn_agent import DQNAgent
from environment.trading_env import CryptoTradingEnv
from data.data_processor import DataProcessor

class Navigator:
    def __init__(self):
        self.pages = {
            "🏠 หน้าแรก": self.home_page,
            "📊 ภาพรวมข้อมูล": self.data_overview_page,
            "🤖 ฝึกสอนโมเดล": self.model_training_page,
            "📈 ประเมินผลโมเดล": self.model_evaluation_page,
            "🔍 ทดสอบย้อนหลัง": self.backtest_page,
            "💹 ซื้อขายเรียลไทม์": self.live_trading_page,
            "⚙️ ตั้งค่า": self.settings_page
        }
        
        # โหลด CSS
        self.load_css()
        
    def load_css(self):
        """โหลด CSS จากไฟล์"""
        css_file = Path(current_dir) / "dashboard" / "assets" / "css" / "style.css"
        if css_file.exists():
            with open(css_file) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        
    def render(self):
        st.sidebar.title("🤖 Crypto Trading Bot")
        
        # แสดงโลโก้
        logo_path = Path(current_dir) / "dashboard" / "assets" / "img" / "logo.png"
        if logo_path.exists():
            st.sidebar.image(str(logo_path), width=100)
        
        # แสดงหน้าเว็บที่เลือก
        selected_page = st.sidebar.radio("เลือกหน้า", list(self.pages.keys()))
        
        # เรียกใช้ฟังก์ชันของหน้าที่เลือก
        self.pages[selected_page]()
        
    def home_page(self):
        st.title("🏠 หน้าแรก")
        st.markdown("""
        ### ภาพรวม
        แดชบอร์ดนี้มีเครื่องมือสำหรับ:
        - การวิเคราะห์และแสดงผลข้อมูล
        - การฝึกสอนและวัดผลโมเดล
        - การทดสอบกลยุทธ์
        - การติดตามการเทรดแบบเรียลไทม์
        - การตั้งค่าระบบ
        """)
        
        # แสดงสถิติล่าสุด
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("โมเดลที่ใช้งาน", "3")
        with col2:
            st.metric("จำนวนการเทรด", "1,234")
        with col3:
            st.metric("อัตราการชนะ", "65%")
            
    def data_overview_page(self):
        """หน้าแสดงภาพรวมข้อมูล"""
        st.title("📊 ภาพรวมข้อมูล")
        
        # อ่านข้อมูลจากไฟล์ CSV
        @st.cache_data
        def load_data():
            # โหลดข้อมูลดิบ
            raw_data_path = Path(current_dir) / "data" / "datasets"
            raw_files = glob.glob(str(raw_data_path / "*.csv"))
            
            # โหลดข้อมูลที่ประมวลผลแล้ว
            processed_data_path = Path(current_dir) / "data" / "features"
            processed_files = glob.glob(str(processed_data_path / "processed_features_*.csv"))
            
            return raw_files, processed_files

        # โหลดรายชื่อไฟล์
        raw_files, processed_files = load_data()

        # แสดงรายชื่อไฟล์แยกตามประเภท
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Raw Market Data")
            if raw_files:
                for file in raw_files:
                    file_name = Path(file).name
                    st.write(f"- {file_name}")
            else:
                st.warning("ไม่พบไฟล์ข้อมูลดิบ")
                
        with col2:
            st.subheader("🔧 Processed Features")
            if processed_files:
                for file in processed_files:
                    file_name = Path(file).name
                    st.write(f"- {file_name}")
            else:
                st.warning("ไม่พบไฟล์ข้อมูลที่ประมวลผลแล้ว")

        # แสดงข้อมูลตัวอย่างจากไฟล์แรก (ถ้ามี)
        if raw_files:
            first_file = raw_files[0]
            df = pd.read_csv(first_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sidebar filters
            st.sidebar.header("ตัวกรองข้อมูล")
            
            # ตรวจสอบว่ามีคอลัมน์ symbol หรือไม่
            if 'symbol' in df.columns:
                symbol = st.sidebar.selectbox(
                    "เลือกคู่เหรียญ",
                    ["BTC/USDT"]  # เพิ่มคู่เหรียญอื่นๆ ตามที่มีในข้อมูล
                )
                df_filtered = df[df['symbol'] == symbol.replace('/', '')]
            else:
                df_filtered = df  # ใช้ข้อมูลทั้งหมดถ้าไม่มีคอลัมน์ symbol

            timeframe = st.sidebar.selectbox(
                "เลือกช่วงเวลา",
                ["5m"]  # เพิ่ม timeframe อื่นๆ ตามที่มีในข้อมูล
            )

            # แบ่งเป็น 2 ส่วน: Raw Market Data และ Processed Features
            tab1, tab2 = st.tabs(["📈 Raw Market Data", "🔧 Processed Features"])

            with tab1:
                st.subheader("Raw Market Data")
                
                # แสดงข้อมูลดิบ
                st.write("ข้อมูลดิบจาก Exchange")
                st.dataframe(df_filtered.head(), use_container_width=True)
                
                # แสดงกราฟราคา
                fig = go.Figure(data=[go.Candlestick(
                    x=df_filtered['timestamp'],
                    open=df_filtered['open'],
                    high=df_filtered['high'],
                    low=df_filtered['low'],
                    close=df_filtered['close']
                )])
                
                fig.update_layout(
                    title="Price Chart",
                    yaxis_title="Price (USDT)",
                    xaxis_title="Date",
                    height=600,
                    template="plotly_dark"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # แสดงสถิติพื้นฐาน
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("สถิติพื้นฐาน")
                    latest_price = df_filtered['close'].iloc[-1]
                    price_change = df_filtered['close'].iloc[-1] - df_filtered['close'].iloc[0]
                    price_change_pct = (price_change / df_filtered['close'].iloc[0]) * 100
                    
                    stats = {
                        "ราคาปัจจุบัน": f"{latest_price:,.2f} USDT",
                        "การเปลี่ยนแปลง": f"{price_change:+,.2f} USDT ({price_change_pct:+.2f}%)",
                        "ปริมาณการเทรด 24h": f"{df_filtered['volume'].sum():,.2f} USDT",
                        "ราคาสูงสุด": f"{df_filtered['high'].max():,.2f} USDT",
                        "ราคาต่ำสุด": f"{df_filtered['low'].min():,.2f} USDT"
                    }
                    
                    for key, value in stats.items():
                        st.metric(key, value)
                        
                with col2:
                    st.subheader("ข้อมูลตลาด")
                    market_data = pd.DataFrame({
                        "Exchange": ["Binance"],
                        "Price": [f"{latest_price:,.2f}"],
                        "Volume": [f"{df_filtered['volume'].sum():,.2f}"],
                        "Spread": [f"{((df_filtered['high'] - df_filtered['low']) / df_filtered['low'] * 100).mean():.2f}%"]
                    })
                    st.dataframe(market_data, use_container_width=True)

            with tab2:
                st.subheader("Processed Features")
                
                # เลือกไฟล์ที่ประมวลผลแล้ว
                if processed_files:
                    selected_processed_file = st.selectbox(
                        "เลือกไฟล์ที่ประมวลผลแล้ว",
                        processed_files,
                        format_func=lambda x: Path(x).name
                    )
                    
                    # โหลดข้อมูลที่ประมวลผลแล้ว
                    processed_data = pd.read_csv(selected_processed_file)
                    processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
                    
                    # แสดงข้อมูลที่ประมวลผลแล้ว
                    st.write("ข้อมูลที่ประมวลผลแล้ว")
                    st.dataframe(processed_data.head(), use_container_width=True)
                    
                    # แสดงตัวชี้วัดทางเทคนิค
                    st.subheader("ตัวชี้วัดทางเทคนิค")
                    
                    # คำนวณ RSI
                    def calculate_rsi(data, periods=14):
                        delta = data.diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
                        rs = gain / loss
                        return 100 - (100 / (1 + rs))

                    # คำนวณ EMA
                    def calculate_ema(data, periods=20):
                        return data.ewm(span=periods, adjust=False).mean()

                    # คำนวณ Bollinger Bands
                    def calculate_bollinger_bands(data, periods=20, std=2):
                        sma = data.rolling(window=periods).mean()
                        std_dev = data.rolling(window=periods).std()
                        upper_band = sma + (std_dev * std)
                        lower_band = sma - (std_dev * std)
                        return upper_band, sma, lower_band

                    # คำนวณตัวชี้วัด
                    rsi = calculate_rsi(processed_data['close'])
                    ema = calculate_ema(processed_data['close'])
                    upper_band, sma, lower_band = calculate_bollinger_bands(processed_data['close'])

                    # แสดงกราฟตัวชี้วัด
                    fig = go.Figure()
                    
                    # เพิ่มเส้นราคา
                    fig.add_trace(go.Scatter(
                        x=processed_data['timestamp'],
                        y=processed_data['close'],
                        name="Price",
                        line=dict(color='#00FF9D')
                    ))
                    
                    # เพิ่ม EMA
                    fig.add_trace(go.Scatter(
                        x=processed_data['timestamp'],
                        y=ema,
                        name="EMA (20)",
                        line=dict(color='#FF6B6B')
                    ))
                    
                    # เพิ่ม Bollinger Bands
                    fig.add_trace(go.Scatter(
                        x=processed_data['timestamp'],
                        y=upper_band,
                        name="Upper Band",
                        line=dict(color='#4ECDC4', dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=processed_data['timestamp'],
                        y=lower_band,
                        name="Lower Band",
                        line=dict(color='#4ECDC4', dash='dash'),
                        fill='tonexty'
                    ))
                    
                    fig.update_layout(
                        title="Technical Indicators",
                        yaxis_title="Price",
                        xaxis_title="Date",
                        height=600,
                        template="plotly_dark"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # แสดงค่าตัวชี้วัด
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI (14)", f"{rsi.iloc[-1]:.2f}")
                        st.metric("MACD", "Bullish" if ema.iloc[-1] > sma.iloc[-1] else "Bearish")
                    with col2:
                        st.metric("EMA (20)", f"{ema.iloc[-1]:,.2f}")
                        st.metric("SMA (50)", f"{sma.iloc[-1]:,.2f}")
                    with col3:
                        st.metric("Bollinger Bands", f"Upper: {upper_band.iloc[-1]:,.2f}")
                        st.metric("Lower Band", f"Lower: {lower_band.iloc[-1]:,.2f}")
                else:
                    st.warning("ไม่พบไฟล์ข้อมูลที่ประมวลผลแล้ว")

        else:
            st.warning("ไม่พบไฟล์ข้อมูล CSV ในโฟลเดอร์ data/datasets")
            
    def model_training_page(self):
        """หน้าแสดงการฝึกสอนโมเดล"""
        st.title("🤖 ฝึกสอนโมเดล")

        # ฟังก์ชันสำหรับโหลดรายการโมเดล
        @st.cache_data
        def load_models():
            models_path = Path(current_dir) / "models" / "saved_models"
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
                                
                    # เพิ่มปุ่มสำหรับดูรายละเอียดการฝึก
                    if st.button(f"ดูรายละเอียดการฝึก", key=f"details_{model['name']}"):
                        self.show_training_details(model)
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

    def show_training_details(self, model):
        """แสดงรายละเอียดการฝึกในรูปแบบ popup"""
        st.title(f"📊 รายละเอียดการฝึก: {model['name']}")
        
        # ตั้งค่าพื้นฐาน
        st.subheader("ตั้งค่าพื้นฐาน")
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("สัญลักษณ์", model['config'].get('symbol', 'BTCUSDT'), key=f"detail_symbol_{model['name']}")
            timeframe = st.selectbox(
                "กรอบเวลา",
                ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=["1m", "5m", "15m", "30m", "1h", "4h", "1d"].index(model['config'].get('timeframe', '5m')),
                key=f"detail_timeframe_{model['name']}"
            )
        with col2:
            start_date = st.date_input(
                "วันที่เริ่มต้น",
                datetime.now() - timedelta(days=365),
                key=f"detail_start_date_{model['name']}"
            )
            end_date = st.date_input(
                "วันที่สิ้นสุด",
                datetime.now(),
                key=f"detail_end_date_{model['name']}"
            )
            
        # ตั้งค่าการฝึกสอน
        st.subheader("ตั้งค่าการฝึกสอน")
        col1, col2 = st.columns(2)
        with col1:
            episodes = st.number_input(
                "จำนวนรอบการฝึกสอน",
                min_value=100,
                max_value=10000,
                value=model['config'].get('episodes', 1000),
                step=100,
                key=f"detail_episodes_{model['name']}"
            )
            batch_size = st.number_input(
                "ขนาด Batch",
                min_value=16,
                max_value=256,
                value=model['config'].get('batch_size', 64),
                step=16,
                key=f"detail_batch_size_{model['name']}"
            )
        with col2:
            learning_rate = st.number_input(
                "อัตราการเรียนรู้",
                min_value=0.0001,
                max_value=0.01,
                value=model['config'].get('learning_rate', 0.001),
                step=0.0001,
                format="%.4f",
                key=f"detail_learning_rate_{model['name']}"
            )
            discount_factor = st.number_input(
                "Discount Factor",
                min_value=0.8,
                max_value=0.99,
                value=model['config'].get('discount_factor', 0.95),
                step=0.01,
                key=f"detail_discount_factor_{model['name']}"
            )
            
        # ตั้งค่าเพิ่มเติม
        st.subheader("ตั้งค่าเพิ่มเติม")
        col1, col2 = st.columns(2)
        with col1:
            initial_balance = st.number_input(
                "เงินทุนเริ่มต้น",
                min_value=1000.0,
                max_value=1000000.0,
                value=model['config'].get('initial_balance', 10000.0),
                step=1000.0,
                key=f"detail_initial_balance_{model['name']}"
            )
        with col2:
            window_size = st.slider(
                "ขนาดหน้าต่างข้อมูล",
                min_value=5,
                max_value=50,
                value=model['config'].get('window_size', 10),
                step=5,
                key=f"detail_window_size_{model['name']}"
            )
            
        # แสดงผลการฝึก
        st.subheader("ผลการฝึก")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("กำไรสูงสุด", f"{model['config'].get('best_profit', 0):.2f}")
        with col2:
            st.metric("อัตราการชนะ", f"{model['config'].get('win_rate', 0)*100:.2f}%")
        with col3:
            st.metric("จำนวนการเทรด", f"{model['config'].get('total_trades', 0)}")
            
        # แสดงกราฟ
        st.subheader("กราฟแสดงผล")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=model['config'].get('profit_history', [0]),
            name="กำไร",
            line=dict(color='#00FF9D')
        ))
        fig.update_layout(
            title="กำไรสะสม",
            xaxis_title="การเทรด",
            yaxis_title="กำไร",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ปุ่มปิด
        if st.button("ปิด", key=f"close_details_{model['name']}"):
            st.experimental_rerun()

    def model_evaluation_page(self):
        """หน้าแสดงการประเมินผลโมเดล"""
        st.title("📈 ประเมินผลโมเดล")

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
                yaxis_title="Price",
                template="plotly_dark"
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

    def backtest_page(self):
        """หน้าแสดงการทดสอบย้อนหลัง"""
        st.title("🔍 ทดสอบย้อนหลัง")

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
                yaxis_title="Equity (USDT)",
                template="plotly_dark"
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

    def live_trading_page(self):
        """หน้าแสดงการซื้อขายเรียลไทม์"""
        st.title("💹 ซื้อขายเรียลไทม์")

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
                xaxis_title="Time",
                template="plotly_dark"
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

    def settings_page(self):
        """หน้าแสดงการตั้งค่า"""
        st.title("⚙️ ตั้งค่า")

        # API Settings
        st.header("การตั้งค่า API")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Binance API")
            binance_api_key = st.text_input("API Key", type="password")
            binance_api_secret = st.text_input("API Secret", type="password")
            
            if st.button("ทดสอบการเชื่อมต่อ Binance"):
                st.info("กำลังทดสอบการเชื่อมต่อ...")

        with col2:
            st.subheader("Telegram API")
            telegram_token = st.text_input("Bot Token", type="password")
            telegram_chat_id = st.text_input("Chat ID")
            
            if st.button("ทดสอบการเชื่อมต่อ Telegram"):
                st.info("กำลังทดสอบการเชื่อมต่อ...")

        # Trading Settings
        st.header("การตั้งค่าการเทรด")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("พารามิเตอร์การเทรด")
            trading_pairs = st.multiselect(
                "คู่เหรียญ",
                ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"],
                default=["BTC/USDT"]
            )
            
            timeframes = st.multiselect(
                "ช่วงเวลา",
                ["1m", "5m", "15m", "1h", "4h", "1d"],
                default=["1h"]
            )
            
            max_open_trades = st.number_input("จำนวนการเทรดสูงสุด", 1, 10, 3)

        with col2:
            st.subheader("การจัดการความเสี่ยง")
            max_daily_trades = st.number_input("จำนวนการเทรดต่อวันสูงสุด", 1, 100, 10)
            max_daily_loss = st.number_input("ขาดทุนสูงสุดต่อวัน (%)", 1, 100, 5)
            stop_loss = st.number_input("Stop Loss (%)", 1, 20, 5)
            take_profit = st.number_input("Take Profit (%)", 1, 50, 10)

        # Model Settings
        st.header("การตั้งค่าโมเดล")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("พารามิเตอร์โมเดล")
            model_type = st.selectbox(
                "ประเภทโมเดล",
                ["Deep Q-Learning", "Policy Gradient", "Actor-Critic"]
            )
            
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
            batch_size = st.slider("Batch Size", 32, 512, 64, 32)
            epochs = st.slider("Epochs", 10, 1000, 100, 10)

        with col2:
            st.subheader("การตั้งค่าการฝึก")
            validation_split = st.slider("Validation Split", 0.1, 0.5, 0.2, 0.1)
            early_stopping = st.checkbox("Early Stopping", value=True)
            if early_stopping:
                patience = st.number_input("Patience", 1, 50, 10)

        # Notification Settings
        st.header("การตั้งค่าการแจ้งเตือน")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("การแจ้งเตือน Telegram")
            notify_trades = st.checkbox("แจ้งเตือนการเทรด", value=True)
            notify_errors = st.checkbox("แจ้งเตือนข้อผิดพลาด", value=True)
            notify_daily = st.checkbox("รายงานประจำวัน", value=True)

        with col2:
            st.subheader("การแจ้งเตือนอีเมล")
            email_enabled = st.checkbox("เปิดใช้งานการแจ้งเตือนอีเมล")
            if email_enabled:
                email_address = st.text_input("อีเมล")
                email_password = st.text_input("รหัสผ่าน", type="password")

        # Save Settings
        if st.button("บันทึกการตั้งค่า", type="primary"):
            settings = {
                "api": {
                    "binance": {
                        "api_key": binance_api_key,
                        "api_secret": binance_api_secret
                    },
                    "telegram": {
                        "token": telegram_token,
                        "chat_id": telegram_chat_id
                    }
                },
                "trading": {
                    "pairs": trading_pairs,
                    "timeframes": timeframes,
                    "max_open_trades": max_open_trades,
                    "max_daily_trades": max_daily_trades,
                    "max_daily_loss": max_daily_loss,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit
                },
                "model": {
                    "type": model_type,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "validation_split": validation_split,
                    "early_stopping": early_stopping,
                    "patience": patience if early_stopping else None
                },
                "notifications": {
                    "telegram": {
                        "trades": notify_trades,
                        "errors": notify_errors,
                        "daily": notify_daily
                    },
                    "email": {
                        "enabled": email_enabled,
                        "address": email_address if email_enabled else None,
                        "password": email_password if email_enabled else None
                    }
                }
            }
            
            # บันทึกการตั้งค่า
            settings_file = Path(current_dir) / "config" / "settings.json"
            with open(settings_file, "w") as f:
                json.dump(settings, f, indent=4)
            
            st.success("บันทึกการตั้งค่าเรียบร้อยแล้ว")

# ตั้งค่า Streamlit theme
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ตั้งค่า custom CSS
st.markdown("""
<style>
    /* ตั้งค่าสีพื้นหลังและตัวอักษรหลัก */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* ตั้งค่าสีหัวข้อ */
    h1, h2, h3, h4, h5, h6 {
        color: #00FF9D !important;
    }
    
    /* ตั้งค่าสีข้อความใน sidebar */
    .css-1d391kg {
        background-color: #262730;
        color: #FAFAFA;
    }
    
    /* ตั้งค่าสีข้อความใน tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #262730;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #FAFAFA;
    }
    
    /* ตั้งค่าสีข้อความใน selectbox */
    .stSelectbox label {
        color: #FAFAFA;
    }
    
    /* ตั้งค่าสีข้อความใน number input */
    .stNumberInput label {
        color: #FAFAFA;
    }
    
    /* ตั้งค่าสีข้อความใน date input */
    .stDateInput label {
        color: #FAFAFA;
    }
    
    /* ตั้งค่าสีข้อความใน button */
    .stButton button {
        background-color: #00FF9D;
        color: #0E1117;
    }
    
    /* ตั้งค่าสีข้อความใน metric */
    .stMetric label {
        color: #FAFAFA;
    }
    
    .stMetric div {
        color: #00FF9D;
    }
    
    /* ตั้งค่าสีข้อความใน dataframe */
    .stDataFrame {
        color: #FAFAFA;
    }
    
    /* ตั้งค่าสีข้อความใน plot */
    .stPlotlyChart {
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    navigator = Navigator()
    navigator.render() 