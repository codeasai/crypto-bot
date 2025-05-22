import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import glob
from datetime import datetime, timedelta

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.data_processor import DataProcessor

st.set_page_config(
    page_title="Data Overview",
    page_icon="📊",
    layout="wide"
)

st.title("ภาพรวมข้อมูล")

# อ่านข้อมูลจากไฟล์ CSV
@st.cache_data
def load_data():
    data_path = project_root / "data" / "datasets"
    files = glob.glob(str(data_path / "*.csv"))
    return files

# โหลดรายชื่อไฟล์
files = load_data()

# แสดงรายชื่อไฟล์
st.subheader("รายชื่อไฟล์ข้อมูลที่มีอยู่")
for file in files:
    file_name = Path(file).name
    st.write(f"- {file_name}")

# แสดงข้อมูลตัวอย่างจากไฟล์แรก (ถ้ามี)
if files:
    first_file = files[0]
    df = pd.read_csv(first_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sidebar filters
    st.sidebar.header("ตัวกรองข้อมูล")
    symbol = st.sidebar.selectbox(
        "เลือกคู่เหรียญ",
        ["BTC/USDT"]  # เพิ่มคู่เหรียญอื่นๆ ตามที่มีในข้อมูล
    )

    timeframe = st.sidebar.selectbox(
        "เลือกช่วงเวลา",
        ["5m"]  # เพิ่ม timeframe อื่นๆ ตามที่มีในข้อมูล
    )

    # กรองข้อมูลตามที่เลือก
    df_filtered = df[df['symbol'] == symbol.replace('/', '')]

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
            title=f"{symbol} Price Chart",
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
        
        # ประมวลผลข้อมูล
        data_processor = DataProcessor()
        processed_data = data_processor.add_technical_indicators(df_filtered)
        processed_data = data_processor.normalize_data(processed_data)
        
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
            st.metric("", f"Lower: {lower_band.iloc[-1]:,.2f}")

else:
    st.warning("ไม่พบไฟล์ข้อมูล CSV ในโฟลเดอร์ data/datasets") 