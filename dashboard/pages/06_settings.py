import streamlit as st
from pathlib import Path
import sys
import json

# เพิ่ม path ของโปรเจค
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide"
)

st.title("ตั้งค่า")

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
    settings_file = project_root / "config" / "settings.json"
    with open(settings_file, "w") as f:
        json.dump(settings, f, indent=4)
    
    st.success("บันทึกการตั้งค่าเรียบร้อยแล้ว") 