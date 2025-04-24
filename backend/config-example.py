# Binance Testnet API Configuration
# ไปที่ https://testnet.binance.vision/ เพื่อสร้าง API Keys

# API Keys สำหรับ Binance Testnet
TESTNET_API_KEY = 'your_testnet_api_key_here'
TESTNET_API_SECRET = 'your_testnet_api_secret_here'

# Binance API URLs (ไม่ต้องแก้ไขส่วนนี้)
TESTNET_API_URL = 'https://testnet.binance.vision/api'  # Testnet API URL
TESTNET_STREAM_URL = 'wss://testnet.binance.vision/ws'  # Testnet WebSocket URL

"""
วิธีการใช้งาน:
1. คัดลอกไฟล์นี้และเปลี่ยนชื่อเป็น config.py
2. แก้ไข TESTNET_API_KEY และ TESTNET_API_SECRET เป็นของคุณ
3. อย่าเผยแพร่ไฟล์ config.py ที่มี API Keys จริง

การสร้าง API Keys:
1. ไปที่ https://testnet.binance.vision/
2. ล็อกอินด้วย GitHub
3. กดสร้าง API Keys ใหม่
4. เปิดสิทธิ์ที่จำเป็น:
   - Enable Reading
   - Enable Spot & Margin Trading
5. คัดลอก API Key และ Secret มาใส่ในไฟล์ config.py
""" 