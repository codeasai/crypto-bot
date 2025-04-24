# Crypto Trading Bot

ระบบเทรดคริปโตอัตโนมัติพร้อม UI สำหรับควบคุมและติดตามผล

## การติดตั้ง

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. ติดตั้ง dependencies สำหรับ UI:
```bash
cd ui
npm install
```

## การใช้งาน

### 1. เริ่มต้นระบบ

```bash
python main.py
```

ตัวเลือกเพิ่มเติม:
- `--port`: กำหนดพอร์ตของ API server (ค่าเริ่มต้น: 5000)
- `--debug`: เปิดใช้งานโหมด debug
- `--config`: ระบุไฟล์ config (ค่าเริ่มต้น: bot_config.json)

ตัวอย่าง:
```bash
python main.py --port 8080 --debug
```

### 2. เริ่มต้น UI

```bash
cd ui
npm start
```

### 3. การควบคุมบอท

1. เปิดเว็บเบราว์เซอร์ไปที่ `http://localhost:3000`
2. ไปที่หน้า Bot Control
3. ใช้ปุ่ม Start/Stop เพื่อควบคุมบอท
4. ตั้งค่าต่างๆ ผ่านหน้า UI

## โครงสร้างโปรเจค

```
project/
├── main.py           # จัดการ API server
├── api.py            # จัดการ API endpoints
├── bot/              # โฟลเดอร์สำหรับ bot
│   ├── __init__.py
│   └── crypto_trading_bot.py
├── data/             # โฟลเดอร์สำหรับเก็บข้อมูล
│   └── <user_id>/    # ข้อมูลแยกตาม user
└── ui/               # โฟลเดอร์สำหรับ UI
    └── src/
        └── pages/
            └── BotControl.js
```

## การตั้งค่า

1. สร้างไฟล์ `bot_config.json`:
```json
{
  "api_key": "your_api_key",
  "api_secret": "your_api_secret",
  "is_testnet": true,
  "symbols": ["BTC/USDT", "ETH/USDT"],
  "timeframe": "5m",
  "profit_target_percentage": 0.05,
  "buy_dip_percentage": 0.10,
  "max_position_percentage": 0.20,
  "max_trade_percentage": 0.05,
  "check_interval": 60,
  "maturity_level": 1
}
```

2. ตั้งค่า API key และ API secret ในไฟล์ config

## การใช้งาน UI

1. Dashboard
   - แสดงสถานะบอท
   - แสดงราคาปัจจุบัน
   - แสดงพอร์ตโฟลิโอ
   - แสดงประวัติออเดอร์

2. Bot Control
   - ควบคุมการทำงานของบอท
   - ตั้งค่าต่างๆ
   - ดูโค้ดบอท

3. Portfolio
   - แสดงเหรียญทั้งหมดที่ถือ
   - แสดงมูลค่ารวม
   - แสดงกำไร/ขาดทุน

## การแก้ไขปัญหา

1. ถ้าไม่สามารถเชื่อมต่อกับ Exchange:
   - ตรวจสอบ API key และ API secret
   - ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
   - ตรวจสอบว่า Exchange ทำงานปกติ

2. ถ้า UI ไม่แสดงข้อมูล:
   - ตรวจสอบว่า API server ทำงานอยู่
   - ตรวจสอบการเชื่อมต่อระหว่าง UI และ API
   - ตรวจสอบ console ในเว็บเบราว์เซอร์

## การพัฒนาเพิ่มเติม

1. เพิ่มกลยุทธ์การเทรดใหม่
2. เพิ่มการวิเคราะห์ทางเทคนิค
3. เพิ่มการแจ้งเตือน
4. เพิ่มการรายงานผล 