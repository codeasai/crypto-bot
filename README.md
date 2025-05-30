# 🤖 Crypto Trading Bot

บอทเทรดคริปโตที่ใช้ Deep Q-Learning (DQN) ในการตัดสินใจเทรด

## 📋 คุณสมบัติ

- 🧠 ใช้ Deep Q-Learning (DQN) ในการเรียนรู้กลยุทธ์การเทรด
- 📊 แสดงผลการเทรดด้วยกราฟและตาราง
- 🔄 รองรับการฝึกสอนโมเดลใหม่
- 📈 รองรับการวัดผลและ Backtest
- 🎯 รองรับการเทรดหลายคู่เหรียญและกรอบเวลา

## 🚀 การติดตั้ง

1. ติดตั้ง dependencies:
```bash
pip install -r requirements.txt
```

2. ตั้งค่า API Key ของ Binance (ถ้าต้องการดึงข้อมูลแบบ real-time):
```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"
```

## 💻 การใช้งาน

1. รัน Streamlit app:
```bash
streamlit run app.py
```

2. เลือกโหมดการทำงาน:
   - 🔄 ฝึกสอน: ฝึกสอนโมเดลใหม่
   - 📊 วัดผล: วัดผลโมเดลที่มีอยู่
   - 🔍 Backtest: ทดสอบโมเดลกับข้อมูลในอดีต

3. ตั้งค่าพารามิเตอร์:
   - สัญลักษณ์คู่เหรียญ (เช่น BTCUSDT)
   - กรอบเวลา (1m, 5m, 15m, 30m, 1h, 4h, 1d)
   - วันที่เริ่มต้นและสิ้นสุด
   - เงินทุนเริ่มต้น
   - ขนาดหน้าต่างข้อมูล

4. กดปุ่มเริ่มต้นเพื่อดำเนินการ

## 📊 ผลลัพธ์

- กราฟแสดงผลการฝึกสอน
- ตัวชี้วัดประสิทธิภาพ (กำไร, อัตราการชนะ, Sharpe Ratio)
- รายการเทรดและกำไรสะสม
- กราฟแสดงผลการ Backtest

## 📝 หมายเหตุ

- โมเดลจะถูกบันทึกในโฟลเดอร์ `outputs`
- ข้อมูลจะถูกบันทึกในโฟลเดอร์ `data/datasets`
- สามารถดูประวัติการฝึกสอนได้ในโฟลเดอร์ `outputs`

## ⚠️ คำเตือน

- การเทรดมีความเสี่ยง ควรทดสอบกับเงินจำลองก่อน
- ผลการ Backtest ไม่ได้การันตีผลลัพธ์ในอนาคต
- ควรใช้ความระมัดระวังในการตั้งค่า API Key