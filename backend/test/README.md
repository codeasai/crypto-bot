# 🚀 คู่มือการใช้งาน Binance Bot CLI

## 📋 การใช้งานพื้นฐาน

```bash
python ccxt_api.py --action <action> [options]
```

## 🎯 Actions ที่รองรับ

1. **ดูข้อมูลบัญชี**
```bash
python ccxt_api.py --action account
```

2. **ดูราคาล่าสุด**
```bash
python ccxt_api.py --action price --symbol BTC/USDT
```

3. **ดูข้อมูลแท่งเทียน**
```bash
python ccxt_api.py --action klines --symbol BTC/USDT --interval 1h --limit 10
```

4. **ดูข้อมูลคำสั่ง**
```bash
python ccxt_api.py --action orders
```

## ⚙️ Options ที่รองรับ

| Option | คำอธิบาย | ค่าเริ่มต้น | ตัวอย่าง |
|--------|----------|------------|----------|
| `--action` | เลือกการทำงาน | (จำเป็น) | `account`, `price`, `klines`, `orders` |
| `--symbol` | คู่เหรียญ | `BTC/USDT` | `ETH/USDT`, `BNB/USDT` |
| `--interval` | ช่วงเวลาแท่งเทียน | `1h` | `1m`, `5m`, `15m`, `4h`, `1d` |
| `--limit` | จำนวนแท่งเทียน | `10` | `50`, `100`, `500` |

## 📊 ตัวอย่างการใช้งาน

1. **ดูข้อมูลบัญชี**
```bash
python ccxt_api.py --action account
```

2. **ดูราคา ETH**
```bash
python ccxt_api.py --action price --symbol ETH/USDT
```

3. **ดูแท่งเทียน 5 นาที 50 แท่ง**
```bash
python ccxt_api.py --action klines --symbol BTC/USDT --interval 5m --limit 50
```

4. **ดูแท่งเทียน 1 วัน 100 แท่ง**
```bash
python ccxt_api.py --action klines --symbol ETH/USDT --interval 1d --limit 100
```

## ⚠️ ข้อควรระวัง

1. ต้องมีไฟล์ `config/credentials.py` ที่มี API key และ secret
2. ใช้ Binance Testnet สำหรับการทดสอบ
3. รองรับเฉพาะคู่เหรียญที่มีใน Binance
4. ช่วงเวลา (interval) ต้องเป็นไปตามที่ Binance รองรับ

## 🔧 การแก้ไขปัญหา

1. **ไม่พบ API key**
   - ตรวจสอบไฟล์ `config/credentials.py`
   - ตรวจสอบการตั้งค่า API key ใน Binance

2. **ไม่สามารถเชื่อมต่อได้**
   - ตรวจสอบการเชื่อมต่ออินเทอร์เน็ต
   - ตรวจสอบการตั้งค่า testnet

3. **ไม่พบข้อมูล**
   - ตรวจสอบคู่เหรียญว่าถูกต้อง
   - ตรวจสอบช่วงเวลาว่ารองรับ 