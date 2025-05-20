# Crypto Trading Bot ที่ใช้ Reinforcement Learning

บอทซื้อขายสินทรัพย์คริปโตอัตโนมัติที่ใช้เทคนิค Reinforcement Learning ในการเรียนรู้และปรับกลยุทธ์การเทรดเพื่อเพิ่มประสิทธิภาพในการสร้างผลตอบแทนที่ปรับด้วยความเสี่ยง (risk-adjusted returns)

## คุณสมบัติ

- ใช้ Deep Q-Network (DQN) เพื่อเรียนรู้กลยุทธ์การเทรดที่เหมาะสม
- รองรับการฝึกสอน (training) ด้วยข้อมูลย้อนหลัง
- รองรับการทดสอบย้อนหลัง (backtesting) เพื่อประเมินประสิทธิภาพของโมเดล
- รองรับการเทรดแบบเรียลไทม์ผ่าน Exchange API
- ใช้ตัวชี้วัดทางเทคนิคต่างๆ เช่น MACD, RSI, Bollinger Bands เป็นต้น
- มีระบบจัดการความเสี่ยงแบบอัตโนมัติด้วย stop loss และ take profit

## การติดตั้ง

โคลนโปรเจกต์:
```bash
git clone https://github.com/yourusername/crypto-trading-bot.git
cd crypto-trading-bot
```

### ติดตั้งแพคเกจที่จำเป็น:

```bash
pip install -r requirements.txt
```
### ตั้งค่าโครงสร้างโปรเจกต์:

```bash
python main.py setup
```
### สร้างไฟล์ config.py
```bash
cat > config/config.py << EOL
"""
ไฟล์การตั้งค่าหลักสำหรับระบบ RL-Agent Crypto Trading
"""

import os
from pathlib import Path

# พาธหลักของโปรเจค
BASE_DIR = Path(__file__).resolve().parent.parent

# การตั้งค่า Binance API
BINANCE_CONFIG = {
    'test_net': True,  # ตั้งเป็น False เมื่อต้องการเทรดบนตลาดจริง
    'base_url': 'https://testnet.binance.vision/api',  # URL สำหรับ testnet
    'live_url': 'https://api.binance.com/api',         # URL สำหรับตลาดจริง
    'wss_testnet': 'wss://testnet.binance.vision/ws',  # WebSocket สำหรับ testnet
    'wss_live': 'wss://stream.binance.com:9443/ws',    # WebSocket สำหรับตลาดจริง
}

# การตั้งค่าสำหรับการเทรด
TRADING_CONFIG = {
    'symbol': 'BTCUSDT',          # คู่เหรียญที่ต้องการเทรด
    'interval': '1h',             # ช่วงเวลาของแท่งเทียน (1m, 5m, 15m, 1h, 4h, 1d)
    'lookback_window': 48,        # จำนวนแท่งเทียนย้อนหลังที่ใช้ตัดสินใจ
    'max_position': 0.1,          # สัดส่วนสูงสุดของเงินทุนที่สามารถใช้ต่อการเทรดหนึ่งครั้ง
    'stop_loss_pct': 0.05,        # เปอร์เซ็นต์ stop loss
    'take_profit_pct': 0.15,      # เปอร์เซ็นต์ take profit
    'commission_fee': 0.001,      # ค่าธรรมเนียมการซื้อขาย (0.1%)
}

# การตั้งค่าสำหรับการเก็บข้อมูล
DATA_CONFIG = {
    'data_dir': os.path.join(BASE_DIR, 'data', 'datasets'),  # ที่เก็บข้อมูล
    'historical_days': 365,       # จำนวนวันย้อนหลังที่ต้องการเก็บข้อมูล
    'technical_indicators': [     # ตัวชี้วัดทางเทคนิคที่ต้องการใช้
        'sma_10', 'sma_30', 'ema_10', 'ema_30', 
        'rsi_14', 'macd', 'bbands', 'atr_14'
    ],
    'price_features': [           # ข้อมูลราคาที่ต้องการใช้
        'open', 'high', 'low', 'close', 'volume'
    ],
}

# การตั้งค่าสำหรับ Reinforcement Learning
RL_CONFIG = {
    'algorithm': 'PPO',           # อัลกอริทึมที่ใช้ (DQN, PPO, SAC, TD3)
    'gamma': 0.99,                # discount factor
    'learning_rate': 0.0001,      # อัตราการเรียนรู้
    'batch_size': 64,             # ขนาด batch
    'buffer_size': 100000,        # ขนาด replay buffer
    'train_episodes': 10000,      # จำนวน episodes ในการฝึกสอน
    'eval_frequency': 10,         # ความถี่ในการประเมินผลระหว่างฝึกสอน (ทุก n episodes)
    'model_save_frequency': 100,  # ความถี่ในการบันทึกโมเดล (ทุก n episodes)
    'output_dir': os.path.join(BASE_DIR, 'outputs'),  # ที่เก็บผลลัพธ์การฝึกสอน
}

# การตั้งค่าสำหรับสภาพแวดล้อมการซื้อขาย
ENV_CONFIG = {
    'initial_balance': 10000,     # เงินทุนเริ่มต้น (USDT)
    'reward_function': 'sharpe',  # ฟังก์ชัน reward ที่ใช้ (sharpe, sortino, return, custom)
    'action_space': 'discrete',   # ประเภทของ action space (discrete, continuous)
    'discrete_actions': [         # การกระทำที่เป็นไปได้สำหรับ discrete action space
        'buy_100pct', 'buy_50pct', 'buy_25pct', 'hold', 'sell_25pct', 'sell_50pct', 'sell_100pct'
    ],
    'max_episodes_length': 180,   # ความยาวสูงสุดของ episode (จำนวนขั้นตอน)
}

# การตั้งค่า Logger
LOGGER_CONFIG = {
    'log_dir': os.path.join(BASE_DIR, 'logs'),  # ที่เก็บไฟล์ log
    'level': 'INFO',               # ระดับการ log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# การตั้งค่า Risk Management
RISK_CONFIG = {
    'max_drawdown_limit': 0.2,      # Maximum Drawdown สูงสุดที่ยอมรับได้
    'volatility_scaling': True,     # ปรับขนาดการเทรดตามความผันผวนของตลาด
    'daily_risk_limit': 0.02,       # ความเสี่ยงสูงสุดต่อวัน (2% ของพอร์ต)
    'max_trades_per_day': 10,       # จำนวนการเทรดสูงสุดต่อวัน
    'min_trade_interval': 3600,     # ระยะเวลาขั้นต่ำระหว่างการเทรด (วินาที)
}
EOL
```

### สร้างไฟล์ credentials.py
```bash
cat > config/credentials.py << EOL
"""
ไฟล์เก็บ API key (ไม่ควร push ขึ้น git)
"""

# Binance API Keys
# ขอได้จาก https://testnet.binance.vision/ สำหรับ testnet
# หรือจาก https://www.binance.com/en/my/settings/api-management สำหรับบัญชีจริง

API_KEY = 'YOUR_API_KEY'
API_SECRET = 'YOUR_API_SECRET'
EOL
```
### สร้างไฟล์ .gitignore

### ไฟล์สำคัญอื่น ๆ
```
data/data_collector.py
data/feature_engineering.py
utils/metrics.py
environment/trading_env.py
utils/data_processor.py
models/dqn_agent.py
train.py
backtest.py
live_trading/live_trading.py
main.py
README.md
```

## การใช้งาน
### 1. เตรียมข้อมูล
ดาวน์โหลดข้อมูลประวัติราคา (historical price data) จาก Exchange หรือแหล่งข้อมูลอื่นๆ และเก็บไว้ในโฟลเดอร์ data/ โดยใช้รูปแบบไฟล์ตามนี้:
```
data/{SYMBOL}_{TIMEFRAME}_{STARTDATE}_{ENDDATE}.csv
เช่น:
data/BTCUSDT_1h_20200101_20221231.csv
```
### 2. ฝึกสอนโมเดล
```bash
python main.py train --symbol BTCUSDT --timeframe 1h --start_date 2020-01-01 --end_date 2022-12-31 --episodes 1000
```
โมเดลที่ฝึกสอนแล้วจะถูกบันทึกไว้ในโฟลเดอร์ outputs/ พร้อมกับกราฟแสดงผลการฝึกสอน

### 3. ทดสอบย้อนหลัง
```bash
python main.py backtest --model_path outputs/BTCUSDT_1h_20230101_123456/best_model.h5 --symbol BTCUSDT --timeframe 1h --start_date 2023-01-01 --end_date 2023-12-31
```
ผลการทดสอบย้อนหลังจะถูกบันทึกไว้ในโฟลเดอร์ backtest_results/ พร้อมกับกราฟแสดงผลการเทรด
### 4. เทรดแบบเรียลไทม์
```bash 
python main.py live --model_path outputs/BTCUSDT_1h_20230101_123456/best_model.h5 --symbol BTCUSDT --timeframe 15m --exchange binance --api_key YOUR_API_KEY --api_secret YOUR_API_SECRET --duration 24
```
ประวัติการเทรดและข้อมูลสถานะต่างๆ จะถูกบันทึกไว้ในโฟลเดอร์ live_trading_logs/

## โครงสร้างโปรเจค
```
crypto-trading-bot/
├── backtest.py              # สคริปต์สำหรับการทดสอบย้อนหลัง
├── backtest_results/        # ผลลัพธ์การทดสอบย้อนหลัง
├── data/                   # ข้อมูลประวัติราคา
├── environment/            # สภาพแวดล้อมการเทรด
│   └── trading_env.py     # สภาพแวดล้อมการเทรดสำหรับ Reinforcement Learning
├── live_trading.py         # สคริปต์สำหรับการเทรดแบบเรียลไทม์
├── live_trading_logs/      # ล็อกการเทรดแบบเรียลไทม์
├── main.py                 # จุดเริ่มต้นของโปรแกรม
├── outputs/                # ผลลัพธ์การฝึกสอนและโมเดล
├── README.md               # เอกสารนี้
├── requirements.txt        # แพคเกจที่จำเป็น
├── train.py                # สคริปต์สำหรับการฝึกสอน
└── utils/                  # ยูทิลิตี้ต่างๆ
    └── data_processor.py  # ตัวประมวลผลข้อมูล
```

## หมายเหตุสำคัญเกี่ยวกับความเสี่ยง
การเทรดสินทรัพย์คริปโตมีความเสี่ยงสูง คุณอาจสูญเสียเงินทั้งหมดที่ลงทุน บอทนี้เป็นเพียงเครื่องมือช่วยในการเทรดเท่านั้น ไม่ใช่คำแนะนำทางการเงิน กรุณาศึกษาและทดสอบอย่างละเอียดก่อนใช้งานจริง

เราได้สร้างองค์ประกอบหลักทั้งหมดของระบบ Crypto Trading Bot ที่ใช้ Reinforcement Learning ตามที่ระบุในเอกสารที่คุณแชร์มา โครงสร้างระบบประกอบด้วย:
```
1. **Environment (สภาพแวดล้อม)** 
- ไฟล์ `environment/trading_env.py`
2. **Data Processor (ตัวประมวลผลข้อมูล)** 
- ไฟล์ `utils/data_processor.py` 
3. **Agent (ตัวแทน AI)** 
- ไฟล์ `models/dqn_agent.py`
4. **Training (การฝึกสอน)** 
- ไฟล์ `train.py`
5. **Backtesting (การทดสอบย้อนหลัง)** 
- ไฟล์ `backtest.py`
6. **Live Trading (การเทรดแบบเรียลไทม์)** 
- ไฟล์ `live_trading.py`
7. **Main Program (โปรแกรมหลัก)** 
- ไฟล์ `main.py`
```
ระบบนี้รองรับฟังก์ชันทั้งหมดตามที่ระบุในเอกสาร รวมถึงการใช้ตัวชี้วัดทางเทคนิคต่างๆ การจัดการความเสี่ยง และการประเมินประสิทธิภาพของโมเดล

คุณสามารถใช้โค้ดนี้เป็นพื้นฐานในการพัฒนาบอทเทรดของคุณเอง และปรับแต่งพารามิเตอร์ต่างๆ เพื่อให้เหมาะกับกลยุทธ์การเทรดของคุณ