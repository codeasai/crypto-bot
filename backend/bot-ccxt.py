import ccxt
import pandas as pd
import numpy as np
import time
from config import TESTNET_API_KEY, TESTNET_API_SECRET

class BinanceBot:
    def __init__(self, config=None):
        # ตั้งค่าเริ่มต้น
        default_config = {
            'symbol': 'BTC/USDT',
            'timeframe': '5m',
            'risk_per_trade': 0.01,  # 1% ต่อการเทรด
            'short_period': 12,
            'long_period': 26,
            'check_interval': 10,  # ตรวจสอบทุก 10 วินาที
            'max_orders': 10,  # จำนวน orders สูงสุดต่อประเภท (buy/sell)
            'min_order_value': 10,  # มูลค่าขั้นต่ำต่อ order (USDT)
        }

        # อัพเดทค่าจาก config ที่ส่งเข้ามา (ถ้ามี)
        self.config = default_config
        if config:
            self.config.update(config)

        # ตั้งค่า Binance Testnet
        self.exchange = ccxt.binance({
            'apiKey': TESTNET_API_KEY,
            'secret': TESTNET_API_SECRET,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'testnet': True,
                'adjustForTimeDifference': True,
            },
            'urls': {
                'api': {
                    'public': 'https://testnet.binance.vision/api/v3',
                    'private': 'https://testnet.binance.vision/api/v3',
                }
            }
        })
        
        # ตัวแปรสำหรับติดตามจำนวน orders
        self.order_counts = {
            'BUY': 0,
            'SELL': 0
        }
        self.last_reset_time = time.time()
        self.reset_interval = 24 * 60 * 60  # รีเซ็ตทุก 24 ชั่วโมง

    def reset_order_counts(self):
        """รีเซ็ตจำนวน orders เมื่อครบกำหนดเวลา"""
        current_time = time.time()
        if current_time - self.last_reset_time >= self.reset_interval:
            self.order_counts = {'BUY': 0, 'SELL': 0}
            self.last_reset_time = current_time
            print("\n🔄 รีเซ็ตจำนวน orders แล้ว")

    def fetch_data(self, limit=100):
        """ดึงข้อมูลแท่งเทียน"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.config['symbol'], self.config['timeframe'], limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            print(f"❌ ไม่สามารถดึงข้อมูลได้: {str(e)}")
            return None

    def calculate_ema(self, df):
        """คำนวณ EMA"""
        df['ema_12'] = df['close'].ewm(span=self.config['short_period'], adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=self.config['long_period'], adjust=False).mean()
        return df

    def check_signals(self, df):
        """เช็คสัญญาณซื้อขาย"""
        signal = None
        if df['ema_12'].iloc[-2] < df['ema_26'].iloc[-2] and df['ema_12'].iloc[-1] > df['ema_26'].iloc[-1]:
            if self.order_counts['BUY'] < self.config['max_orders']:
                signal = "BUY"
        elif df['ema_12'].iloc[-2] > df['ema_26'].iloc[-2] and df['ema_12'].iloc[-1] < df['ema_26'].iloc[-1]:
            if self.order_counts['SELL'] < self.config['max_orders']:
                signal = "SELL"
        return signal

    def calculate_position_size(self):
        """คำนวณขนาดการเทรด"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            current_price = self.exchange.fetch_ticker(self.config['symbol'])['last']
            
            # คำนวณจำนวน BTC ที่จะซื้อโดยใช้ risk_per_trade
            risk_amount = usdt_balance * self.config['risk_per_trade']
            
            # ตรวจสอบมูลค่าขั้นต่ำ
            if risk_amount < self.config['min_order_value']:
                print(f"⚠️ มูลค่าการเทรดต่ำกว่า {self.config['min_order_value']} USDT")
                return None
                
            btc_amount = risk_amount / current_price
            
            # ปัดเศษให้พอดีกับ min lot size ของ Binance
            btc_amount = round(btc_amount, 6)  # Binance ต้องการ 6 ตำแหน่ง
            
            return btc_amount
        except Exception as e:
            print(f"❌ ไม่สามารถคำนวณขนาดการเทรดได้: {str(e)}")
            return None

    def place_order(self, order_type):
        """ส่งคำสั่งซื้อขาย"""
        try:
            # ตรวจสอบจำนวน orders
            if self.order_counts[order_type] >= self.config['max_orders']:
                print(f"⚠️ ถึงขีดจำกัดจำนวน {order_type} orders แล้ว ({self.config['max_orders']})")
                return False

            amount = self.calculate_position_size()
            if amount is None:
                return False

            print(f"\n🔄 กำลังส่งคำสั่ง {order_type}...")
            print(f"💰 จำนวน: {amount} BTC")

            if order_type == "BUY":
                order = self.exchange.create_market_buy_order(self.config['symbol'], amount)
            else:  # SELL
                order = self.exchange.create_market_sell_order(self.config['symbol'], amount)

            # เพิ่มจำนวน orders
            self.order_counts[order_type] += 1
            print(f"✅ คำสั่งสำเร็จ: {order['id']}")
            print(f"📊 จำนวน {order_type} ที่เหลือ: {self.config['max_orders'] - self.order_counts[order_type]}")
            return True

        except Exception as e:
            print(f"❌ ไม่สามารถส่งคำสั่งได้: {str(e)}")
            return False

    def run(self):
        """เริ่มการทำงานของ Bot"""
        print(f"\n🤖 เริ่มต้นการทำงานของ Bot...")
        print(f"📊 คู่เทรด: {self.config['symbol']}")
        print(f"⏱️ Timeframe: {self.config['timeframe']}")
        print(f"💰 Risk per trade: {self.config['risk_per_trade'] * 100}%")
        print(f"⚡ ตรวจสอบทุก: {self.config['check_interval']} วินาที")
        print(f"🔒 จำกัด Orders: {self.config['max_orders']} ต่อประเภท")
        print(f"💵 มูลค่าขั้นต่ำ: {self.config['min_order_value']} USDT")

        while True:
            try:
                # รีเซ็ตจำนวน orders ถ้าถึงเวลา
                self.reset_order_counts()

                # ดึงและวิเคราะห์ข้อมูล
                df = self.fetch_data()
                if df is not None:
                    df = self.calculate_ema(df)
                    signal = self.check_signals(df)

                    # แสดงสถานะปัจจุบัน
                    current_price = df['close'].iloc[-1]
                    print(f"\n⏰ {pd.Timestamp.now()}")
                    print(f"💵 ราคา: ${current_price:,.2f}")
                    print(f"📈 EMA{self.config['short_period']}: ${df['ema_12'].iloc[-1]:,.2f}")
                    print(f"📉 EMA{self.config['long_period']}: ${df['ema_26'].iloc[-1]:,.2f}")
                    print(f"📊 Orders - BUY: {self.order_counts['BUY']}/{self.config['max_orders']}, SELL: {self.order_counts['SELL']}/{self.config['max_orders']}")

                    # ทำการซื้อขายตามสัญญาณ
                    if signal:
                        print(f"\n🎯 พบสัญญาณ {signal}!")
                        self.place_order(signal)

                # รอจนกว่าจะถึงรอบถัดไป
                time.sleep(self.config['check_interval'])

            except KeyboardInterrupt:
                print("\n\n🛑 กำลังปิดการทำงานของ Bot...")
                break
            except Exception as e:
                print(f"\n❌ เกิดข้อผิดพลาด: {str(e)}")
                time.sleep(self.config['check_interval'])

if __name__ == "__main__":
    # ตัวอย่างการกำหนดค่า config (ไม่จำเป็นถ้าใช้ค่าเริ่มต้น)
    custom_config = {
        'check_interval': 10,  # ตรวจสอบทุก 10 วินาที
        'max_orders': 10,      # จำกัด 10 orders ต่อประเภท
        'min_order_value': 10  # มูลค่าขั้นต่ำ 10 USDT
    }
    
    bot = BinanceBot(custom_config)
    bot.run() 