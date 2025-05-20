"""
สคริปต์สำหรับการเทรดสินทรัพย์คริปโตแบบเรียลไทม์ด้วยโมเดล Reinforcement Learning
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow import keras
import argparse
import time
import ccxt
import logging
from typing import Dict, List, Tuple, Optional

# นำเข้าโมดูลที่เราสร้างขึ้น
from utils.data_processor import DataProcessor
from models.dqn_agent import DQNAgent


# ตั้งค่าการบันทึกล็อก
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("live_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class LiveTradingBot:
    """
    บอทสำหรับการเทรดสินทรัพย์คริปโตแบบเรียลไทม์
    """
    
    def __init__(self, 
                 model_path: str,
                 exchange_id: str,
                 api_key: str,
                 api_secret: str,
                 symbol: str,
                 timeframe: str,
                 window_size: int = 10,
                 max_position: float = 1.0,
                 risk_per_trade: float = 0.02,
                 stop_loss_pct: float = 0.03,
                 take_profit_pct: float = 0.06,
                 use_limit_orders: bool = True,
                 order_timeout: int = 60,
                 data_dir: str = 'data'):
        """
        กำหนดค่าเริ่มต้นของบอทเทรดแบบเรียลไทม์
        
        Args:
            model_path (str): เส้นทางไปยังไฟล์โมเดล
            exchange_id (str): ID ของ Exchange ('binance', 'coinbase', 'ftx', etc.)
            api_key (str): API Key
            api_secret (str): API Secret
            symbol (str): สัญลักษณ์คู่เหรียญ (เช่น 'BTC/USDT')
            timeframe (str): กรอบเวลา (เช่น '1h', '15m')
            window_size (int): ขนาดหน้าต่างข้อมูลย้อนหลัง
            max_position (float): ตำแหน่งสูงสุดที่อนุญาตให้ถือครอง (1.0 = 100% ของพอร์ต)
            risk_per_trade (float): ความเสี่ยงต่อการเทรดหนึ่งครั้ง (เป็นเศษส่วนของพอร์ต)
            stop_loss_pct (float): เปอร์เซ็นต์ stop loss
            take_profit_pct (float): เปอร์เซ็นต์ take profit
            use_limit_orders (bool): ใช้คำสั่ง limit order แทน market order
            order_timeout (int): เวลาที่รอให้คำสั่งสำเร็จก่อนยกเลิก (วินาที)
            data_dir (str): โฟลเดอร์ที่เก็บข้อมูล
        """
        self.model_path = model_path
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.max_position = max_position
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.use_limit_orders = use_limit_orders
        self.order_timeout = order_timeout
        self.data_dir = data_dir
        
        # ตั้งค่า Exchange
        self._setup_exchange()
        
        # โหลดโมเดล
        self._load_model()
        
        # ตั้งค่าตัวประมวลผลข้อมูล
        self.data_processor = DataProcessor(data_dir=data_dir)
        
        # ข้อมูลสถานะการเทรด
        self.current_position = 0.0
        self.current_price = 0.0
        self.current_balance = 0.0
        self.current_market_value = 0.0
        self.last_action_time = None
        self.trades_history = []
        self.orders = []
        
        # ลงทะเบียนตัวแปรสำหรับ stop loss และ take profit
        self.stop_loss_price = None
        self.take_profit_price = None
        
        # ความถี่ในการอัพเดทข้อมูล
        self.timeframe_seconds = self._convert_timeframe_to_seconds(timeframe)
        
        # โฟลเดอร์สำหรับบันทึกข้อมูล
        self.log_dir = f"live_trading_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"บอทเทรด {self.symbol} เริ่มต้นเรียบร้อยแล้ว ด้วยกรอบเวลา {self.timeframe}")
    
    def _setup_exchange(self):
        """
        ตั้งค่าการเชื่อมต่อกับ Exchange
        """
        try:
            # สร้างอินสแตนซ์ของ Exchange
            exchange_class = getattr(ccxt, self.exchange_id)
            self.exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,  # เพื่อหลีกเลี่ยงข้อจำกัดอัตราการใช้งาน API
                'options': {
                    'defaultType': 'spot'  # ใช้งานในโหมด spot trading
                }
            })
            
            # ตรวจสอบการเชื่อมต่อ
            self.exchange.load_markets()
            logger.info(f"เชื่อมต่อกับ {self.exchange_id} สำเร็จแล้ว")
            
            # ตรวจสอบว่ามีคู่เหรียญที่ต้องการหรือไม่
            if self.symbol not in self.exchange.symbols:
                raise ValueError(f"ไม่พบคู่เหรียญ {self.symbol} ใน {self.exchange_id}")
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเชื่อมต่อกับ Exchange: {str(e)}")
            raise
    
    def _load_model(self):
        """
        โหลดโมเดลสำหรับการทำนาย
        """
        try:
            # โหลดโมเดล
            self.model = keras.models.load_model(self.model_path)
            logger.info(f"โหลดโมเดลจาก {self.model_path} สำเร็จแล้ว")
            
            # ตั้งค่าตัวแทน DQN
            # หมายเหตุ: เราต้องทราบขนาดของ state_size ที่โมเดลใช้
            # สำหรับตัวอย่างนี้เราจะสมมติว่ารู้ค่า
            state_size = self.model.layers[0].input_shape[1]  # ขนาด input ของโมเดล
            action_size = 7  # ตามที่กำหนดในการฝึกสอน
            
            self.agent = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                learning_rate=0.001,  # ไม่สำคัญเพราะเราไม่ได้ฝึกสอน
                batch_size=32
            )
            
            # โหลดน้ำหนักของโมเดล
            self.agent.model = self.model
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
            raise
    
    def _convert_timeframe_to_seconds(self, timeframe: str) -> int:
        """
        แปลงกรอบเวลาให้เป็นวินาที
        
        Args:
            timeframe (str): กรอบเวลา (เช่น '1m', '5m', '1h', '1d')
            
        Returns:
            int: จำนวนวินาที
        """
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 60 * 60
        elif unit == 'd':
            return value * 60 * 60 * 24
        else:
            raise ValueError(f"ไม่รองรับกรอบเวลา {timeframe}")
    
    def _fetch_current_market_data(self) -> pd.DataFrame:
        """
        ดึงข้อมูลตลาดล่าสุด
        
        Returns:
            pd.DataFrame: DataFrame ที่มีข้อมูลตลาดล่าสุด
        """
        try:
            # คำนวณเวลาเริ่มต้นสำหรับการดึงข้อมูล
            # เราต้องการข้อมูลย้อนหลังที่เพียงพอสำหรับตัวชี้วัดทางเทคนิคและ window size
            limit = 200  # จำนวนแท่งเทียนที่ต้องการ
            
            # ดึงข้อมูล OHLCV
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=limit
            )
            
            # แปลงเป็น DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # อัพเดทราคาปัจจุบัน
            self.current_price = df.iloc[-1]['close']
            
            # เพิ่มตัวชี้วัดทางเทคนิค
            df_with_indicators = self.data_processor.add_technical_indicators(df)
            
            # ปรับข้อมูลให้เป็นปกติ
            columns_to_exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # ปรับคอลัมน์ที่ต้องการให้เป็นปกติ
            columns_to_normalize = [col for col in df_with_indicators.columns if col not in columns_to_exclude]
            
            for col in columns_to_normalize:
                if df_with_indicators[col].min() != df_with_indicators[col].max():
                    df_with_indicators[col] = (df_with_indicators[col] - df_with_indicators[col].min()) / (df_with_indicators[col].max() - df_with_indicators[col].min())
                else:
                    df_with_indicators[col] = 0.5
                    
            return df_with_indicators
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลตลาด: {str(e)}")
            return None
    
    def _update_account_info(self):
        """
        อัพเดทข้อมูลบัญชี
        """
        try:
            # ดึงข้อมูลบัญชี
            balance = self.exchange.fetch_balance()
            
            # แยกสกุลเงินหลักและสกุลเงินรอง
            base_currency, quote_currency = self.symbol.split('/')
            
            # ดึงจำนวนเงินที่มีอยู่
            base_balance = float(balance[base_currency]['free']) if base_currency in balance else 0
            quote_balance = float(balance[quote_currency]['free']) if quote_currency in balance else 0
            
            # คำนวณมูลค่าทั้งหมด
            self.current_position = base_balance
            self.current_balance = quote_balance
            self.current_market_value = quote_balance + (base_balance * self.current_price)
            
            logger.info(f"อัพเดทข้อมูลบัญชี: {base_currency}: {base_balance}, {quote_currency}: {quote_balance}, "
                       f"ราคาปัจจุบัน: {self.current_price}, มูลค่ารวม: {self.current_market_value}")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอัพเดทข้อมูลบัญชี: {str(e)}")
    
    def _create_state(self, market_data: pd.DataFrame) -> np.ndarray:
        """
        สร้าง state vector สำหรับโมเดล
        
        Args:
            market_data (pd.DataFrame): ข้อมูลตลาด
            
        Returns:
            np.ndarray: state vector
        """
        # ใช้ข้อมูลตลาดล่าสุดตามขนาดหน้าต่าง
        market_features = []
        for i in range(len(market_data) - self.window_size, len(market_data)):
            row_features = market_data.iloc[i].drop(['timestamp', 'open', 'high', 'low', 'close', 'volume']).values
            market_features.extend(row_features)
        
        # เพิ่มข้อมูลพอร์ตโฟลิโอ
        # ปรับให้เป็นค่าปกติ
        base_currency, quote_currency = self.symbol.split('/')
        total_balance = self.current_market_value
        
        if total_balance > 0:
            position_ratio = (self.current_position * self.current_price) / total_balance
        else:
            position_ratio = 0
            
        portfolio_features = np.array([
            self.current_balance / total_balance if total_balance > 0 else 0,  # สัดส่วนเงินสด
            position_ratio  # สัดส่วนสินทรัพย์
        ])
        
        # รวมข้อมูลทั้งหมด
        state = np.concatenate([portfolio_features, market_features])
        
        return state
    
    def _execute_trade(self, action_idx: int):
        """
        ดำเนินการซื้อขายตามการกระทำที่เลือก
        
        Args:
            action_idx (int): ดัชนีของการกระทำ
        """
        try:
            # แปลงดัชนีการกระทำเป็นการกระทำจริง
            if action_idx == 0:  # ขายหนัก
                action_type = 'sell'
                action_size = 1.0
            elif action_idx == 1:  # ขายปานกลาง
                action_type = 'sell'
                action_size = 0.66
            elif action_idx == 2:  # ขายเบา
                action_type = 'sell'
                action_size = 0.33
            elif action_idx == 3:  # ถือครอง
                action_type = 'hold'
                action_size = 0.0
            elif action_idx == 4:  # ซื้อเบา
                action_type = 'buy'
                action_size = 0.33
            elif action_idx == 5:  # ซื้อปานกลาง
                action_type = 'buy'
                action_size = 0.66
            elif action_idx == 6:  # ซื้อหนัก
                action_type = 'buy'
                action_size = 1.0
            
            # ถ้าเป็นการถือครอง ไม่ต้องดำเนินการใดๆ
            if action_type == 'hold':
                logger.info("การกระทำ: ถือครอง (HODL)")
                return
            
            base_currency, quote_currency = self.symbol.split('/')
            
            # ดำเนินการซื้อขาย
            if action_type == 'buy':
                # ตรวจสอบว่ามีเงินเพียงพอหรือไม่
                if self.current_balance <= 0:
                    logger.warning(f"ไม่สามารถซื้อได้: ไม่มี {quote_currency} เพียงพอ")
                    return
                
                # คำนวณจำนวนเงินที่จะใช้ซื้อ
                amount_to_use = self.current_balance * action_size * self.max_position
                
                # ใช้ความเสี่ยงต่อการเทรดหนึ่งครั้งเพื่อปรับจำนวนเงิน
                risk_adjusted_amount = min(amount_to_use, self.current_market_value * self.risk_per_trade)
                
                # คำนวณจำนวนสินทรัพย์ที่จะซื้อ
                amount_to_buy = risk_adjusted_amount / self.current_price
                
                # ปัดเศษตามข้อกำหนดของ Exchange
                market_info = self.exchange.markets[self.symbol]
                precision = market_info['precision']['amount']
                amount_to_buy = float(f"%.{precision}f" % amount_to_buy)
                
                if amount_to_buy * self.current_price < market_info['limits']['cost']['min']:
                    logger.warning(f"มูลค่าการซื้อต่ำกว่าขั้นต่ำ: {amount_to_buy * self.current_price} < {market_info['limits']['cost']['min']}")
                    return
                
                # ส่งคำสั่งซื้อ
                if self.use_limit_orders:
                    # คำนวณราคา limit order
                    limit_price = self.current_price * 1.002  # ราคาสูงกว่าตลาด 0.2%
                    order = self.exchange.create_limit_buy_order(self.symbol, amount_to_buy, limit_price)
                else:
                    order = self.exchange.create_market_buy_order(self.symbol, amount_to_buy)
                
                logger.info(f"ส่งคำสั่งซื้อ: {amount_to_buy} {base_currency} ที่ราคา {self.current_price} {quote_currency}")
                
                # บันทึกคำสั่ง
                self.orders.append({
                    'id': order['id'],
                    'timestamp': datetime.now(),
                    'type': 'buy',
                    'amount': amount_to_buy,
                    'price': self.current_price,
                    'status': order['status']
                })
                
                # ตั้งค่า stop loss และ take profit
                self.stop_loss_price = self.current_price * (1 - self.stop_loss_pct)
                self.take_profit_price = self.current_price * (1 + self.take_profit_pct)
                
                logger.info(f"ตั้งค่า Stop Loss: {self.stop_loss_price}, Take Profit: {self.take_profit_price}")
                
            elif action_type == 'sell':
                # ตรวจสอบว่ามีสินทรัพย์เพียงพอหรือไม่
                if self.current_position <= 0:
                    logger.warning(f"ไม่สามารถขายได้: ไม่มี {base_currency} เพียงพอ")
                    return
                
                # คำนวณจำนวนสินทรัพย์ที่จะขาย
                amount_to_sell = self.current_position * action_size
                
                # ปัดเศษตามข้อกำหนดของ Exchange
                market_info = self.exchange.markets[self.symbol]
                precision = market_info['precision']['amount']
                amount_to_sell = float(f"%.{precision}f" % amount_to_sell)
                
                if amount_to_sell * self.current_price < market_info['limits']['cost']['min']:
                    logger.warning(f"มูลค่าการขายต่ำกว่าขั้นต่ำ: {amount_to_sell * self.current_price} < {market_info['limits']['cost']['min']}")
                    return
                
                # ส่งคำสั่งขาย
                if self.use_limit_orders:
                    # คำนวณราคา limit order
                    limit_price = self.current_price * 0.998  # ราคาต่ำกว่าตลาด 0.2%
                    order = self.exchange.create_limit_sell_order(self.symbol, amount_to_sell, limit_price)
                else:
                    order = self.exchange.create_market_sell_order(self.symbol, amount_to_sell)
                
                logger.info(f"ส่งคำสั่งขาย: {amount_to_sell} {base_currency} ที่ราคา {self.current_price} {quote_currency}")
                
                # บันทึกคำสั่ง
                self.orders.append({
                    'id': order['id'],
                    'timestamp': datetime.now(),
                    'type': 'sell',
                    'amount': amount_to_sell,
                    'price': self.current_price,
                    'status': order['status']
                })
                
                # รีเซ็ต stop loss และ take profit
                self.stop_loss_price = None
                self.take_profit_price = None
        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดำเนินการซื้อขาย: {str(e)}")
    
    def _check_and_update_orders(self):
        """
        ตรวจสอบและอัพเดทสถานะคำสั่งที่รอดำเนินการ
        """
        try:
            # ตรวจสอบสถานะของคำสั่งที่รอดำเนินการ
            pending_orders = [order for order in self.orders if order['status'] in ['open', 'created']]
            
            for order in pending_orders:
                # ดึงข้อมูลคำสั่งล่าสุด
                updated_order = self.exchange.fetch_order(order['id'], self.symbol)
                
                # อัพเดทสถานะ
                order['status'] = updated_order['status']
                
                # ถ้าคำสั่งเสร็จสมบูรณ์ ให้บันทึกการเทรด
                if updated_order['status'] == 'closed':
                    self.trades_history.append({
                        'timestamp': datetime.now(),
                        'type': order['type'],
                        'amount': float(updated_order['amount']),
                        'price': float(updated_order['price']),
                        'cost': float(updated_order['cost']),
                        'fee': float(updated_order['fee']['cost']) if 'fee' in updated_order and updated_order['fee'] is not None else 0
                    })
                    
                    logger.info(f"คำสั่ง {order['id']} เสร็จสมบูรณ์: {order['type']} {updated_order['amount']} {self.symbol} ที่ราคา {updated_order['price']}")
                
                # ถ้าคำสั่งรอนานเกินไป ให้ยกเลิก
                elif (datetime.now() - order['timestamp']).total_seconds() > self.order_timeout:
                    self.exchange.cancel_order(order['id'], self.symbol)
                    order['status'] = 'canceled'
                    
                    logger.warning(f"ยกเลิกคำสั่ง {order['id']} เนื่องจากหมดเวลา")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบคำสั่ง: {str(e)}")
    
    def _check_stop_loss_take_profit(self):
        """
        ตรวจสอบและดำเนินการ stop loss หรือ take profit ถ้าจำเป็น
        """
        if self.current_position <= 0 or self.stop_loss_price is None or self.take_profit_price is None:
            return
            
        try:
            # ตรวจสอบ stop loss
            if self.current_price <= self.stop_loss_price:
                logger.info(f"ทริกเกอร์ Stop Loss ที่ราคา {self.current_price} (ตั้งค่าไว้ที่ {self.stop_loss_price})")
                
                # ขายทั้งหมด
                market_info = self.exchange.markets[self.symbol]
                precision = market_info['precision']['amount']
                amount_to_sell = float(f"%.{precision}f" % self.current_position)
                
                order = self.exchange.create_market_sell_order(self.symbol, amount_to_sell)
                
                # บันทึกคำสั่ง
                self.orders.append({
                    'id': order['id'],
                    'timestamp': datetime.now(),
                    'type': 'sell',
                    'amount': amount_to_sell,
                    'price': self.current_price,
                    'status': order['status'],
                    'reason': 'stop_loss'
                })
                
                # รีเซ็ต stop loss และ take profit
                self.stop_loss_price = None
                self.take_profit_price = None
                
            # ตรวจสอบ take profit
            elif self.current_price >= self.take_profit_price:
                logger.info(f"ทริกเกอร์ Take Profit ที่ราคา {self.current_price} (ตั้งค่าไว้ที่ {self.take_profit_price})")
                
                # ขายทั้งหมด
                # ขายทั้งหมด
                market_info = self.exchange.markets[self.symbol]
                precision = market_info['precision']['amount']
                amount_to_sell = float(f"%.{precision}f" % self.current_position)
                
                order = self.exchange.create_market_sell_order(self.symbol, amount_to_sell)
                
                # บันทึกคำสั่ง
                self.orders.append({
                    'id': order['id'],
                    'timestamp': datetime.now(),
                    'type': 'sell',
                    'amount': amount_to_sell,
                    'price': self.current_price,
                    'status': order['status'],
                    'reason': 'take_profit'
                })
                
                # รีเซ็ต stop loss และ take profit
                self.stop_loss_price = None
                self.take_profit_price = None
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบ stop loss/take profit: {str(e)}")
    
    def _save_state(self):
        """
        บันทึกสถานะปัจจุบันของบอท
        """
        try:
            # บันทึกประวัติการเทรด
            pd.DataFrame(self.trades_history).to_csv(f"{self.log_dir}/trades_history.csv", index=False)
            
            # บันทึกประวัติคำสั่ง
            pd.DataFrame(self.orders).to_csv(f"{self.log_dir}/orders.csv", index=False)
            
            # บันทึกข้อมูลสถานะ
            with open(f"{self.log_dir}/status.txt", 'w') as f:
                f.write(f"Last Update: {datetime.now()}\n")
                f.write(f"Symbol: {self.symbol}\n")
                f.write(f"Current Price: {self.current_price}\n")
                f.write(f"Current Position: {self.current_position}\n")
                f.write(f"Current Balance: {self.current_balance}\n")
                f.write(f"Current Market Value: {self.current_market_value}\n")
                f.write(f"Stop Loss Price: {self.stop_loss_price}\n")
                f.write(f"Take Profit Price: {self.take_profit_price}\n")
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกสถานะ: {str(e)}")
    
    def run(self, duration_hours: Optional[float] = None):
        """
        รันบอทเทรดแบบเรียลไทม์
        
        Args:
            duration_hours (Optional[float]): ระยะเวลาที่ต้องการให้บอททำงาน (ชั่วโมง)
                                            ถ้าไม่ระบุ บอทจะทำงานไปเรื่อยๆ จนกว่าจะมีการหยุด
        """
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours) if duration_hours else None
        
        logger.info(f"เริ่มบอทเทรดแบบเรียลไทม์สำหรับ {self.symbol} ที่ {start_time}")
        if end_time:
            logger.info(f"บอทจะทำงานจนถึง {end_time}")
        
        try:
            running = True
            last_check_time = datetime.now()
            
            while running:
                current_time = datetime.now()
                
                # ตรวจสอบว่าถึงเวลาสิ้นสุดหรือยัง
                if end_time and current_time > end_time:
                    logger.info(f"ครบกำหนดเวลา {duration_hours} ชั่วโมง บอทจะหยุดทำงาน")
                    break
                
                # ดำเนินการทุกรอบตามกรอบเวลาที่กำหนด
                # หรือทุก 1 นาทีสำหรับการตรวจสอบอื่นๆ
                time_diff = (current_time - last_check_time).total_seconds()
                
                if time_diff >= min(self.timeframe_seconds, 60):
                    # อัพเดทข้อมูลบัญชี
                    self._update_account_info()
                    
                    # ตรวจสอบและอัพเดทคำสั่งที่รอดำเนินการ
                    self._check_and_update_orders()
                    
                    # ตรวจสอบ stop loss และ take profit
                    self._check_stop_loss_take_profit()
                    
                    # บันทึกสถานะ
                    self._save_state()
                    
                    last_check_time = current_time
                
                # ดำเนินการทุกรอบตามกรอบเวลาที่กำหนด
                if time_diff >= self.timeframe_seconds:
                    # ดึงข้อมูลตลาดล่าสุด
                    market_data = self._fetch_current_market_data()
                    
                    if market_data is not None:
                        # สร้าง state vector
                        state = self._create_state(market_data)
                        
                        # ทำนายการกระทำ
                        action_idx = self.agent.act(state, training=False)
                        
                        # ดำเนินการซื้อขาย
                        self._execute_trade(action_idx)
                        
                        # บันทึกเวลาล่าสุดที่ดำเนินการ
                        self.last_action_time = current_time
                    
                # หน่วงเวลาเพื่อลดการใช้งาน CPU
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("ผู้ใช้หยุดการทำงานของบอท")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการรันบอท: {str(e)}")
        finally:
            # บันทึกสถานะสุดท้าย
            self._save_state()
            
            # สรุปผลการทำงาน
            total_trades = len(self.trades_history)
            if total_trades > 0:
                # คำนวณกำไร/ขาดทุน
                base_currency, quote_currency = self.symbol.split('/')
                initial_value = self.trades_history[0]['price'] * self.trades_history[0]['amount'] if self.trades_history[0]['type'] == 'buy' else 0
                final_value = self.current_market_value
                
                profit_loss = ((final_value - initial_value) / initial_value) * 100 if initial_value > 0 else 0
                
                logger.info(f"สรุปการทำงานของบอท:")
                logger.info(f"จำนวนการเทรดทั้งหมด: {total_trades}")
                logger.info(f"กำไร/ขาดทุน: {profit_loss:.2f}%")
                logger.info(f"มูลค่าพอร์ตสุดท้าย: {self.current_market_value} {quote_currency}")
            else:
                logger.info("ไม่มีการเทรดเกิดขึ้นในช่วงเวลาที่บอททำงาน")
            
            logger.info(f"บอทหยุดทำงานที่ {datetime.now()}")


def main():
    """
    ฟังก์ชันหลักสำหรับการรันสคริปต์
    """
    parser = argparse.ArgumentParser(description='Crypto Trading Bot แบบเรียลไทม์')
    parser.add_argument('--model_path', type=str, required=True, help='เส้นทางไปยังไฟล์โมเดล')
    parser.add_argument('--exchange', type=str, default='binance', help='ID ของ Exchange')
    parser.add_argument('--api_key', type=str, required=True, help='API Key')
    parser.add_argument('--api_secret', type=str, required=True, help='API Secret')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='สัญลักษณ์คู่เหรียญ')
    parser.add_argument('--timeframe', type=str, default='15m', help='กรอบเวลา')
    parser.add_argument('--window_size', type=int, default=10, help='ขนาดหน้าต่างข้อมูลย้อนหลัง')
    parser.add_argument('--max_position', type=float, default=1.0, help='ตำแหน่งสูงสุดที่อนุญาตให้ถือครอง')
    parser.add_argument('--risk_per_trade', type=float, default=0.02, help='ความเสี่ยงต่อการเทรดหนึ่งครั้ง')
    parser.add_argument('--stop_loss_pct', type=float, default=0.03, help='เปอร์เซ็นต์ stop loss')
    parser.add_argument('--take_profit_pct', type=float, default=0.06, help='เปอร์เซ็นต์ take profit')
    parser.add_argument('--use_limit_orders', action='store_true', help='ใช้คำสั่ง limit order แทน market order')
    parser.add_argument('--order_timeout', type=int, default=60, help='เวลาที่รอให้คำสั่งสำเร็จก่อนยกเลิก (วินาที)')
    parser.add_argument('--duration', type=float, help='ระยะเวลาที่ต้องการให้บอททำงาน (ชั่วโมง)')
    
    args = parser.parse_args()
    
    bot = LiveTradingBot(
        model_path=args.model_path,
        exchange_id=args.exchange,
        api_key=args.api_key,
        api_secret=args.api_secret,
        symbol=args.symbol,
        timeframe=args.timeframe,
        window_size=args.window_size,
        max_position=args.max_position,
        risk_per_trade=args.risk_per_trade,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        use_limit_orders=args.use_limit_orders,
        order_timeout=args.order_timeout
    )
    
    bot.run(duration_hours=args.duration)


if __name__ == '__main__':
    main()