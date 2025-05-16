"""
ฟังก์ชันสำหรับเชื่อมต่อกับ Binance API ผ่าน CCXT
"""

import os
import time
import ccxt
import logging
import sys

# เพิ่ม path ของ root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.config import BINANCE_CONFIG, BASE_URL, STREAM_URL, TESTNET
from config.credentials import API_KEY, API_SECRET
from utils.logger import setup_logger
from utils.APIKeyValidator import APIKeyValidator

# ตั้งค่า logger
logger = setup_logger('binance_api')

class BinanceAPI:
    """
    คลาสสำหรับเชื่อมต่อกับ Binance API ผ่าน CCXT
    """
    
    def __init__(self, symbol=None, testnet=True):
        """
        เริ่มต้นการเชื่อมต่อกับ Binance API
        
        Args:
            symbol (str): สัญลักษณ์ของคู่เหรียญ (เช่น 'BTCUSDT' หรือ 'BTC/USDT')
            testnet (bool): ใช้ testnet หรือไม่
        """
        self.symbol = symbol
        # แปลงรูปแบบ symbol ให้ถูกต้อง (BTC/USDT สำหรับ CCXT)
        if symbol and '/' in symbol:
            self.ccxt_symbol = symbol
        elif symbol:
            self.ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}" if len(symbol) > 4 else symbol
        else:
            self.ccxt_symbol = None
            
        self.testnet = testnet
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        
        # สร้าง APIKeyValidator
        self.validator = APIKeyValidator(
            api_key=self.api_key,
            api_secret=self.api_secret,
            base_url=BASE_URL
        )
        
        # ซิงค์เวลาครั้งแรก
        if not self.validator.sync_time():
            logger.error("ไม่สามารถซิงค์เวลาได้")
            
        # ตรวจสอบ API key
        if not self.validator.validate_api_key():
            logger.error("API key ไม่ถูกต้อง")
            
        # สร้าง Binance Exchange ด้วย CCXT
        config = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'testnet': testnet,
                'adjustForTimeDifference': True,
                'recvWindow': 60000
            }
        }
        
        if testnet:
            config['urls'] = {
                'api': {
                    'public': BASE_URL,
                    'private': BASE_URL,
                }
            }
        
        self.exchange = ccxt.binance(config)
        
        # ตั้งค่า testnet
        if testnet:
            self.exchange.set_sandbox_mode(True)
        
        logger.info(f"เริ่มต้น BinanceAPI สำหรับ {symbol} บน {'Testnet' if testnet else 'Live'}")
    
    def _sync_time_if_needed(self):
        """
        ตรวจสอบและซิงค์เวลาถ้าจำเป็น
        
        Returns:
            bool: True หากซิงค์สำเร็จหรือไม่จำเป็นต้องซิงค์, False หากซิงค์ล้มเหลว
        """
        return self.validator.sync_time()
        
    def _validate_request(self):
        """
        ตรวจสอบความพร้อมก่อนทำ request
        
        Returns:
            bool: True หากพร้อมทำ request, False หากไม่พร้อม
        """
        if not self._sync_time_if_needed():
            logger.error("ไม่สามารถซิงค์เวลาได้")
            return False
        return True
    
    def get_exchange_info(self):
        """
        ดึงข้อมูลของ exchange
        
        Returns:
            dict: ข้อมูลของ exchange
        """
        try:
            markets = self.exchange.fetch_markets()
            return {market['symbol']: market for market in markets}
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล exchange: {e}")
            return {}
    
    def get_ticker(self):
        """
        ดึงข้อมูลราคาล่าสุด
        
        Returns:
            dict: ข้อมูลราคาล่าสุด
        """
        try:
            if not self.ccxt_symbol:
                raise ValueError("ต้องระบุ symbol")
                
            ticker = self.exchange.fetch_ticker(self.ccxt_symbol)
            return {
                'lastPrice': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume': ticker['baseVolume']
            }
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลราคา: {e}")
            return {'lastPrice': '0'}
    
    def get_klines(self, interval, limit=500, start_time=None, end_time=None):
        """
        ดึงข้อมูลแท่งเทียน
        
        Args:
            interval (str): ช่วงเวลา (1m, 5m, 15m, 1h, 4h, 1d)
            limit (int): จำนวนแท่งเทียนที่ต้องการ
            start_time (int): เวลาเริ่มต้น (timestamp in milliseconds)
            end_time (int): เวลาสิ้นสุด (timestamp in milliseconds)
            
        Returns:
            list: รายการข้อมูลแท่งเทียน
        """
        try:
            if not self.ccxt_symbol:
                raise ValueError("ต้องระบุ symbol")
                
            # แปลง interval ให้ตรงกับรูปแบบของ CCXT
            timeframe_map = {
                '1m': '1m', '5m': '5m', '15m': '15m',
                '1h': '1h', '4h': '4h', '1d': '1d'
            }
            timeframe = timeframe_map.get(interval, '1h')
            
            # สร้างพารามิเตอร์สำหรับ API
            params = {
                'limit': limit
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            # ดึงข้อมูลแท่งเทียน
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.ccxt_symbol,
                timeframe=timeframe,
                params=params
            )
            
            if not ohlcv:
                logger.warning(f"ไม่พบข้อมูลแท่งเทียนสำหรับ {self.ccxt_symbol}")
                return []
                
            return ohlcv
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลแท่งเทียน: {e}")
            return []
    
    def get_account_info(self):
        """
        ดึงข้อมูลบัญชี
        
        Returns:
            dict: ข้อมูลบัญชี
        """
        try:
            if not self._validate_request():
                return {'total': {}}
                
            # ตรวจสอบการเชื่อมต่อ
            if not self.exchange.has['fetchBalance']:
                raise Exception("ไม่รองรับการดึงข้อมูลบัญชี")

            # ดึงข้อมูลบัญชี
            balance = self.exchange.fetch_balance()
            
            if not balance or 'total' not in balance:
                logger.warning("ไม่พบข้อมูลบัญชี")
                return {'total': {}}
            
            # แสดงข้อมูลยอดคงเหลือที่มีมูลค่ามากกว่า 0
            logger.info("\n💰 ยอดคงเหลือในบัญชี:")
            logger.info("-" * 50)
            for currency, amount in balance['total'].items():
                if amount > 0:
                    logger.info(f"{currency}: {amount}")

            # แสดงข้อมูลการตั้งค่าบัญชี
            logger.info("\n⚙️ การตั้งค่าบัญชี:")
            logger.info("-" * 50)
            logger.info(f"ประเภทบัญชี: {self.exchange.options['defaultType']}")
            logger.info(f"สถานะการเชื่อมต่อ: {'เชื่อมต่อสำเร็จ' if self.exchange.has['fetchBalance'] else 'ไม่สามารถเชื่อมต่อได้'}")
            
            return balance
        except Exception as e:
            logger.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            return {'total': {}}
    
    def create_order(self, **params):
        """
        สร้างคำสั่งซื้อขาย
        
        Args:
            **params: พารามิเตอร์ของคำสั่ง (symbol, side, type, quantity, price, ...)
            
        Returns:
            dict: ผลลัพธ์การสร้างคำสั่ง
        """
        try:
            if not self._validate_request():
                raise Exception("ไม่สามารถทำ request ได้")
                
            symbol = params.get('symbol', self.ccxt_symbol)
            if not symbol:
                raise ValueError("ต้องระบุ symbol")
                
            # สร้างคำสั่งผ่าน CCXT
            order = self.exchange.create_order(
                symbol=symbol,
                type=params.get('type', 'limit').lower(),
                side=params.get('side', 'buy').lower(),
                amount=params.get('quantity'),
                price=params.get('price'),
                params={
                    'timeInForce': params.get('timeInForce', 'GTC')
                } if params.get('type', 'limit').lower() == 'limit' else {}
            )
            
            return order
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างคำสั่ง: {e}")
            raise
    
    def get_open_orders(self, symbol=None):
        """
        ดึงรายการคำสั่งที่ยังไม่เสร็จสมบูรณ์
        
        Args:
            symbol (str): สัญลักษณ์ของคู่เหรียญ
            
        Returns:
            list: รายการคำสั่งที่ยังไม่เสร็จสมบูรณ์
        """
        try:
            if not self._validate_request():
                return []
                
            symbol = symbol or self.ccxt_symbol
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงคำสั่งที่ยังไม่เสร็จสมบูรณ์: {e}")
            return []
    
    def cancel_order(self, **params):
        """
        ยกเลิกคำสั่ง
        
        Args:
            **params: พารามิเตอร์ของคำสั่ง (symbol, orderId, ...)
            
        Returns:
            dict: ผลลัพธ์การยกเลิกคำสั่ง
        """
        try:
            if not self._validate_request():
                raise Exception("ไม่สามารถทำ request ได้")
                
            symbol = params.get('symbol', self.ccxt_symbol)
            order_id = params.get('orderId')
            
            if not symbol or not order_id:
                raise ValueError("ต้องระบุ symbol และ orderId")
                
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการยกเลิกคำสั่ง: {e}")
            raise
    
    def get_order(self, **params):
        """
        ดึงข้อมูลคำสั่ง
        
        Args:
            **params: พารามิเตอร์ของคำสั่ง (symbol, orderId, ...)
            
        Returns:
            dict: ข้อมูลคำสั่ง
        """
        try:
            if not self._validate_request():
                raise Exception("ไม่สามารถทำ request ได้")
                
            symbol = params.get('symbol', self.ccxt_symbol)
            order_id = params.get('orderId')
            
            if not symbol or not order_id:
                raise ValueError("ต้องระบุ symbol และ orderId")
                
            return self.exchange.fetch_order(order_id, symbol)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลคำสั่ง: {e}")
            raise
    
    def get_all_orders(self, **params):
        """
        ดึงประวัติคำสั่งทั้งหมด
        
        Args:
            **params: พารามิเตอร์ (symbol, limit, ...)
            
        Returns:
            list: ประวัติคำสั่ง
        """
        try:
            if not self._validate_request():
                return []
                
            symbol = params.get('symbol', self.ccxt_symbol)
            
            if not symbol:
                raise ValueError("ต้องระบุ symbol")
                
            return self.exchange.fetch_orders(
                symbol,
                limit=params.get('limit'),
                since=params.get('startTime'),
                params={'endTime': params.get('endTime')} if params.get('endTime') else None
            )
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงประวัติคำสั่ง: {e}")
            return []


if __name__ == "__main__":
    # ทดสอบการใช้งาน BinanceAPI
    binance_api = BinanceAPI(symbol='BTC/USDT', testnet=True)
    
    # ทดสอบดึงข้อมูล exchange
    exchange_info = binance_api.get_exchange_info()
    print(f"Exchange Info: {len(exchange_info)} markets")
    
    # ทดสอบดึงข้อมูลราคาล่าสุด
    ticker = binance_api.get_ticker()
    print(f"Last Price: {ticker['lastPrice']}")
    
    # ทดสอบดึงข้อมูลแท่งเทียน
    klines = binance_api.get_klines(interval='1h', limit=10)
    print(f"Klines: {len(klines)}")
    if klines:
        print("ตัวอย่างข้อมูลแท่งเทียน:")
        for k in klines[:2]:
            print(f"เวลา: {k[0]}, เปิด: {k[1]}, สูง: {k[2]}, ต่ำ: {k[3]}, ปิด: {k[4]}, ปริมาณ: {k[5]}")
    
    # ทดสอบดึงข้อมูลบัญชี
    account_info = binance_api.get_account_info()
    print(f"Account Info: {account_info.keys() if isinstance(account_info, dict) else 'Not a dict'}")