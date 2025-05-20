"""
เก็บข้อมูลราคาและปริมาณการซื้อขายจาก Binance Exchange
"""

import os
import time
import datetime
import pandas as pd
import numpy as np
import ccxt
import logging
import sys
from typing import Optional, Dict, Any

# เพิ่ม path ของ root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.config import DATA_CONFIG, TRADING_CONFIG, BINANCE_CONFIG
from config.credentials import API_KEY, API_SECRET
from utils.logger import setup_logger

# ตั้งค่า logger
logger = setup_logger('data_collector')

class BinanceDataCollector:
    """
    คลาสสำหรับการเก็บข้อมูลราคาและปริมาณการซื้อขายจาก Binance Exchange
    """
    
    def __init__(self, symbol: str, interval: str, start_date: str, end_date: str, testnet: bool = True):
        """
        เริ่มต้นตัวเก็บข้อมูลจาก Binance
        
        Args:
            symbol (str): คู่เหรียญ (เช่น 'BTC/USDT')
            interval (str): กรอบเวลา (เช่น '5m', '1h', '4h', '1d')
            start_date (str): วันที่เริ่มต้น (YYYY-MM-DD)
            end_date (str): วันที่สิ้นสุด (YYYY-MM-DD)
            testnet (bool): ใช้ testnet หรือไม่
        """
        self.symbol = symbol.replace('USDT', '/USDT')  # แปลง BTCUSDT เป็น BTC/USDT
        self.interval = interval
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # ตั้งค่า exchange
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        if testnet:
            self.exchange.set_sandbox_mode(True)
            
        # ตั้งค่า logger
        self.logger = logging.getLogger(__name__)
        
        # สร้างโฟลเดอร์สำหรับเก็บข้อมูล
        self.data_dir = os.path.join(DATA_CONFIG['data_dir'])
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        logger.info(f"เริ่มต้น BinanceDataCollector สำหรับ {self.symbol} ช่วงเวลา {self.interval}")
    
    def get_historical_klines(self, save_to_csv: bool = True) -> pd.DataFrame:
        """
        ดึงข้อมูลราคาย้อนหลัง
        
        Returns:
            pd.DataFrame: ข้อมูลราคาย้อนหลัง
        """
        try:
            # ตรวจสอบว่ามีไฟล์ข้อมูลอยู่แล้วหรือไม่
            file_name = f"{self.symbol.replace('/', '')}_{self.interval}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
            file_path = os.path.join(self.data_dir, file_name)
            
            if os.path.exists(file_path):
                self.logger.info(f"พบไฟล์ข้อมูล {file_path}")
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                return df
            
            # ถ้าไม่มีไฟล์ ให้ดึงข้อมูลจาก exchange
            self.logger.info("ไม่พบไฟล์ข้อมูล กำลังดึงข้อมูลจาก exchange...")
            
            # แปลง interval เป็นรูปแบบที่ ccxt ใช้
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            timeframe = interval_map.get(self.interval, '1h')
            
            # ดึงข้อมูล
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=timeframe,
                since=int(self.start_date.timestamp() * 1000),
                limit=1000  # จำนวนแท่งสูงสุดที่ดึงได้ต่อครั้ง
            )
            
            # แปลงเป็น DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # กรองข้อมูลตามช่วงวันที่
            df = df[(df.index >= self.start_date) & (df.index <= self.end_date)]
            
            self.logger.info(f"ดึงข้อมูลสำเร็จ: {len(df)} แท่ง")
            
            # ตรวจสอบความถูกต้องของ timestamp
            if df['timestamp'].min() < datetime.datetime(2020, 1, 1):
                self.logger.error(f"พบ timestamp ที่ไม่ถูกต้อง: {df['timestamp'].min()}")
                return pd.DataFrame()
            
            # เพิ่มคอลัมน์วันที่
            df['date'] = df['timestamp'].dt.date
            
            # ลบคอลัมน์ที่ไม่จำเป็น
            df = df.drop(['close_time', 'ignore'], axis=1)
            
            # ตั้งค่า index เป็น timestamp
            df.set_index('timestamp', inplace=True)
            
            # บันทึกเป็นไฟล์ CSV
            if save_to_csv and not df.empty:
                df.to_csv(file_path)
                self.logger.info(f"บันทึกข้อมูลไปที่ {file_path}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {str(e)}")
            return pd.DataFrame()
    
    def update_historical_data(self) -> pd.DataFrame:
        """
        อัพเดทข้อมูลประวัติให้เป็นปัจจุบัน
        
        Returns:
            pd.DataFrame: ข้อมูลที่อัพเดทแล้ว
        """
        # หาไฟล์ข้อมูลล่าสุด
        pattern = f"{self.symbol}_{self.interval}_*.csv"
        files = [f for f in os.listdir(self.data_dir) if f.startswith(f"{self.symbol}_{self.interval}_")]
        
        if not files:
            # ถ้าไม่มีไฟล์ ให้ดึงข้อมูลใหม่ทั้งหมด
            return self.get_historical_klines()
        
        # เลือกไฟล์ล่าสุด
        latest_file = max(files)
        file_path = os.path.join(self.data_dir, latest_file)
        
        # อ่านข้อมูลจากไฟล์
        df = pd.read_csv(file_path)
        
        # แปลง timestamp เป็น datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # หาวันที่ล่าสุดในข้อมูล
        latest_date = df['timestamp'].max()
        
        # กำหนดวันที่เริ่มต้นสำหรับการอัพเดท
        self.start_date = latest_date + datetime.timedelta(minutes=1)
        
        # ถ้าวันที่เริ่มต้นอยู่ในอนาคต ไม่ต้องอัพเดท
        if self.start_date > datetime.datetime.now():
            self.logger.info("ข้อมูลเป็นปัจจุบันแล้ว")
            return df
        
        # ดึงข้อมูลใหม่
        self.logger.info(f"กำลังอัพเดทข้อมูลตั้งแต่ {self.start_date}")
        new_data = self.get_historical_klines(save_to_csv=False)
        
        if new_data.empty:
            self.logger.info("ไม่มีข้อมูลใหม่")
            return df
        
        # รวมข้อมูลเก่าและใหม่
        df = pd.concat([df, new_data], ignore_index=True)
        
        # ลบข้อมูลซ้ำ
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # เรียงลำดับข้อมูลตาม timestamp
        df = df.sort_values('timestamp')
        
        # บันทึกไฟล์
        file_name = f"{self.symbol}_{self.interval}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
        file_path = os.path.join(self.data_dir, file_name)
        df.to_csv(file_path, index=False)
        self.logger.info(f"บันทึกข้อมูลที่อัพเดทแล้วไปที่ {file_path}")
        
        return df

    @staticmethod
    def collect_and_process_data(
        symbol: str,
        interval: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: Optional[int] = None,
        testnet: bool = True,
        update: bool = False,
        process: bool = False,
        normalize: str = 'zscore',
        missing_method: str = 'forward'
    ) -> pd.DataFrame:
        """
        ฟังก์ชันหลักสำหรับเก็บและประมวลผลข้อมูล
        
        Args:
            symbol (str): สัญลักษณ์คู่เหรียญ
            interval (str): กรอบเวลา
            start_date (Optional[str]): วันที่เริ่มต้น
            end_date (Optional[str]): วันที่สิ้นสุด
            days (Optional[int]): จำนวนวันย้อนหลัง
            testnet (bool): ใช้ testnet หรือไม่
            update (bool): อัพเดทข้อมูลเดิมหรือไม่
            process (bool): ประมวลผลข้อมูลหรือไม่
            normalize (str): วิธีการทำ normalization
            missing_method (str): วิธีการจัดการข้อมูลที่หายไป
            
        Returns:
            pd.DataFrame: ข้อมูลที่ประมวลผลแล้ว
        """
        # คำนวณวันที่เริ่มต้นจากจำนวนวัน
        if days:
            start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        # สร้าง DataCollector
        collector = BinanceDataCollector(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            testnet=testnet
        )
        
        # ถ้าระบุให้อัพเดทข้อมูล
        if update:
            self.logger.info("กำลังอัพเดทข้อมูลให้เป็นปัจจุบัน")
            df = collector.update_historical_data()
        else:
            # เก็บข้อมูลใหม่
            self.logger.info("กำลังเก็บข้อมูลใหม่")
            df = collector.get_historical_klines()
        
        self.logger.info(f"เก็บข้อมูลเสร็จสิ้น ได้ข้อมูลทั้งหมด {len(df)} แถว")
        
        # ถ้าระบุให้ประมวลผลข้อมูล
        if process and not df.empty:
            self.logger.info("กำลังประมวลผลข้อมูล")
            # สร้าง features
            from data.feature_engineering import FeatureEngineering
            fe = FeatureEngineering(df=df)
            processed_df = fe.process_data_pipeline(
                normalize_method=normalize,
                missing_method=missing_method
            )
            
            self.logger.info(f"ประมวลผลข้อมูลเสร็จสิ้น ได้ข้อมูลทั้งหมด {len(processed_df)} แถว และ {len(processed_df.columns)} คอลัมน์")
            return processed_df
            
        return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='เก็บข้อมูลราคาและปริมาณการซื้อขายจาก Binance')
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='สัญลักษณ์ของคู่เหรียญ (เช่น BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h',
                        help='ช่วงเวลาของแท่งเทียน (1m, 5m, 15m, 1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=None,
                        help='จำนวนวันย้อนหลังที่ต้องการเก็บข้อมูล')
    parser.add_argument('--start_date', type=str, default=None,
                        help='วันที่เริ่มต้นในรูปแบบ YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default=None,
                        help='วันที่สิ้นสุดในรูปแบบ YYYY-MM-DD')
    parser.add_argument('--testnet', action='store_true',
                        help='ใช้ testnet')
    parser.add_argument('--update', action='store_true',
                        help='อัพเดทข้อมูลเดิมให้เป็นปัจจุบัน')
    parser.add_argument('--process', action='store_true',
                        help='ประมวลผลข้อมูลหลังจากเก็บข้อมูลเสร็จ')
    parser.add_argument('--normalize', type=str, default='zscore',
                        help='วิธีการทำ normalization (zscore หรือ minmax)')
    parser.add_argument('--missing_method', type=str, default='forward',
                        help='วิธีการจัดการกับข้อมูลที่หายไป (forward, backward, interpolate, drop)')
    
    args = parser.parse_args()
    
    BinanceDataCollector.collect_and_process_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        days=args.days,
        testnet=args.testnet,
        update=args.update,
        process=args.process,
        normalize=args.normalize,
        missing_method=args.missing_method
    )
