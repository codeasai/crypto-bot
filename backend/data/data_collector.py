"""
เก็บข้อมูลราคาและปริมาณการซื้อขายจาก Binance Exchange
"""

import os
import time
import datetime
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_CONFIG, TRADING_CONFIG, BINANCE_CONFIG
from config.credentials import API_KEY, API_SECRET
from utils.logger import setup_logger

# ตั้งค่า logger
logger = setup_logger('data_collector')


class BinanceDataCollector:
    """
    คลาสสำหรับการเก็บข้อมูลราคาและปริมาณการซื้อขายจาก Binance Exchange
    """
    
    def __init__(self, symbol=None, interval=None, start_date=None, end_date=None, testnet=True):
        """
        เริ่มต้นการเชื่อมต่อกับ Binance API
        
        Args:
            symbol (str): สัญลักษณ์ของคู่เหรียญ (เช่น 'BTCUSDT')
            interval (str): ช่วงเวลาของแท่งเทียน (เช่น '1h', '1d')
            start_date (str): วันที่เริ่มต้นในรูปแบบ 'YYYY-MM-DD'
            end_date (str): วันที่สิ้นสุดในรูปแบบ 'YYYY-MM-DD'
            testnet (bool): ใช้ testnet หรือไม่
        """
        # ตั้งค่าพารามิเตอร์
        self.symbol = symbol or TRADING_CONFIG['symbol']
        self.interval = interval or TRADING_CONFIG['interval']
        self.testnet = testnet
        
        # สร้าง Binance Client
        self.client = Client(API_KEY, API_SECRET, testnet=testnet)
        
        # กำหนดวันที่เริ่มต้นและสิ้นสุด
        if start_date:
            self.start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        else:
            # ใช้ค่าเริ่มต้นจาก config
            days_ago = DATA_CONFIG['historical_days']
            self.start_date = datetime.datetime.now() - datetime.timedelta(days=days_ago)
            
        self.end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.datetime.now()
        
        # สร้างโฟลเดอร์สำหรับเก็บข้อมูล
        self.data_dir = DATA_CONFIG['data_dir']
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        logger.info(f"เริ่มต้น BinanceDataCollector สำหรับ {self.symbol} ช่วงเวลา {self.interval}")
    
    def get_historical_klines(self, save_to_csv=True):
        """
        ดึงข้อมูลแท่งเทียนย้อนหลัง
        
        Args:
            save_to_csv (bool): บันทึกข้อมูลเป็นไฟล์ CSV หรือไม่
            
        Returns:
            pd.DataFrame: ข้อมูลแท่งเทียน
        """
        # แปลงวันที่เป็น timestamp ในรูปแบบที่ Binance ต้องการ
        start_ts = int(self.start_date.timestamp() * 1000)
        end_ts = int(self.end_date.timestamp() * 1000)
        
        logger.info(f"กำลังดึงข้อมูล {self.symbol} ตั้งแต่ {self.start_date} ถึง {self.end_date}")
        
        # เตรียมรายการสำหรับเก็บข้อมูล
        all_klines = []
        
        # คำนวณจำนวนแท่งเทียนที่ต้องการ
        if self.interval == '1m':
            limit = 1000  # Binance จำกัดการเรียกข้อมูลสูงสุด 1000 แท่งต่อครั้ง
        elif self.interval == '5m':
            limit = 1000
        elif self.interval == '15m':
            limit = 1000
        elif self.interval == '1h':
            limit = 1000
        elif self.interval == '4h':
            limit = 1000
        elif self.interval == '1d':
            limit = 1000
        else:
            limit = 1000
        
        # ดึงข้อมูลแบบแบ่งตามช่วงเวลา
        current_ts = start_ts
        
        while current_ts < end_ts:
            try:
                # เรียกใช้ API ของ Binance
                klines = self.client.get_klines(
                    symbol=self.symbol,
                    interval=self.interval,
                    startTime=current_ts,
                    endTime=end_ts,
                    limit=limit
                )
                
                # ถ้าไม่มีข้อมูล ให้หยุด
                if not klines:
                    break
                
                # เพิ่มข้อมูลเข้าไปในรายการ
                all_klines.extend(klines)
                
                # อัพเดท timestamp สุดท้าย
                current_ts = klines[-1][0] + 1
                
                # หน่วงเวลาเพื่อไม่ให้เกิน rate limit ของ Binance
                time.sleep(0.5)
                
                logger.info(f"ดึงข้อมูลแล้ว {len(all_klines)} แท่ง")
                
            except BinanceAPIException as e:
                logger.error(f"เกิดข้อผิดพลาดจาก Binance API: {e}")
                # หน่วงเวลานานขึ้นถ้าเกิด rate limit
                time.sleep(60)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาด: {e}")
                break
        
        # แปลงข้อมูลเป็น DataFrame
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(all_klines, columns=columns)
        
        # แปลงประเภทข้อมูล
        numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                          'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # แปลง timestamp เป็น datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # เพิ่มคอลัมน์วันที่
        df['date'] = df['timestamp'].dt.date
        
        # ลบคอลัมน์ที่ไม่จำเป็น
        df = df.drop(['close_time', 'ignore'], axis=1)
        
        # บันทึกเป็นไฟล์ CSV
        if save_to_csv:
            file_name = f"{self.symbol}_{self.interval}_{self.start_date.strftime('%Y%m%d')}_{self.end_date.strftime('%Y%m%d')}.csv"
            file_path = os.path.join(self.data_dir, file_name)
            df.to_csv(file_path, index=False)
            logger.info(f"บันทึกข้อมูลไปที่ {file_path}")
        
        return df
    
    def update_historical_data(self):
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
            logger.info("ข้อมูลเป็นปัจจุบันแล้ว")
            return df
        
        # ดึงข้อมูลใหม่
        logger.info(f"กำลังอัพเดทข้อมูลตั้งแต่ {self.start_date}")
        new_data = self.get_historical_klines(save_to_csv=False)
        
        if new_data.empty:
            logger.info("ไม่มีข้อมูลใหม่")
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
        logger.info(f"บันทึกข้อมูลที่อัพเดทแล้วไปที่ {file_path}")
        
        return df


if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    collector = BinanceDataCollector(symbol='BTCUSDT', interval='1h', testnet=True)
    
    # ดึงข้อมูลแท่งเทียนย้อนหลัง
    klines_df = collector.get_historical_klines()
    print(f"จำนวนข้อมูลแท่งเทียน: {len(klines_df)}")
