"""
สร้าง features และ indicators สำหรับ RL Agent
"""

import os
import pandas as pd
import numpy as np
import ta
import logging
import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import DATA_CONFIG
from utils.logger import setup_logger

# ตั้งค่า logger
logger = setup_logger('feature_engineering')


class FeatureEngineering:
    """
    คลาสสำหรับการสร้าง features และ indicators สำหรับใช้กับ RL Agent
    """
    
    def __init__(self, df=None, file_path=None):
        """
        เริ่มต้นคลาส FeatureEngineering
        
        Args:
            df (pd.DataFrame): DataFrame ที่มีข้อมูลราคา (OHLCV)
            file_path (str): ที่อยู่ไฟล์ CSV ที่มีข้อมูลราคา
        """
        if df is not None:
            self.df = df.copy()
        elif file_path is not None:
            self.df = pd.read_csv(file_path)
            # แปลง timestamp เป็น datetime
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        else:
            raise ValueError("ต้องระบุ df หรือ file_path อย่างใดอย่างหนึ่ง")
        
        # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นหรือไม่
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"ไม่พบคอลัมน์ {col} ในข้อมูล")
        
        # แปลงข้อมูลให้เป็นประเภทที่ถูกต้อง
        for col in required_columns:
            self.df[col] = self.df[col].astype(float)
        
        logger.info(f"เริ่มต้น FeatureEngineering กับข้อมูลขนาด {self.df.shape}")
    
    def add_technical_indicators(self):
        """
        เพิ่มตัวชี้วัดทางเทคนิค (Technical Indicators) ลงใน DataFrame
        
        Returns:
            pd.DataFrame: DataFrame ที่มีตัวชี้วัดเพิ่มเติม
        """
        df = self.df.copy()
        
        # 1. ค่าเฉลี่ยเคลื่อนที่ (Moving Averages)
        logger.info("กำลังเพิ่ม Moving Averages")
        # Simple Moving Averages (SMA)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # Exponential Moving Averages (EMA)
        df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
        df['ema_30'] = ta.trend.ema_indicator(df['close'], window=30)
        
        # 2. ดัชนีกำลังสัมพัทธ์ (Relative Strength Index: RSI)
        logger.info("กำลังเพิ่ม RSI")
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        
        # 3. MACD (Moving Average Convergence Divergence)
        logger.info("กำลังเพิ่ม MACD")
        macd = ta.trend.MACD(
            df['close'], 
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # 4. Bollinger Bands
        logger.info("กำลังเพิ่ม Bollinger Bands")
        bollinger = ta.volatility.BollingerBands(
            df['close'], 
            window=20, 
            window_dev=2
        )
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        # คำนวณ %B (ตำแหน่งราคาใน Bollinger Bands)
        df['bb_pct_b'] = bollinger.bollinger_pband()
        
        # 5. Average True Range (ATR)
        logger.info("กำลังเพิ่ม ATR")
        df['atr_14'] = ta.volatility.average_true_range(
            df['high'], 
            df['low'], 
            df['close'], 
            window=14
        )
        
        # 6. ผลตอบแทนรายวัน (Daily Return)
        df['daily_return'] = df['close'].pct_change()
        
        # บันทึก DataFrame ที่มี features ทั้งหมด
        self.df_features = df
        
        logger.info(f"เพิ่มตัวชี้วัดทางเทคนิคเสร็จสิ้น มีคอลัมน์ทั้งหมด {len(df.columns)} คอลัมน์")
        
        return df
    
    def normalize_features(self, method='zscore'):
        """
        ทำ normalization กับ features
        
        Args:
            method (str): วิธีการทำ normalization ('zscore', 'minmax')
            
        Returns:
            pd.DataFrame: DataFrame ที่ทำ normalization แล้ว
        """
        if not hasattr(self, 'df_features'):
            logger.warning("ยังไม่มีการสร้าง features กรุณาเรียก add_technical_indicators() ก่อน")
            self.add_technical_indicators()
        
        df = self.df_features.copy()
        
        # คอลัมน์ที่ไม่ต้องทำ normalization
        exclude_columns = ['timestamp', 'date']
        
        # คอลัมน์ที่ต้องทำ normalization
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        if method == 'zscore':
            logger.info("กำลังทำ Z-score normalization")
            # Z-score normalization: (x - mean) / std
            for col in feature_columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = (df[col] - mean) / std
                else:
                    logger.warning(f"คอลัมน์ {col} มีค่า std เป็น 0 ไม่สามารถทำ Z-score normalization ได้")
        
        elif method == 'minmax':
            logger.info("กำลังทำ Min-Max normalization")
            # Min-Max normalization: (x - min) / (max - min)
            for col in feature_columns:
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    logger.warning(f"คอลัมน์ {col} มีค่า max เท่ากับ min ไม่สามารถทำ Min-Max normalization ได้")
        
        else:
            raise ValueError(f"ไม่รู้จักวิธี normalization: {method}")
        
        # บันทึก DataFrame ที่ทำ normalization แล้ว
        self.df_normalized = df
        
        logger.info(f"ทำ normalization แบบ {method} เสร็จสิ้น")
        
        return df
    
    def handle_missing_values(self, method='forward'):
        """
        จัดการกับข้อมูลที่หายไป
        
        Args:
            method (str): วิธีการจัดการ ('forward', 'backward', 'interpolate', 'drop')
            
        Returns:
            pd.DataFrame: DataFrame ที่จัดการกับข้อมูลที่หายไปแล้ว
        """
        if hasattr(self, 'df_normalized'):
            df = self.df_normalized.copy()
        elif hasattr(self, 'df_features'):
            df = self.df_features.copy()
        else:
            df = self.df.copy()
        
        # จำนวนข้อมูลก่อนจัดการ
        total_rows = len(df)
        
        # ตรวจสอบข้อมูลที่หายไป
        missing_values = df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        
        if len(missing_values) > 0:
            logger.info(f"พบข้อมูลที่หายไป:\n{missing_values}")
            
            if method == 'forward':
                logger.info("กำลังใช้วิธี forward fill")
                df.fillna(method='ffill', inplace=True)
                
            elif method == 'backward':
                logger.info("กำลังใช้วิธี backward fill")
                df.fillna(method='bfill', inplace=True)
                
            elif method == 'interpolate':
                logger.info("กำลังใช้วิธี interpolation")
                df.interpolate(method='linear', inplace=True)
                
            elif method == 'drop':
                logger.info("กำลังใช้วิธี drop rows")
                df.dropna(inplace=True)
                
            else:
                raise ValueError(f"ไม่รู้จักวิธีจัดการกับข้อมูลที่หายไป: {method}")
            
            # ตรวจสอบข้อมูลที่หายไปหลังจากจัดการ
            remaining_missing = df.isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"ยังมีข้อมูลที่หายไปเหลืออยู่ {remaining_missing} ค่า")
            
            # จำนวนข้อมูลหลังจัดการ
            new_total_rows = len(df)
            if method == 'drop' and new_total_rows < total_rows:
                logger.info(f"ลบแถวที่มีข้อมูลหายไปแล้ว {total_rows - new_total_rows} แถว")
        else:
            logger.info("ไม่พบข้อมูลที่หายไป")
        
        # บันทึก DataFrame ที่จัดการกับข้อมูลที่หายไปแล้ว
        self.df_cleaned = df
        
        return df
    
    def select_features(self, features=None):
        """
        เลือก features ที่ต้องการใช้
        
        Args:
            features (list): รายการ features ที่ต้องการ (ถ้าไม่ระบุจะใช้ค่าจาก config)
            
        Returns:
            pd.DataFrame: DataFrame ที่มีเฉพาะ features ที่เลือก
        """
        if hasattr(self, 'df_cleaned'):
            df = self.df_cleaned.copy()
        elif hasattr(self, 'df_normalized'):
            df = self.df_normalized.copy()
        elif hasattr(self, 'df_features'):
            df = self.df_features.copy()
        else:
            df = self.df.copy()
        
        # ถ้าไม่ระบุ features ใช้ค่าจาก config
        if features is None:
            # รวม technical indicators และ price features จาก config
            features = DATA_CONFIG['technical_indicators'] + DATA_CONFIG['price_features']
        
        # ตรวจสอบว่า features ทั้งหมดมีอยู่ใน DataFrame หรือไม่
        missing_features = [feat for feat in features if feat not in df.columns]
        if missing_features:
            logger.warning(f"ไม่พบ features ต่อไปนี้ในข้อมูล: {missing_features}")
            # ลบ features ที่ไม่มีออกจากรายการ
            features = [feat for feat in features if feat in df.columns]
        
        # ต้องมี timestamp และ date เสมอ (ถ้ามี)
        if 'timestamp' in df.columns and 'timestamp' not in features:
            features.insert(0, 'timestamp')
        if 'date' in df.columns and 'date' not in features:
            features.insert(1, 'date')
        
        # สร้าง DataFrame ใหม่ที่มีเฉพาะ features ที่เลือก
        selected_df = df[features].copy()
        
        # บันทึก DataFrame ที่เลือก features แล้ว
        self.df_selected = selected_df
        
        logger.info(f"เลือก {len(features)} features เรียบร้อยแล้ว")
        
        return selected_df
    
    def save_processed_data(self, file_name=None):
        """
        บันทึกข้อมูลที่ผ่านการประมวลผลแล้ว
        
        Args:
            file_name (str): ชื่อไฟล์ที่ต้องการบันทึก (ถ้าไม่ระบุจะสร้างชื่อจากพารามิเตอร์)
            
        Returns:
            str: ที่อยู่ไฟล์ที่บันทึก
        """
        # ใช้ DataFrame ล่าสุดที่มีการประมวลผล
        if hasattr(self, 'df_selected'):
            df = self.df_selected.copy()
        elif hasattr(self, 'df_cleaned'):
            df = self.df_cleaned.copy()
        elif hasattr(self, 'df_normalized'):
            df = self.df_normalized.copy()
        elif hasattr(self, 'df_features'):
            df = self.df_features.copy()
        else:
            df = self.df.copy()
        
        # สร้างชื่อไฟล์อัตโนมัติถ้าไม่ได้ระบุ
        if file_name is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            file_name = f"processed_features_{timestamp}.csv"
        
        # ตรวจสอบว่ามีโฟลเดอร์ features หรือไม่
        features_dir = os.path.join(DATA_CONFIG['data_dir'], 'features')
        if not os.path.exists(features_dir):
            os.makedirs(features_dir)
        
        # ที่อยู่ไฟล์เต็ม
        file_path = os.path.join(features_dir, file_name)
        
        # บันทึกไฟล์
        df.to_csv(file_path, index=False)
        logger.info(f"บันทึกข้อมูลที่ประมวลผลแล้วไปที่ {file_path}")
        
        return file_path
    
    def process_data_pipeline(self, normalize_method='zscore', missing_method='forward', file_name=None):
        """
        ประมวลผลข้อมูลทั้งหมดในขั้นตอนเดียว
        
        Args:
            normalize_method (str): วิธีการทำ normalization ('zscore', 'minmax')
            missing_method (str): วิธีการจัดการกับข้อมูลที่หายไป ('forward', 'backward', 'interpolate', 'drop')
            file_name (str): ชื่อไฟล์ที่ต้องการบันทึก
            
        Returns:
            pd.DataFrame: DataFrame ที่ผ่านการประมวลผลแล้ว
        """
        logger.info("เริ่มต้นกระบวนการประมวลผลข้อมูลทั้งหมด")
        
        # 1. สร้าง technical indicators
        self.add_technical_indicators()
        
        # 2. จัดการกับข้อมูลที่หายไป
        self.handle_missing_values(method=missing_method)
        
        # 3. ทำ normalization
        self.normalize_features(method=normalize_method)
        
        # 4. เลือก features
        self.select_features()
        
        # 5. บันทึกข้อมูล
        file_path = self.save_processed_data(file_name)
        
        logger.info(f"กระบวนการประมวลผลข้อมูลเสร็จสิ้น บันทึกไปที่ {file_path}")
        
        return self.df_selected


if __name__ == "__main__":
    # ตัวอย่างการใช้งาน
    from data.data_collector import BinanceDataCollector
    
    # 1. เก็บข้อมูล
    collector = BinanceDataCollector(symbol='BTCUSDT', interval='1h', testnet=True)
    df = collector.get_historical_klines()
    
    # 2. สร้าง features
    fe = FeatureEngineering(df=df)
    processed_df = fe.process_data_pipeline()
    
    print(f"ข้อมูลที่ผ่านการประมวลผลแล้วมีขนาด: {processed_df.shape}")
    print(f"คอลัมน์: {processed_df.columns.tolist()}")
