"""
ตัวประมวลผลข้อมูล (Data Processor) สำหรับเตรียมข้อมูลและคำนวณตัวชี้วัดทางเทคนิค
"""

import numpy as np
import pandas as pd
import os
import glob
import ta
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from .data_collector import BinanceDataCollector
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    คลาสสำหรับการเตรียมข้อมูลและคำนวณตัวชี้วัดทางเทคนิคสำหรับ Crypto Trading Bot
    """
    
    def __init__(self, data_dir: str = 'data'):
        """
        กำหนดค่าเริ่มต้นของตัวประมวลผลข้อมูล
        
        Args:
            data_dir (str): โฟลเดอร์ที่เก็บไฟล์ข้อมูล
        """
        self.data_dir = data_dir
        
    def load_data(self, symbol: str, timeframe: str = '1h', 
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        โหลดข้อมูลราคาจากไฟล์
        
        Args:
            symbol (str): สัญลักษณ์คู่เหรียญ เช่น 'BTCUSDT'
            timeframe (str): กรอบเวลา เช่น '1m', '5m', '1h', '1d'
            start_date (str, optional): วันที่เริ่มต้น ในรูปแบบ 'YYYY-MM-DD'
            end_date (str, optional): วันที่สิ้นสุด ในรูปแบบ 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame ที่มีข้อมูลราคา
        """
        # ค้นหาไฟล์ข้อมูล
        filepath_pattern = os.path.join(self.data_dir, f"{symbol}_{timeframe}_*.csv")
        filepaths = glob.glob(filepath_pattern)
        
        if not filepaths:
            print(f"ไม่พบไฟล์ข้อมูลสำหรับ {symbol} ที่กรอบเวลา {timeframe} กำลังดึงข้อมูลจาก Binance...")
            # สร้าง instance ของ BinanceDataCollector
            collector = BinanceDataCollector(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                testnet=True  # ใช้ testnet สำหรับการทดสอบ
            )
            # ดึงข้อมูลจาก Binance
            df = collector.get_historical_klines()
            return df
        
        # อ่านและรวมข้อมูลจากทุกไฟล์ที่พบ
        dfs = []
        for filepath in filepaths:
            df = pd.read_csv(filepath)
            dfs.append(df)
        
        if not dfs:
            print(f"ไม่สามารถโหลดข้อมูลสำหรับ {symbol} กำลังดึงข้อมูลจาก Binance...")
            # สร้าง instance ของ BinanceDataCollector
            collector = BinanceDataCollector(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                testnet=True  # ใช้ testnet สำหรับการทดสอบ
            )
            # ดึงข้อมูลจาก Binance
            df = collector.get_historical_klines()
            return df
        
        # รวมข้อมูลและจัดเรียงตามเวลา
        data = pd.concat(dfs)
        
        # ตรวจสอบและเพิ่มคอลัมน์เวลา
        if 'timestamp' not in data.columns and 'time' in data.columns:
            data = data.rename(columns={'time': 'timestamp'})
        
        # แปลงคอลัมน์เวลาให้เป็นรูปแบบ datetime
        if 'timestamp' in data.columns:
            if pd.api.types.is_numeric_dtype(data['timestamp']):
                # ถ้าเป็นตัวเลข (timestamp) ให้แปลงเป็น datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            else:
                # ถ้าเป็นสตริง ให้แปลงเป็น datetime
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # กรองตามช่วงวันที่
        if start_date:
            start_date = pd.to_datetime(start_date)
            data = data[data['timestamp'] >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            data = data[data['timestamp'] <= end_date]
        
        # จัดเรียงข้อมูลตามเวลา
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # ตรวจสอบว่ามีคอลัมน์ที่จำเป็นหรือไม่
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"ไม่พบคอลัมน์ {col} ในข้อมูล")
        
        return data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        เพิ่ม technical indicators ให้กับข้อมูล
        
        Args:
            df (pd.DataFrame): ข้อมูลที่ต้องการเพิ่ม indicators
            
        Returns:
            pd.DataFrame: ข้อมูลที่มี indicators เพิ่มเติม
        """
        try:
            # ตรวจสอบข้อมูลที่เข้ามา
            logger.info("ข้อมูลก่อนเพิ่ม technical indicators:")
            logger.info(f"\n{df.dtypes}")
            
            # ตรวจสอบคอลัมน์ที่จำเป็น
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"ไม่พบคอลัมน์ {col} ในข้อมูล")
            
            # แปลงข้อมูลเป็น numeric
            df_with_indicators = df.copy()
            for col in df_with_indicators.columns:
                if col not in ['date', 'timestamp']:
                    df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
            
            # ตรวจสอบและจัดการค่า NaN
            nan_counts = df_with_indicators.isna().sum()
            if nan_counts.any():
                logger.warning(f"พบค่า NaN ในข้อมูล: {nan_counts[nan_counts > 0]}")
                # แทนที่ค่า NaN ด้วยค่าเฉลี่ย
                df_with_indicators = df_with_indicators.fillna(df_with_indicators.mean())
            
            # ตรวจสอบและจัดการค่า inf
            inf_mask = np.isinf(df_with_indicators.select_dtypes(include=np.number))
            if inf_mask.any().any():
                logger.warning("พบค่า inf ในข้อมูล กำลังแทนที่ด้วยค่า NaN")
                df_with_indicators = df_with_indicators.replace([np.inf, -np.inf], np.nan)
                df_with_indicators = df_with_indicators.fillna(df_with_indicators.mean())
            
            # ตรวจสอบและจัดการค่าลบ
            for col in df_with_indicators.select_dtypes(include=np.number).columns:
                if (df_with_indicators[col] < 0).any():
                    logger.warning(f"พบค่าลบในคอลัมน์ {col} กำลังแปลงเป็นค่าสัมบูรณ์")
                    df_with_indicators[col] = df_with_indicators[col].abs()
            
            # ตรวจสอบและจัดการค่า 0
            for col in df_with_indicators.select_dtypes(include=np.number).columns:
                if (df_with_indicators[col] == 0).any():
                    logger.warning(f"พบค่า 0 ในคอลัมน์ {col} กำลังแทนที่ด้วยค่าเฉลี่ย")
                    mean_value = df_with_indicators[col].mean()
                    df_with_indicators[col] = df_with_indicators[col].replace(0, mean_value)
            
            # ตรวจสอบความยาวของข้อมูล
            if len(df_with_indicators) < 100:
                raise ValueError(f"ข้อมูลไม่เพียงพอ ต้องการอย่างน้อย 100 แท่ง แต่มีเพียง {len(df_with_indicators)} แท่ง")
            
            # ตรวจสอบการเรียงลำดับข้อมูล
            if not df_with_indicators.index.is_monotonic_increasing:
                logger.warning("ข้อมูลไม่ได้เรียงตามเวลา กำลังเรียงลำดับใหม่")
                df_with_indicators = df_with_indicators.sort_index()
            
            # คำนวณ indicators
            try:
                # Moving Averages
                df_with_indicators['sma_7'] = ta.trend.sma_indicator(df_with_indicators['close'], window=7, fillna=True)
                df_with_indicators['sma_25'] = ta.trend.sma_indicator(df_with_indicators['close'], window=25, fillna=True)
                df_with_indicators['sma_99'] = ta.trend.sma_indicator(df_with_indicators['close'], window=99, fillna=True)
                df_with_indicators['ema_7'] = ta.trend.ema_indicator(df_with_indicators['close'], window=7, fillna=True)
                df_with_indicators['ema_25'] = ta.trend.ema_indicator(df_with_indicators['close'], window=25, fillna=True)
                df_with_indicators['ema_99'] = ta.trend.ema_indicator(df_with_indicators['close'], window=99, fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Moving Averages: {str(e)}")
                raise
            
            try:
                # RSI
                df_with_indicators['rsi_14'] = ta.momentum.rsi(df_with_indicators['close'], window=14, fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ RSI: {str(e)}")
                raise
            
            try:
                # MACD
                macd = ta.trend.MACD(df_with_indicators['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
                df_with_indicators['macd'] = macd.macd()
                df_with_indicators['macd_signal'] = macd.macd_signal()
                df_with_indicators['macd_hist'] = macd.macd_diff()
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ MACD: {str(e)}")
                raise
            
            try:
                # Bollinger Bands
                bollinger = ta.volatility.BollingerBands(df_with_indicators['close'], window=20, window_dev=2, fillna=True)
                df_with_indicators['bb_upper'] = bollinger.bollinger_hband()
                df_with_indicators['bb_middle'] = bollinger.bollinger_mavg()
                df_with_indicators['bb_lower'] = bollinger.bollinger_lband()
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Bollinger Bands: {str(e)}")
                raise
            
            try:
                # Stochastic Oscillator
                stoch = ta.momentum.StochasticOscillator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=14, smooth_window=3, fillna=True)
                df_with_indicators['stoch_k'] = stoch.stoch()
                df_with_indicators['stoch_d'] = stoch.stoch_signal()
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Stochastic Oscillator: {str(e)}")
                raise
            
            try:
                # ADX
                df_with_indicators['adx'] = ta.trend.adx(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=14, fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ ADX: {str(e)}")
                raise
            
            try:
                # OBV
                df_with_indicators['obv'] = ta.volume.on_balance_volume(df_with_indicators['close'], df_with_indicators['volume'], fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ OBV: {str(e)}")
                raise
            
            try:
                # ATR
                df_with_indicators['atr'] = ta.volatility.average_true_range(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=14, fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ ATR: {str(e)}")
                raise
            
            try:
                # CCI
                df_with_indicators['cci'] = ta.trend.cci(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=20, fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ CCI: {str(e)}")
                raise
            
            try:
                # MFI
                df_with_indicators['mfi'] = ta.volume.money_flow_index(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], df_with_indicators['volume'], window=14, fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ MFI: {str(e)}")
                raise
            
            try:
                # ROC
                df_with_indicators['roc'] = ta.momentum.roc(df_with_indicators['close'], window=12, fillna=True)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ ROC: {str(e)}")
                raise
            
            try:
                # Price to Moving Average Ratios
                df_with_indicators['close_sma_7_pct'] = df_with_indicators['close'] / df_with_indicators['sma_7'] - 1
                df_with_indicators['close_sma_25_pct'] = df_with_indicators['close'] / df_with_indicators['sma_25'] - 1
                df_with_indicators['close_sma_99_pct'] = df_with_indicators['close'] / df_with_indicators['sma_99'] - 1
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Price to Moving Average Ratios: {str(e)}")
                raise
            
            try:
                # Price Changes
                df_with_indicators['close_pct_change_1'] = df_with_indicators['close'].pct_change(1).fillna(0)
                df_with_indicators['close_pct_change_5'] = df_with_indicators['close'].pct_change(5).fillna(0)
                df_with_indicators['close_pct_change_10'] = df_with_indicators['close'].pct_change(10).fillna(0)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Price Changes: {str(e)}")
                raise
            
            try:
                # Volatility
                df_with_indicators['volatility_5'] = df_with_indicators['close'].rolling(window=5).std().fillna(0)
                df_with_indicators['volatility_15'] = df_with_indicators['close'].rolling(window=15).std().fillna(0)
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Volatility: {str(e)}")
                raise
            
            try:
                # Volume Indicators
                df_with_indicators['volume_sma_5'] = ta.trend.sma_indicator(df_with_indicators['volume'], window=5, fillna=True)
                df_with_indicators['volume_sma_20'] = ta.trend.sma_indicator(df_with_indicators['volume'], window=20, fillna=True)
                df_with_indicators['volume_ratio'] = df_with_indicators['volume'] / df_with_indicators['volume_sma_20']
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการคำนวณ Volume Indicators: {str(e)}")
                raise
            
            # ตรวจสอบค่า NaN หลังคำนวณ
            nan_counts = df_with_indicators.isna().sum()
            if nan_counts.any():
                logger.warning(f"พบค่า NaN ในข้อมูลหลังคำนวณ indicators:\n{nan_counts[nan_counts > 0]}")
                # แทนที่ค่า NaN ด้วยค่าเฉลี่ย
                df_with_indicators = df_with_indicators.fillna(df_with_indicators.mean())
            
            # ตรวจสอบความยาวของข้อมูล
            if len(df_with_indicators) != len(df):
                logger.warning(f"ความยาวของข้อมูลเปลี่ยนจาก {len(df)} เป็น {len(df_with_indicators)}")
                # ตัดข้อมูลให้มีความยาวเท่ากับข้อมูลเดิม
                df_with_indicators = df_with_indicators.iloc[-len(df):]
            
            return df_with_indicators
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณ technical indicators: {str(e)}")
            raise
    
    def normalize_data(self, df: pd.DataFrame, columns_to_exclude: List[str] = ['timestamp']) -> pd.DataFrame:
        """
        ปรับข้อมูลให้อยู่ในรูปแบบปกติ (Normalized) เพื่อให้เหมาะกับการเรียนรู้ของโมเดล
        
        Args:
            df (pd.DataFrame): DataFrame ที่ต้องการปรับให้เป็นปกติ
            columns_to_exclude (List[str]): คอลัมน์ที่ไม่ต้องการปรับ
            
        Returns:
            pd.DataFrame: DataFrame ที่ถูกปรับให้เป็นปกติแล้ว
        """
        try:
            df_normalized = df.copy()
            
            # เลือกเฉพาะคอลัมน์ที่ต้องการปรับให้เป็นปกติ
            columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]
            
            for col in columns_to_normalize:
                # แปลงเป็นตัวเลข
                df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
                
                # ตรวจสอบจำนวนค่า NaN
                nan_count = df_normalized[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"พบค่า NaN {nan_count} ค่าในคอลัมน์ {col}")
                    # แทนที่ค่า NaN ด้วยค่าเฉลี่ย
                    mean_value = df_normalized[col].mean()
                    df_normalized[col] = df_normalized[col].fillna(mean_value)
                
                # ตรวจสอบและแทนที่ค่า inf และ -inf
                inf_count = df_normalized[col].isin([np.inf, -np.inf]).sum()
                if inf_count > 0:
                    logger.warning(f"พบค่า inf หรือ -inf {inf_count} ค่าในคอลัมน์ {col}")
                    # แทนที่ด้วยค่า NaN แล้วใช้ค่าเฉลี่ย
                    df_normalized[col] = df_normalized[col].replace([np.inf, -np.inf], np.nan)
                    mean_value = df_normalized[col].mean()
                    df_normalized[col] = df_normalized[col].fillna(mean_value)
                
                # ตรวจสอบว่าคอลัมน์มีค่า min และ max ที่แตกต่างกันหรือไม่
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                
                if min_val != max_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                else:
                    # ถ้า min และ max เท่ากัน ให้ตั้งค่าเป็น 0.5
                    df_normalized[col] = 0.5
                
                # ตรวจสอบว่ามีค่าอยู่นอกช่วง [0, 1] หรือไม่
                if (df_normalized[col] < 0).any() or (df_normalized[col] > 1).any():
                    logger.warning(f"พบค่าอยู่นอกช่วง [0, 1] ในคอลัมน์ {col} กำลังปรับให้อยู่ในช่วง...")
                    df_normalized[col] = df_normalized[col].clip(0, 1)
            
            return df_normalized
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการ normalize ข้อมูล: {str(e)}")
            raise
    
    def prepare_data_for_training(self, symbol: str, timeframe: str = '1h', 
                                 start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        เตรียมข้อมูลสำหรับการฝึกสอนโมเดล โดยแบ่งเป็นชุดฝึกสอน (training), ชุดตรวจสอบ (validation), และชุดทดสอบ (testing)
        
        Args:
            symbol (str): สัญลักษณ์คู่เหรียญ
            timeframe (str): กรอบเวลา
            start_date (str, optional): วันที่เริ่มต้น
            end_date (str, optional): วันที่สิ้นสุด
            train_ratio (float): สัดส่วนของข้อมูลที่ใช้ในการฝึกสอน
            val_ratio (float): สัดส่วนของข้อมูลที่ใช้ในการตรวจสอบ
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: ข้อมูลสำหรับการฝึกสอน, ตรวจสอบ, และทดสอบ
        """
        # โหลดข้อมูล
        df = self.load_data(symbol, timeframe, start_date, end_date)
        
        # เพิ่มตัวชี้วัดทางเทคนิค
        df_with_indicators = self.add_technical_indicators(df)
        
        # ปรับข้อมูลให้เป็นปกติ
        df_normalized = self.normalize_data(df_with_indicators)
        
        # แบ่งข้อมูลตามสัดส่วน
        total_rows = len(df_normalized)
        train_size = int(total_rows * train_ratio)
        val_size = int(total_rows * val_ratio)
        
        train_data = df_normalized.iloc[:train_size]
        val_data = df_normalized.iloc[train_size:train_size+val_size]
        test_data = df_normalized.iloc[train_size+val_size:]
        
        return train_data, val_data, test_data
    
    def create_features_for_backtesting(self, symbol: str, timeframe: str = '1h', 
                                       start_date: Optional[str] = None, 
                                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        สร้างคุณลักษณะ (features) สำหรับการทดสอบย้อนหลัง (backtesting)
        
        Args:
            symbol (str): สัญลักษณ์คู่เหรียญ
            timeframe (str): กรอบเวลา
            start_date (str, optional): วันที่เริ่มต้น
            end_date (str, optional): วันที่สิ้นสุด
            
        Returns:
            pd.DataFrame: DataFrame ที่มีคุณลักษณะสำหรับการทดสอบย้อนหลัง
        """
        # โหลดข้อมูล
        df = self.load_data(symbol, timeframe, start_date, end_date)
        
        # เพิ่มตัวชี้วัดทางเทคนิค
        df_with_indicators = self.add_technical_indicators(df)
        
        # ปรับข้อมูลให้เป็นปกติ (ยกเว้นข้อมูลที่เกี่ยวกับราคาจริง)
        columns_to_exclude = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # สร้าง DataFrame ใหม่ที่มีทั้งข้อมูลราคาจริงและข้อมูลที่ปรับให้เป็นปกติแล้ว
        df_features = df_with_indicators.copy()
        
        # ปรับคอลัมน์ที่ต้องการให้เป็นปกติ
        columns_to_normalize = [col for col in df_features.columns if col not in columns_to_exclude]
        
        for col in columns_to_normalize:
            if df_features[col].min() != df_features[col].max():
                df_features[col] = (df_features[col] - df_features[col].min()) / (df_features[col].max() - df_features[col].min())
            else:
                df_features[col] = 0.5
                
        return df_features