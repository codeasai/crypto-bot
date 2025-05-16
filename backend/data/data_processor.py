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
        เพิ่มตัวชี้วัดทางเทคนิคเข้าไปในข้อมูล
        
        Args:
            df (pd.DataFrame): DataFrame ที่มีข้อมูลราคา OHLCV
            
        Returns:
            pd.DataFrame: DataFrame ที่เพิ่มตัวชี้วัดทางเทคนิคแล้ว
        """
        # ตรวจสอบความยาวของข้อมูล
        if len(df) < 100:  # ต้องการข้อมูลอย่างน้อย 100 แท่ง
            raise ValueError(f"ข้อมูลไม่เพียงพอ ต้องการอย่างน้อย 100 แท่ง แต่มีเพียง {len(df)} แท่ง")
            
        # สร้าง DataFrame ใหม่เพื่อไม่เปลี่ยนแปลงข้อมูลต้นฉบับ
        df_with_indicators = df.copy()
        
        # 1. Moving Averages
        df_with_indicators['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
        df_with_indicators['sma_25'] = ta.trend.sma_indicator(df['close'], window=25)
        df_with_indicators['sma_99'] = ta.trend.sma_indicator(df['close'], window=99)
        
        df_with_indicators['ema_7'] = ta.trend.ema_indicator(df['close'], window=7)
        df_with_indicators['ema_25'] = ta.trend.ema_indicator(df['close'], window=25)
        df_with_indicators['ema_99'] = ta.trend.ema_indicator(df['close'], window=99)
        
        # 2. RSI (Relative Strength Index)
        df_with_indicators['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        
        # 3. MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df_with_indicators['macd'] = macd.macd()
        df_with_indicators['macd_signal'] = macd.macd_signal()
        df_with_indicators['macd_hist'] = macd.macd_diff()
        
        # 4. Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df_with_indicators['bb_upper'] = bollinger.bollinger_hband()
        df_with_indicators['bb_middle'] = bollinger.bollinger_mavg()
        df_with_indicators['bb_lower'] = bollinger.bollinger_lband()
        
        # 5. Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
        df_with_indicators['stoch_k'] = stoch.stoch()
        df_with_indicators['stoch_d'] = stoch.stoch_signal()
        
        # 6. ADX (Average Directional Index)
        try:
            df_with_indicators['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        except ValueError as e:
            print(f"ไม่สามารถคำนวณ ADX ได้: {e}")
            df_with_indicators['adx'] = np.nan
        
        # 7. OBV (On-Balance Volume)
        df_with_indicators['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # 8. ATR (Average True Range) - ค่าความผันผวน
        try:
            df_with_indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        except ValueError as e:
            print(f"ไม่สามารถคำนวณ ATR ได้: {e}")
            df_with_indicators['atr'] = np.nan
        
        # 9. CCI (Commodity Channel Index)
        try:
            df_with_indicators['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)
        except ValueError as e:
            print(f"ไม่สามารถคำนวณ CCI ได้: {e}")
            df_with_indicators['cci'] = np.nan
        
        # 10. MFI (Money Flow Index) - ปริมาณการไหลของเงินทุน
        try:
            df_with_indicators['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=14)
        except ValueError as e:
            print(f"ไม่สามารถคำนวณ MFI ได้: {e}")
            df_with_indicators['mfi'] = np.nan
        
        # 11. Price Rate of Change
        df_with_indicators['roc'] = ta.momentum.roc(df['close'], window=10)
        
        # คำนวณราคาเทียบกับค่าเฉลี่ยเคลื่อนที่ (เปอร์เซ็นต์)
        df_with_indicators['close_sma_7_pct'] = (df['close'] - df_with_indicators['sma_7']) / df_with_indicators['sma_7']
        df_with_indicators['close_sma_25_pct'] = (df['close'] - df_with_indicators['sma_25']) / df_with_indicators['sma_25']
        df_with_indicators['close_sma_99_pct'] = (df['close'] - df_with_indicators['sma_99']) / df_with_indicators['sma_99']
        
        # คำนวณการเปลี่ยนแปลงของราคา (เปอร์เซ็นต์)
        df_with_indicators['close_pct_change_1'] = df['close'].pct_change(1)
        df_with_indicators['close_pct_change_5'] = df['close'].pct_change(5)
        df_with_indicators['close_pct_change_10'] = df['close'].pct_change(10)
        
        # คำนวณความผันผวนย้อนหลัง
        df_with_indicators['volatility_5'] = df['close'].pct_change().rolling(5).std()
        df_with_indicators['volatility_15'] = df['close'].pct_change().rolling(15).std()
        
        # ความสัมพันธ์กับปริมาณการซื้อขาย
        df_with_indicators['volume_sma_5'] = ta.trend.sma_indicator(df['volume'], window=5)
        df_with_indicators['volume_sma_20'] = ta.trend.sma_indicator(df['volume'], window=20)
        df_with_indicators['volume_ratio'] = df['volume'] / df_with_indicators['volume_sma_5']
        
        # ลบแถวที่มีค่า NaN
        df_with_indicators = df_with_indicators.dropna()
        
        return df_with_indicators
    
    def normalize_data(self, df: pd.DataFrame, columns_to_exclude: List[str] = ['timestamp']) -> pd.DataFrame:
        """
        ปรับข้อมูลให้อยู่ในรูปแบบปกติ (Normalized) เพื่อให้เหมาะกับการเรียนรู้ของโมเดล
        
        Args:
            df (pd.DataFrame): DataFrame ที่ต้องการปรับให้เป็นปกติ
            columns_to_exclude (List[str]): คอลัมน์ที่ไม่ต้องการปรับ
            
        Returns:
            pd.DataFrame: DataFrame ที่ถูกปรับให้เป็นปกติแล้ว
        """
        df_normalized = df.copy()
        
        # เลือกเฉพาะคอลัมน์ที่ต้องการปรับให้เป็นปกติ
        columns_to_normalize = [col for col in df.columns if col not in columns_to_exclude]
        
        for col in columns_to_normalize:
            # ตรวจสอบว่าคอลัมน์มีค่า min และ max ที่แตกต่างกันหรือไม่
            if df[col].min() != df[col].max():
                df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            else:
                # ถ้า min และ max เท่ากัน ให้ตั้งค่าเป็น 0 หรือ 0.5
                df_normalized[col] = 0.5
                
        return df_normalized
    
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