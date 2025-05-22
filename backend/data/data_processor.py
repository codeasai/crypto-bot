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
import datetime # Added for save_processed_data

# Attempt to import DATA_CONFIG, but provide a fallback if not found
try:
    from backend.config.config import DATA_CONFIG
except ImportError:
    logger.warning("Could not import DATA_CONFIG from backend.config.config. Using default values.")
    DATA_CONFIG = {
        'data_dir': 'data/features', # Default data directory for saving processed data
        'technical_indicators': [ # Default list of technical indicators
            'sma_10', 'sma_30', 'sma_50', 'ema_10', 'ema_30', 'rsi_14', 
            'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 
            'bb_lower', 'bb_pct_b', 'atr_14', 'daily_return'
        ],
        'price_features': ['open', 'high', 'low', 'close', 'volume'] # Default price features
    }


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
        self.df = None # Will store the current working DataFrame
        self.df_features = None # For DataFrame with added features
        self.df_normalized = None # For normalized DataFrame
        self.df_cleaned = None # For cleaned DataFrame
        self.df_selected = None # For DataFrame with selected features

    def load_data(self, symbol: str, timeframe: str = '1h', 
                 start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Loads price data from files or fetches from Binance if files are not found.
        Saves the loaded data to self.df.
        
        Args:
            symbol (str): Cryptocurrency symbol, e.g., 'BTCUSDT'
            timeframe (str): Timeframe, e.g., '1m', '5m', '1h', '1d'
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: DataFrame with price data
        """
        # Define data directory for raw klines
        raw_data_dir = os.path.join(self.data_dir, 'datasets') # Changed to look into 'datasets' subdirectory
        os.makedirs(raw_data_dir, exist_ok=True) # Ensure directory exists

        # Search for data files
        filepath_pattern = os.path.join(raw_data_dir, f"{symbol}_{timeframe}_*.csv")
        filepaths = glob.glob(filepath_pattern)
        
        if not filepaths:
            logger.info(f"No data files found for {symbol} timeframe {timeframe} in {raw_data_dir}. Attempting to fetch from Binance...")
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
            df_temp = pd.read_csv(filepath) # Renamed to df_temp to avoid conflict with self.df
            dfs.append(df_temp)
        
        if not dfs:
            logger.info(f"No data files found for {symbol} timeframe {timeframe}. Fetching from Binance...")
            collector = BinanceDataCollector(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                # testnet=True, # Consider making this configurable or removing if not always testnet
                data_dir=raw_data_dir # Pass the correct data_dir for saving
            )
            data = collector.get_historical_klines() # Fetched data
            if data.empty:
                logger.error(f"Failed to fetch data for {symbol} from Binance.")
                self.df = pd.DataFrame() # Set self.df to empty DataFrame
                return self.df 
            # Save the fetched data
            # filename = f"{symbol}_{timeframe}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            # save_path = os.path.join(raw_data_dir, filename)
            # data.to_csv(save_path, index=False)
            # logger.info(f"Saved fetched data to {save_path}")

        else:
            # Combine data from all found files
            data = pd.concat(dfs)
        
        # Ensure 'timestamp' column exists and is of datetime type
        if 'timestamp' not in data.columns and 'time' in data.columns:
            data = data.rename(columns={'time': 'timestamp'})
        
        if 'timestamp' in data.columns:
            if pd.api.types.is_numeric_dtype(data['timestamp']):
                data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            else:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
        else: # If no timestamp column, try to use index if it's datetime
            if isinstance(data.index, pd.DatetimeIndex):
                data['timestamp'] = data.index
            else: # if no time information at all, raise error.
                logger.error("Timestamp column not found and index is not datetime. Cannot proceed.")
                self.df = pd.DataFrame()
                return self.df


        # Filter by date range
        if start_date:
            start_date_dt = pd.to_datetime(start_date)
            data = data[data['timestamp'] >= start_date_dt]
        if end_date:
            end_date_dt = pd.to_datetime(end_date)
            data = data[data['timestamp'] <= end_date_dt]
        
        # Sort data by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Verify required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                # raise ValueError(f"ไม่พบคอลัมน์ {col} ในข้อมูล") # Original error
                self.df = pd.DataFrame() # Set self.df to empty DataFrame
                return self.df 

        self.df = data.copy() # Store loaded data in self.df
        logger.info(f"Data loaded successfully for {symbol} with shape {self.df.shape}")
        return self.df

    def handle_missing_values(self, df: pd.DataFrame, method: str = 'forward') -> pd.DataFrame:
        """
        Handles missing data in the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            method (str): Method to handle missing values ('forward', 'backward', 'interpolate', 'drop', 'mean').
            
        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        if df is None or df.empty:
            logger.warning("Input DataFrame for handle_missing_values is None or empty.")
            return pd.DataFrame()

        df_cleaned = df.copy()
        total_rows = len(df_cleaned)
        missing_values = df_cleaned.isnull().sum()
        missing_values = missing_values[missing_values > 0]

        if not missing_values.empty:
            logger.info(f"Found missing values:\n{missing_values}")
            if method == 'forward':
                logger.info("Applying forward fill for missing values.")
                df_cleaned.fillna(method='ffill', inplace=True)
            elif method == 'backward':
                logger.info("Applying backward fill for missing values.")
                df_cleaned.fillna(method='bfill', inplace=True)
            elif method == 'interpolate':
                logger.info("Applying linear interpolation for missing values.")
                df_cleaned.interpolate(method='linear', inplace=True)
            elif method == 'drop':
                logger.info("Dropping rows with missing values.")
                df_cleaned.dropna(inplace=True)
            elif method == 'mean':
                logger.info("Filling missing values with column mean.")
                for col in df_cleaned.columns:
                    if df_cleaned[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                        else:
                            logger.warning(f"Column {col} is not numeric, cannot fill with mean. Skipping.")
            else:
                logger.error(f"Unknown missing value handling method: {method}")
                raise ValueError(f"Unknown missing value handling method: {method}")

            remaining_missing = df_cleaned.isnull().sum().sum()
            if remaining_missing > 0:
                logger.warning(f"Still {remaining_missing} missing values remaining after handling. Consider using 'drop' or checking data source.")
            
            if method == 'drop' and len(df_cleaned) < total_rows:
                logger.info(f"Dropped {total_rows - len(df_cleaned)} rows with missing values.")
        else:
            logger.info("No missing values found.")
        
        self.df_cleaned = df_cleaned.copy()
        return self.df_cleaned

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a comprehensive set of technical indicators to the DataFrame.
        This method combines indicators from both original files.
        
        Args:
            df (pd.DataFrame): DataFrame with price data (OHLCV).
            
        Returns:
            pd.DataFrame: DataFrame with added technical indicators.
        """
        if df is None or df.empty:
            logger.warning("Input DataFrame for add_technical_indicators is None or empty.")
            return pd.DataFrame()

        df_with_indicators = df.copy()
        
        try:
            logger.info("Data before adding technical indicators:")
            logger.info(f"\n{df_with_indicators.dtypes}")

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column for TA calculation: {col}")
                    raise ValueError(f"Missing required column for TA calculation: {col}")

            # Convert to numeric, coercing errors
            for col in df_with_indicators.columns:
                if col not in ['date', 'timestamp']: # Assuming 'date' might be another non-numeric col
                    df_with_indicators[col] = pd.to_numeric(df_with_indicators[col], errors='coerce')
            
            # Initial NaN/inf check and fill (using mean for simplicity here, can be customized)
            df_with_indicators.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Call handle_missing_values to manage NaNs before TA calculation
            # This ensures that TA calculations have valid inputs as much as possible
            df_with_indicators = self.handle_missing_values(df_with_indicators, method='mean')


            # --- Start TA calculations (merged and expanded) ---
            logger.info("Calculating technical indicators...")

            # Moving Averages (SMA & EMA)
            sma_windows = [7, 10, 25, 30, 50, 99]
            ema_windows = [7, 10, 25, 30, 99]
            for window in sma_windows:
                df_with_indicators[f'sma_{window}'] = ta.trend.sma_indicator(df_with_indicators['close'], window=window, fillna=True)
            for window in ema_windows:
                df_with_indicators[f'ema_{window}'] = ta.trend.ema_indicator(df_with_indicators['close'], window=window, fillna=True)

            # RSI
            df_with_indicators['rsi_14'] = ta.momentum.rsi(df_with_indicators['close'], window=14, fillna=True)

            # MACD
            macd = ta.trend.MACD(df_with_indicators['close'], window_slow=26, window_fast=12, window_sign=9, fillna=True)
            df_with_indicators['macd'] = macd.macd()
            df_with_indicators['macd_signal'] = macd.macd_signal()
            df_with_indicators['macd_hist'] = macd.macd_diff()

            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df_with_indicators['close'], window=20, window_dev=2, fillna=True)
            df_with_indicators['bb_upper'] = bollinger.bollinger_hband()
            df_with_indicators['bb_middle'] = bollinger.bollinger_mavg()
            df_with_indicators['bb_lower'] = bollinger.bollinger_lband()
            df_with_indicators['bb_pct_b'] = bollinger.bollinger_pband() # From FeatureEngineering

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=14, smooth_window=3, fillna=True)
            df_with_indicators['stoch_k'] = stoch.stoch()
            df_with_indicators['stoch_d'] = stoch.stoch_signal()

            # ADX
            df_with_indicators['adx'] = ta.trend.adx(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=14, fillna=True)

            # OBV
            df_with_indicators['obv'] = ta.volume.on_balance_volume(df_with_indicators['close'], df_with_indicators['volume'], fillna=True)

            # ATR
            df_with_indicators['atr_14'] = ta.volatility.average_true_range(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=14, fillna=True)
            
            # CCI
            df_with_indicators['cci'] = ta.trend.cci(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], window=20, fillna=True)

            # MFI
            df_with_indicators['mfi'] = ta.volume.money_flow_index(df_with_indicators['high'], df_with_indicators['low'], df_with_indicators['close'], df_with_indicators['volume'], window=14, fillna=True)

            # ROC
            df_with_indicators['roc'] = ta.momentum.roc(df_with_indicators['close'], window=12, fillna=True)

            # Price to Moving Average Ratios
            for window in [7, 25, 99]:
                if f'sma_{window}' in df_with_indicators.columns: # Check if SMA was calculated
                     df_with_indicators[f'close_sma_{window}_pct'] = (df_with_indicators['close'] / df_with_indicators[f'sma_{window}']) - 1
                if f'ema_{window}' in df_with_indicators.columns: # Check if EMA was calculated
                     df_with_indicators[f'close_ema_{window}_pct'] = (df_with_indicators['close'] / df_with_indicators[f'ema_{window}']) - 1


            # Price Changes
            for period in [1, 5, 10]:
                df_with_indicators[f'close_pct_change_{period}'] = df_with_indicators['close'].pct_change(period).fillna(0)
            
            # Daily Return (from FeatureEngineering)
            df_with_indicators['daily_return'] = df_with_indicators['close'].pct_change().fillna(0)


            # Volatility (Standard Deviation)
            for window in [5, 15]:
                df_with_indicators[f'volatility_{window}'] = df_with_indicators['close'].rolling(window=window).std().fillna(0)

            # Volume Indicators
            for window in [5, 20]:
                 df_with_indicators[f'volume_sma_{window}'] = ta.trend.sma_indicator(df_with_indicators['volume'], window=window, fillna=True)
            if 'volume_sma_20' in df_with_indicators.columns and df_with_indicators['volume_sma_20'].notna().all() and (df_with_indicators['volume_sma_20'] != 0).all(): # Ensure sma is calculated and not zero
                df_with_indicators['volume_ratio'] = df_with_indicators['volume'] / df_with_indicators['volume_sma_20']
            else:
                df_with_indicators['volume_ratio'] = 0 # Set to 0 or some other placeholder if sma is zero or not available
                logger.warning("volume_sma_20 contains zeros or NaNs, volume_ratio might be incorrect.")


            # --- End TA calculations ---

            # Final check for NaNs created by TA calculations (especially due to window periods)
            # Using ffill as a common strategy for time-series based indicators
            df_with_indicators.fillna(method='ffill', inplace=True) 
            df_with_indicators.fillna(method='bfill', inplace=True) # backfill any remaining NaNs at the beginning

            # Ensure original length if any rows were dropped internally by TA library (though fillna=True should prevent this)
            if len(df_with_indicators) != len(df):
                logger.warning(f"DataFrame length changed from {len(df)} to {len(df_with_indicators)} after adding indicators. Re-aligning.")
                # This might indicate an issue, but for now, re-aligning to original index
                df_with_indicators = df_with_indicators.reindex(df.index, method='ffill').fillna(method='bfill')


            logger.info(f"Finished adding technical indicators. DataFrame shape: {df_with_indicators.shape}")
            self.df_features = df_with_indicators.copy()
            return self.df_features

        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            # Optionally, return the DataFrame as it was before this method, or an empty one
            # For now, re-raise to indicate failure
            raise

    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax', columns_to_exclude: List[str] = ['timestamp', 'date']) -> pd.DataFrame:
        """
        Normalizes features in the DataFrame using either Min-Max or Z-score scaling.
        
        Args:
            df (pd.DataFrame): DataFrame to normalize.
            method (str): Normalization method ('minmax' or 'zscore').
            columns_to_exclude (List[str]): Columns to exclude from normalization.
            
        Returns:
            pd.DataFrame: Normalized DataFrame.
        """
        if df is None or df.empty:
            logger.warning("Input DataFrame for normalize_features is None or empty.")
            return pd.DataFrame()

        df_normalized = df.copy()
        
        feature_columns = [col for col in df_normalized.columns if col not in columns_to_exclude]
        
        logger.info(f"Starting normalization with method: {method}")

        for col in feature_columns:
            df_normalized[col] = pd.to_numeric(df_normalized[col], errors='coerce')
            
            # Handle NaNs and infs that might have been introduced or missed
            if df_normalized[col].isnull().any() or np.isinf(df_normalized[col]).any():
                logger.warning(f"NaN or inf values found in column '{col}' before normalization. Filling with mean.")
                mean_val = df_normalized[col].mean() # Calculate mean before replacing inf
                df_normalized[col].replace([np.inf, -np.inf], np.nan, inplace=True)
                df_normalized[col].fillna(mean_val, inplace=True)

            if method == 'minmax':
                min_val = df_normalized[col].min()
                max_val = df_normalized[col].max()
                if max_val > min_val:
                    df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
                else: # If min and max are same, all values in this col are same. Normalize to 0 or 0.5.
                    df_normalized[col] = 0.5 
                    logger.warning(f"Column '{col}' has min == max. Normalized to 0.5.")
                # Clip to [0,1] to ensure no values are outside this range due to potential floating point issues
                df_normalized[col] = df_normalized[col].clip(0, 1)

            elif method == 'zscore':
                mean_val = df_normalized[col].mean()
                std_val = df_normalized[col].std()
                if std_val > 0:
                    df_normalized[col] = (df_normalized[col] - mean_val) / std_val
                else: # If std is 0, all values are same. Normalize to 0.
                    df_normalized[col] = 0.0
                    logger.warning(f"Column '{col}' has std == 0 for Z-score. Normalized to 0.0.")
            else:
                logger.error(f"Unknown normalization method: {method}")
                raise ValueError(f"Unknown normalization method: {method}")
        
        logger.info(f"Normalization ({method}) completed. DataFrame shape: {df_normalized.shape}")
        self.df_normalized = df_normalized.copy()
        return self.df_normalized

    def select_features(self, df: pd.DataFrame, features_to_select: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Selects specified features from the DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            features_to_select (list, optional): List of feature names to select. 
                                               If None, uses features from DATA_CONFIG.
        
        Returns:
            pd.DataFrame: DataFrame with only selected features.
        """
        if df is None or df.empty:
            logger.warning("Input DataFrame for select_features is None or empty.")
            return pd.DataFrame()

        if features_to_select is None:
            logger.info("No specific features_to_select provided, using DATA_CONFIG.")
            # Ensure 'timestamp' and OHLCV are included if they exist in the df
            default_features = DATA_CONFIG.get('price_features', []) + DATA_CONFIG.get('technical_indicators', [])
            features_to_select = [feat for feat in default_features if feat in df.columns]
            if not features_to_select: # If DATA_CONFIG is empty or misconfigured
                 logger.warning("DATA_CONFIG did not yield any features. Selecting all columns.")
                 features_to_select = df.columns.tolist()

        # Ensure 'timestamp' is always included if it exists in the original DataFrame
        if 'timestamp' in df.columns and 'timestamp' not in features_to_select:
            features_to_select.insert(0, 'timestamp')
        
        # Check for missing features in the DataFrame
        actual_features = [feat for feat in features_to_select if feat in df.columns]
        missing_in_df = [feat for feat in features_to_select if feat not in df.columns]
        if missing_in_df:
            logger.warning(f"The following requested features are not in the DataFrame and will be ignored: {missing_in_df}")

        if not actual_features:
            logger.error("No features selected. Returning original DataFrame.")
            self.df_selected = df.copy()
            return self.df_selected
            
        selected_df = df[actual_features].copy()
        logger.info(f"Selected {len(actual_features)} features. DataFrame shape: {selected_df.shape}")
        self.df_selected = selected_df.copy()
        return self.df_selected

    def save_processed_data(self, df: pd.DataFrame, file_name_prefix: str = "processed_data", sub_dir: Optional[str] = "features") -> str:
        """
        Saves the processed DataFrame to a CSV file.
        
        Args:
            df (pd.DataFrame): DataFrame to save.
            file_name_prefix (str): Prefix for the output file name.
            sub_dir (str, optional): Sub-directory within self.data_dir to save the file. 
                                     If None, saves directly in self.data_dir.
                                     Defaults to "features" as per original FeatureEngineering.
        
        Returns:
            str: Full path to the saved file.
        """
        if df is None or df.empty:
            logger.warning("DataFrame for save_processed_data is None or empty. Nothing to save.")
            return ""

        timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"{file_name_prefix}_{timestamp_str}.csv"
        
        # Determine save directory
        if sub_dir:
            # Use DATA_CONFIG['data_dir'] as base if it was successfully imported and seems valid, 
            # otherwise default to self.data_dir. This matches original FeatureEngineering behavior for 'features' subdir.
            # The original FE used DATA_CONFIG['data_dir'] directly for the 'features' path.
            # Here, we make it relative to self.data_dir for consistency within DataProcessor.
            save_directory = os.path.join(self.data_dir, sub_dir)
        else:
            save_directory = self.data_dir
            
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
            logger.info(f"Created directory: {save_directory}")
        
        file_path = os.path.join(save_directory, file_name)
        
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Processed data saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving processed data to {file_path}: {e}")
            return ""


    def prepare_data_for_training(self, symbol: str, timeframe: str = '1h',
                                 start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 normalization_method: str = 'minmax',
                                 missing_value_method: str = 'forward',
                                 features_to_use: Optional[List[str]] = None,
                                 save_output: bool = False,
                                 output_filename_prefix: str = "train_ready_data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepares data for model training: loads, cleans, adds indicators, normalizes, and splits.
        
        Args:
            symbol (str): Cryptocurrency symbol.
            timeframe (str): Data timeframe.
            start_date (str, optional): Start date for data loading.
            end_date (str, optional): End date for data loading.
            train_ratio (float): Proportion of data for training.
            val_ratio (float): Proportion of data for validation.
            normalization_method (str): 'minmax' or 'zscore'.
            missing_value_method (str): Method for handling missing values.
            features_to_use (list, optional): Specific features to use. If None, uses DATA_CONFIG.
            save_output (bool): Whether to save the processed DataFrame.
            output_filename_prefix (str): Prefix for the saved file if save_output is True.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and testing DataFrames.
        """
        # This method combines the original Thai docstring with the English one.
        # The English part describes the updated functionality.
        logger.info(f"Starting data preparation for training: {symbol}, {timeframe}")

        # 1. Load data
        raw_df = self.load_data(symbol, timeframe, start_date, end_date)
        if raw_df.empty:
            logger.error("Data loading failed. Cannot prepare data for training.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 2. Handle missing values (initial pass on raw data if necessary)
        # It's often better to handle missing values *after* feature engineering if indicators might introduce NaNs
        # For now, let's assume handle_missing_values is robust enough or called again later.
        # cleaned_df = self.handle_missing_values(raw_df, method=missing_value_method)
        # if cleaned_df.empty:
        #     logger.error("Missing value handling failed. Cannot prepare data for training.")
        #     return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # 3. Add technical indicators
        # The add_technical_indicators method itself now calls handle_missing_values internally before TA calculation
        # and uses ffill/bfill after.
        df_with_indicators = self.add_technical_indicators(raw_df) # Use raw_df as input
        if df_with_indicators.empty:
            logger.error("Adding technical indicators failed. Cannot prepare data for training.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 4. Select features
        # Pass df_with_indicators to select_features
        df_selected_features = self.select_features(df_with_indicators, features_to_select=features_to_use)
        if df_selected_features.empty:
            logger.error("Feature selection failed. Cannot prepare data for training.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 5. Normalize features
        # Exclude 'timestamp' from normalization by default in normalize_features method
        df_normalized = self.normalize_features(df_selected_features, method=normalization_method)
        if df_normalized.empty:
            logger.error("Data normalization failed. Cannot prepare data for training.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # 6. Save processed data if requested
        if save_output:
            self.save_processed_data(df_normalized, file_name_prefix=output_filename_prefix)
            
        # 7. Split data
        total_rows = len(df_normalized)
        if total_rows == 0:
            logger.error("No data available after processing. Cannot split.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        train_size = int(total_rows * train_ratio)
        val_size = int(total_rows * val_ratio)
        
        if train_size + val_size > total_rows:
            logger.warning("Train + Validation size exceeds total data size. Adjusting validation size.")
            val_size = total_rows - train_size
            if val_size < 0: # Should not happen if train_ratio < 1
                 val_size = 0
        
        train_data = df_normalized.iloc[:train_size]
        val_data = df_normalized.iloc[train_size : train_size + val_size]
        test_data = df_normalized.iloc[train_size + val_size :]
        
        logger.info(f"Data split into: Train ({train_data.shape}), Validation ({val_data.shape}), Test ({test_data.shape})")
        
        return train_data, val_data, test_data
    
    def create_features_for_backtesting(self, symbol: str, timeframe: str = '1h', 
                                       start_date: Optional[str] = None, 
                                       end_date: Optional[str] = None,
                                       missing_value_method: str = 'forward',
                                       features_to_use: Optional[List[str]] = None,
                                       normalization_method: str = 'minmax', # Added for consistency
                                       cols_to_keep_raw: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Creates features for backtesting. Normalizes technical indicators but keeps 
        price-related columns (OHLCV) in their original scale by default.
        
        Args:
            symbol (str): Cryptocurrency symbol.
            timeframe (str): Data timeframe.
            start_date (str, optional): Start date for data loading.
            end_date (str, optional): End date for data loading.
            missing_value_method (str): Method for handling missing values.
            features_to_use (list, optional): Specific features to use. If None, uses DATA_CONFIG.
            normalization_method (str): 'minmax' or 'zscore' for normalizable features.
            cols_to_keep_raw (list, optional): Columns to keep in their original scale (e.g., OHLCV for backtesting).
                                               Defaults to ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
            
        Returns:
            pd.DataFrame: DataFrame with features for backtesting.
        """
        logger.info(f"Starting feature creation for backtesting: {symbol}, {timeframe}")

        if cols_to_keep_raw is None:
            cols_to_keep_raw = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

        # 1. Load data
        raw_df = self.load_data(symbol, timeframe, start_date, end_date)
        if raw_df.empty:
            logger.error("Data loading failed. Cannot create features for backtesting.")
            return pd.DataFrame()

        # 2. Add technical indicators
        # add_technical_indicators handles its own NaNs internally now
        df_with_indicators = self.add_technical_indicators(raw_df)
        if df_with_indicators.empty:
            logger.error("Adding technical indicators failed. Cannot create features for backtesting.")
            return pd.DataFrame()
            
        # 3. Select features (if any specific list is provided)
        # This step is important to ensure we only process and carry forward necessary columns.
        # If features_to_use is None, select_features will use DATA_CONFIG or all columns.
        # We need to ensure cols_to_keep_raw are part of the selected features if they are intended to be in the final output.
        
        # First, define the set of features we want to work with *before* normalization.
        # This includes both indicators and the columns we want to keep raw.
        pre_norm_features_list = features_to_use
        if pre_norm_features_list:
            # Ensure cols_to_keep_raw are included in the list if they are requested
            for col_raw in cols_to_keep_raw:
                if col_raw not in pre_norm_features_list and col_raw in df_with_indicators.columns:
                    pre_norm_features_list.append(col_raw)
        
        df_selected = self.select_features(df_with_indicators, features_to_select=pre_norm_features_list)
        if df_selected.empty:
            logger.error("Feature selection failed. Cannot create features for backtesting.")
            return pd.DataFrame()

        # 4. Normalize data, excluding 'cols_to_keep_raw'
        # The df_selected now contains all columns we care about.
        # We will normalize a copy of it, then selectively put back the raw values.
        
        df_normalized_component = self.normalize_features(df_selected, 
                                                          method=normalization_method, 
                                                          columns_to_exclude=cols_to_keep_raw)
        if df_normalized_component.empty:
            logger.error("Normalization step failed for backtesting features.")
            return pd.DataFrame()

        # Create the final features DataFrame
        # Start with the normalized data, then overwrite the columns that should be raw
        # This is safer than trying to pick columns for normalization from df_selected
        final_backtest_df = df_normalized_component.copy()
        
        for col in cols_to_keep_raw:
            if col in df_selected.columns: # Ensure the column exists in the source
                final_backtest_df[col] = df_selected[col]
            else:
                logger.warning(f"Column '{col}' requested in cols_to_keep_raw not found in selected features.")
        
        logger.info(f"Features for backtesting created. DataFrame shape: {final_backtest_df.shape}")
        self.df_selected = final_backtest_df.copy() # Update self.df_selected with the result
        return self.df_selected