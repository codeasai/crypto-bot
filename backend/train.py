"""
ฝึกสอนตัวแทน DQN สำหรับการเทรดสินทรัพย์คริปโต
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import signal
import json
from typing import Tuple
import tensorflow as tf
import logging

# ตั้งค่า TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ปิด warning และ info messages
tf.get_logger().setLevel('ERROR')  # แสดงเฉพาะ error messages

# เพิ่ม path ของ root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.data_processor import DataProcessor
from environment.trading_env import CryptoTradingEnv
from agents.dqn_agent import DQNAgent
from utils.logger import setup_logger

# ตั้งค่า logger
logger = setup_logger('train')

# ตัวแปรสำหรับการยกเลิกการฝึกสอน
training_cancelled = False
current_episode = 0
current_run_dir = None

def setup_tensorflow():
    """
    ตั้งค่า TensorFlow และ GPU
    """
    try:
        # บังคับให้ใช้ CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        logger.info("บังคับให้ใช้ CPU")
        
        # ตั้งค่า memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"พบ GPU: {len(gpus)} เครื่อง")
            for gpu in gpus:
                logger.info(f"ชื่อ GPU: {gpu.name}")
            
            # ตั้งค่า mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("เปิดใช้งาน mixed precision")
        else:
            logger.info("ไม่พบ GPU จะใช้ CPU แทน")
            
        # ตั้งค่า XLA
        tf.config.optimizer.set_jit(True)
        logger.info("เปิดใช้งาน XLA JIT compilation")
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการตั้งค่า TensorFlow: {str(e)}")

def signal_handler(signum, frame):
    """
    จัดการสัญญาณการยกเลิกการฝึกสอน
    """
    global training_cancelled
    logger.info("\nกำลังยกเลิกการฝึกสอน...")
    training_cancelled = True

# ลงทะเบียน signal handler
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def save_training_state(agent, run_dir: str, episode: int, history: dict):
    """
    บันทึกสถานะการฝึกสอนปัจจุบัน
    
    Args:
        agent (DQNAgent): ตัวแทนที่กำลังฝึกสอน
        run_dir (str): โฟลเดอร์สำหรับบันทึกผลลัพธ์
        episode (int): รอบการฝึกสอนปัจจุบัน
        history (dict): ประวัติการฝึกสอน
    """
    try:
        # สร้างโฟลเดอร์สำหรับบันทึกสถานะ
        state_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(state_dir, exist_ok=True)
        
        # ตรวจสอบความยาวของ arrays
        for key in ['train_rewards', 'val_rewards', 'train_profits', 'val_profits', 'exploration_rates']:
            if key in history and not isinstance(history[key], list):
                history[key] = list(history[key])
        
        # บันทึกโมเดล
        model_path = os.path.join(state_dir, f'model_episode_{episode}.keras')
        agent.save(model_path)
        
        # บันทึกประวัติการฝึกสอน
        history_path = os.path.join(state_dir, f'history_episode_{episode}.json')
        with open(history_path, 'w') as f:
            json.dump(history, f)
            
        # บันทึกข้อมูลสถานะ
        state_info = {
            'episode': episode,
            'exploration_rate': agent.exploration_rate,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        state_path = os.path.join(state_dir, f'state_episode_{episode}.json')
        with open(state_path, 'w') as f:
            json.dump(state_info, f)
            
        logger.info(f"บันทึกสถานะการฝึกสอนที่รอบ {episode}")
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการบันทึกสถานะ: {str(e)}")
        logger.error(f"ประวัติการฝึกสอน: {history}")
        raise

def load_data_from_csv(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    โหลดข้อมูลจากไฟล์ CSV
    
    Args:
        symbol (str): สัญลักษณ์คู่เหรียญ
        timeframe (str): กรอบเวลา
        start_date (str): วันที่เริ่มต้น
        end_date (str): วันที่สิ้นสุด
        
    Returns:
        pd.DataFrame: ข้อมูลที่โหลดจาก CSV
    """
    # เตรียมชื่อไฟล์
    start_date_str = pd.to_datetime(start_date).strftime('%Y%m%d')
    end_date_str = pd.to_datetime(end_date).strftime('%Y%m%d')
    file_name = f"{symbol}_{timeframe}_{start_date_str}_{end_date_str}.csv"
    
    # เตรียม path ของไฟล์
    data_dir = os.path.join(current_dir, 'data', 'datasets')
    file_path = os.path.join(data_dir, file_name)
    
    # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
    if not os.path.exists(file_path):
        logger.info(f"ไม่พบไฟล์ข้อมูล {file_path}")
        return pd.DataFrame()
    
    # โหลดข้อมูลจาก CSV
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        logger.info(f"โหลดข้อมูลจาก {file_path} สำเร็จ")
        return df
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการโหลดข้อมูล: {str(e)}")
        return pd.DataFrame()

def load_or_collect_data(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    โหลดข้อมูลจาก CSV หรือดึงข้อมูลจาก exchange ถ้าไม่พบข้อมูล
    
    Args:
        symbol (str): สัญลักษณ์คู่เหรียญ
        timeframe (str): กรอบเวลา
        start_date (str): วันที่เริ่มต้น
        end_date (str): วันที่สิ้นสุด
        
    Returns:
        pd.DataFrame: ข้อมูลที่โหลดหรือดึงมา
    """
    # ลองโหลดจาก CSV ก่อน
    df = load_data_from_csv(symbol, timeframe, start_date, end_date)
    if not df.empty:
        return df
    
    # ถ้าไม่พบข้อมูล ให้ดึงจาก exchange
    logger.info(f"ไม่พบไฟล์ข้อมูลสำหรับ {symbol} ที่กรอบเวลา {timeframe} กำลังดึงข้อมูลจาก Binance...")
    try:
        from data.data_collector import BinanceDataCollector
        collector = BinanceDataCollector(
            symbol=symbol,
            interval=timeframe,
            start_date=start_date,
            end_date=end_date,
            testnet=True
        )
        df = collector.get_historical_klines()
        
        if not df.empty:
            logger.info(f"ดึงข้อมูลสำเร็จ ได้ข้อมูล {len(df)} แท่ง")
            return df
        else:
            logger.error("ไม่สามารถดึงข้อมูลจาก exchange ได้")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล: {str(e)}")
        return pd.DataFrame()

def preview_data_loading(symbol: str, timeframe: str, start_date: str, end_date: str) -> Tuple[bool, pd.DataFrame]:
    """
    แสดงตัวอย่างข้อมูลที่จะโหลดและขอการยืนยัน
    
    Args:
        symbol (str): สัญลักษณ์คู่เหรียญ
        timeframe (str): กรอบเวลา
        start_date (str): วันที่เริ่มต้น
        end_date (str): วันที่สิ้นสุด
        
    Returns:
        Tuple[bool, pd.DataFrame]: (True ถ้าผู้ใช้ยืนยัน, ข้อมูลที่โหลด)
    """
    logger.info("\n=== ข้อมูลที่จะโหลด ===")
    logger.info(f"สัญลักษณ์: {symbol}")
    logger.info(f"กรอบเวลา: {timeframe}")
    logger.info(f"วันที่เริ่มต้น: {start_date}")
    logger.info(f"วันที่สิ้นสุด: {end_date}")
    
    # โหลดหรือดึงข้อมูล
    df = load_or_collect_data(symbol, timeframe, start_date, end_date)
    if df is None or df.empty:
        logger.error("ไม่สามารถโหลดหรือดึงข้อมูลได้")
        return False, pd.DataFrame()
        
    # แสดงตัวอย่างข้อมูล
    logger.info("\nตัวอย่างข้อมูล (5 แท่งแรก):")
    logger.info(f"\n{df.head().to_string()}")
    
    while True:
        choice = input("\nต้องการใช้ข้อมูลชุดนี้ในการฝึกสอนหรือไม่? (Y/n): ").lower()
        if choice in ['', 'y', 'yes']:
            return True, df
        elif choice in ['n', 'no']:
            logger.info("ยกเลิกการฝึกสอน")
            return False, pd.DataFrame()
        else:
            print("กรุณากด Enter สำหรับ Yes หรือพิมพ์ 'n' สำหรับ No")

def validate_data_range(df: pd.DataFrame, start_date: str, end_date: str) -> bool:
    """
    ตรวจสอบช่วงเวลาของข้อมูลและขอการยืนยัน
    
    Args:
        df (pd.DataFrame): ข้อมูลที่ต้องการตรวจสอบ
        start_date (str): วันที่เริ่มต้น
        end_date (str): วันที่สิ้นสุด
        
    Returns:
        bool: True ถ้าผู้ใช้ยืนยัน, False ถ้าผู้ใช้ยกเลิก
    """
    if len(df) == 0:
        logger.error("ไม่พบข้อมูล")
        return False
        
    # แปลงวันที่เป็น datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # ตรวจสอบว่า index เป็น DatetimeIndex หรือไม่
    if not isinstance(df.index, pd.DatetimeIndex):
        logger.error("index ไม่ใช่ DatetimeIndex")
        return False
        
    # ตรวจสอบช่วงเวลาของข้อมูล
    data_start = df.index.min()
    data_end = df.index.max()
    
    logger.info(f"ข้อมูลเริ่มต้น: {data_start}")
    logger.info(f"ข้อมูลสิ้นสุด: {data_end}")
    
    # ตรวจสอบว่าข้อมูลอยู่ในช่วงเวลาที่กำหนด
    if data_start >= start and data_end <= end:
        logger.info(f"ช่วงเวลาของข้อมูลที่จะใช้ train: {data_start} ถึง {data_end}")
        logger.info(f"จำนวนแท่งข้อมูล: {len(df)} แท่ง")
        
        # ตรวจสอบความเหมาะสมของข้อมูล
        if len(df) < 1000:
            logger.warning(f"จำนวนข้อมูลอาจไม่เพียงพอสำหรับการฝึกสอน (น้อยกว่า 1000 แท่ง)")
        elif len(df) > 100000:
            logger.warning(f"จำนวนข้อมูลอาจมากเกินไปสำหรับการฝึกสอน (มากกว่า 100000 แท่ง)")
        else:
            logger.info("จำนวนข้อมูลเหมาะสมสำหรับการฝึกสอน")
            
        # ขอการยืนยัน
        while True:
            choice = input("\nต้องการใช้ข้อมูลชุดนี้ในการฝึกสอนหรือไม่? (Y/n): ").lower()
            if choice in ['', 'y', 'yes']:
                return True
            elif choice in ['n', 'no']:
                logger.info("ยกเลิกการฝึกสอน")
                return False
            else:
                print("กรุณากด Enter สำหรับ Yes หรือพิมพ์ 'n' สำหรับ No")
    else:
        logger.error(f"ข้อมูลไม่อยู่ในช่วงเวลาที่กำหนด")
        logger.error(f"ข้อมูลมีช่วงเวลา: {data_start} ถึง {data_end}")
        logger.error(f"ช่วงเวลาที่ต้องการ: {start} ถึง {end}")
        return False

def validate_and_confirm_data(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
    """
    ตรวจสอบและยืนยันข้อมูลก่อนการฝึกสอน
    
    Args:
        df (pd.DataFrame): ข้อมูลที่ต้องการตรวจสอบ
        symbol (str): สัญลักษณ์คู่เหรียญ
        timeframe (str): กรอบเวลา
        
    Returns:
        bool: True ถ้าผู้ใช้ยืนยัน, False ถ้าผู้ใช้ยกเลิก
    """
    if len(df) == 0:
        logger.error("ไม่พบข้อมูล กรุณาตรวจสอบพารามิเตอร์และลองใหม่อีกครั้ง")
        return False
        
    logger.info(f"\nข้อมูลที่ได้สำหรับ {symbol} ({timeframe}):")
    logger.info(f"จำนวนข้อมูลทั้งหมด: {len(df)} แท่ง")
    logger.info(f"ช่วงเวลา: {df.index[0]} ถึง {df.index[-1]}")
    
    while True:
        choice = input("\nต้องการใช้ข้อมูลชุดนี้ในการฝึกสอนหรือไม่? (Y/n): ").lower()
        if choice in ['', 'y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            logger.info("ยกเลิกการฝึกสอน")
            return False
        else:
            print("กรุณากด Enter สำหรับ Yes หรือพิมพ์ 'n' สำหรับ No")

def find_latest_checkpoint(run_dir: str) -> Tuple[int, str, dict]:
    """
    ค้นหา checkpoint ล่าสุด
    
    Args:
        run_dir (str): โฟลเดอร์ที่เก็บ checkpoints
        
    Returns:
        Tuple[int, str, dict]: (episode, model_path, history)
    """
    try:
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            return 0, None, None
            
        # ค้นหาไฟล์ state ล่าสุด
        state_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('state_episode_')]
        if not state_files:
            return 0, None, None
            
        # หา episode ล่าสุด
        latest_state = max(state_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_episode = int(latest_state.split('_')[2].split('.')[0])
        
        # สร้าง path ของโมเดลและประวัติ
        model_path = os.path.join(checkpoint_dir, f'model_episode_{latest_episode}.keras')
        history_path = os.path.join(checkpoint_dir, f'history_episode_{latest_episode}.json')
        
        # โหลดประวัติ
        history = {}
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
                
        logger.info(f"พบ checkpoint ล่าสุดที่รอบ {latest_episode}")
        return latest_episode, model_path, history
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการค้นหา checkpoint: {str(e)}")
        return 0, None, None

def load_checkpoint(agent: DQNAgent, model_path: str, history: dict) -> dict:
    """
    โหลด checkpoint และคืนค่าประวัติการฝึกสอน
    
    Args:
        agent (DQNAgent): ตัวแทนที่จะโหลด checkpoint
        model_path (str): path ของโมเดล
        history (dict): ประวัติการฝึกสอน
        
    Returns:
        dict: ประวัติการฝึกสอนที่โหลดมา
    """
    try:
        if os.path.exists(model_path):
            agent.load(model_path)
            logger.info(f"โหลดโมเดลจาก {model_path} สำเร็จ")
            return history
        return {}
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการโหลด checkpoint: {str(e)}")
        return {}

def train_dqn_agent(
    symbol: str,
    timeframe: str,
    start_date: str = None,
    end_date: str = None,
    initial_balance: float = 10000.0,
    window_size: int = 10,
    batch_size: int = 64,
    episodes: int = 1000,
    output_dir: str = 'outputs'
):
    """
    ฝึกสอนตัวแทน DQN สำหรับการเทรดสินทรัพย์คริปโต
    """
    global training_cancelled, current_episode, current_run_dir
    
    try:
        # ตั้งค่า TensorFlow
        setup_tensorflow()
        
        # 1. ตรวจสอบและเตรียมวันที่
        current_date = datetime.now()
        if start_date:
            start = pd.to_datetime(start_date)
            if start > current_date:
                logger.error(f"วันที่เริ่มต้น {start_date} อยู่ในอนาคต กรุณาระบุวันที่ในอดีต")
                return None
        else:
            # ใช้ข้อมูลย้อนหลัง 1 ปี
            start_date = (current_date - timedelta(days=365)).strftime('%Y-%m-%d')
            
        if end_date:
            end = pd.to_datetime(end_date)
            if end > current_date:
                logger.warning(f"วันที่สิ้นสุด {end_date} อยู่ในอนาคต จะใช้วันที่ปัจจุบันแทน")
                end_date = current_date.strftime('%Y-%m-%d')
        else:
            end_date = current_date.strftime('%Y-%m-%d')
            
        # ตรวจสอบช่วงเวลา
        if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
            logger.error("วันที่เริ่มต้นต้องน้อยกว่าวันที่สิ้นสุด")
            return None
            
        # 2. แสดงตัวอย่างข้อมูลและขอการยืนยัน
        confirmed, raw_data = preview_data_loading(symbol, timeframe, start_date, end_date)
        if not confirmed or raw_data.empty:
            sys.exit(0)
            
        # 3. เตรียมโฟลเดอร์สำหรับผลลัพธ์
        output_dir = os.path.abspath(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{symbol}_{timeframe}_{timestamp}"
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir)
        
        # 4. ตรวจสอบและแปลงข้อมูล
        if not isinstance(raw_data.index, pd.DatetimeIndex):
            raw_data.index = pd.to_datetime(raw_data.index)
            
        # 4.1 ตรวจสอบช่วงเวลาของข้อมูลและขอการยืนยัน
        if not validate_data_range(raw_data, start_date, end_date):
            return None
            
        # 5. เตรียมข้อมูลสำหรับการฝึกสอน
        logger.info("กำลังเตรียมข้อมูลสำหรับการฝึกสอน...")
        data_processor = DataProcessor(data_dir=os.path.join(current_dir, 'data', 'datasets'))

        # Debug: ตรวจสอบ dtype ของทุก column ใน raw_data
        logger.info(f"ข้อมูลก่อนเพิ่ม technical indicators:\n{raw_data.dtypes}")

        # บังคับแปลงคอลัมน์ตัวเลขให้เป็น float
        for col in raw_data.columns:
            if col not in ['timestamp', 'date']:
                raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce')

        # เพิ่ม technical indicators และ normalize
        processed_data = data_processor.add_technical_indicators(raw_data)
        processed_data = data_processor.normalize_data(processed_data)

        # Debug: ตรวจสอบ dtype ของทุก column ใน processed_data
        logger.info(f"ข้อมูลหลัง normalize:\n{processed_data.dtypes}")

        # บังคับแปลงคอลัมน์ตัวเลขให้เป็น float
        for col in processed_data.columns:
            if col not in ['timestamp', 'date']:
                processed_data[col] = pd.to_numeric(processed_data[col], errors='coerce')

        # ตรวจสอบความยาวของข้อมูล
        if len(processed_data) < window_size:
            logger.error(f"ข้อมูลไม่เพียงพอ ต้องการอย่างน้อย {window_size} แท่ง แต่มีเพียง {len(processed_data)} แท่ง")
            return None

        # แบ่งข้อมูลเป็น train และ validation
        train_size = int(len(processed_data) * 0.8)
        train_data = processed_data.iloc[:train_size]
        val_data = processed_data.iloc[train_size:]
        
        # ตรวจสอบความยาวของข้อมูล train และ validation
        if len(train_data) < window_size or len(val_data) < window_size:
            logger.error(f"ข้อมูล train หรือ validation ไม่เพียงพอ ต้องการอย่างน้อย {window_size} แท่ง")
            logger.error(f"train: {len(train_data)} แท่ง, validation: {len(val_data)} แท่ง")
            return None
        
        # 6. ตรวจสอบและยืนยันข้อมูล
        if not validate_and_confirm_data(train_data, symbol, timeframe):
            sys.exit(0)
            
        logger.info(f"ข้อมูลฝึกสอน: {len(train_data)} แถว, ข้อมูลตรวจสอบ: {len(val_data)} แถว")
        
        # 7. สร้างสภาพแวดล้อมการเทรด
        env = CryptoTradingEnv(
            df=train_data,
            window_size=window_size,
            initial_balance=initial_balance,
            commission_fee=0.001,
            use_risk_adjusted_rewards=True
        )
        
        val_env = CryptoTradingEnv(
            df=val_data,
            window_size=window_size,
            initial_balance=initial_balance,
            commission_fee=0.001,
            use_risk_adjusted_rewards=True
        )
        
        # 8. สร้างตัวแทน DQN
        state_size = env.observation_space.shape[0]
        discrete_action_size = 7  # 7 ประเภทการกระทำ
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=discrete_action_size,
            learning_rate=0.001,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.995,
            exploration_min=0.01,
            batch_size=batch_size,
            memory_size=10000
        )
        
        # 9. ฝึกสอนตัวแทน
        logger.info(f"เริ่มการฝึกสอนตัวแทน DQN สำหรับ {episodes} รอบ...")
        
        # ตั้งค่าตัวแปรสำหรับการยกเลิก
        training_cancelled = False
        current_episode = 0
        current_run_dir = run_dir
        
        # ตัวแปรสำหรับการติดตาม
        train_rewards = []
        val_rewards = []
        train_profits = []
        val_profits = []
        best_val_profit = -np.inf
        exploration_rates = []
        
        # ค้นหาและโหลด checkpoint ล่าสุด
        latest_episode, model_path, history = find_latest_checkpoint(run_dir)
        if latest_episode > 0 and model_path:
            try:
                # โหลด checkpoint
                loaded_history = load_checkpoint(agent, model_path, history)
                if loaded_history:
                    # อัพเดทประวัติการฝึกสอน
                    train_rewards = loaded_history.get('train_rewards', [])
                    val_rewards = loaded_history.get('val_rewards', [])
                    train_profits = loaded_history.get('train_profits', [])
                    val_profits = loaded_history.get('val_profits', [])
                    exploration_rates = loaded_history.get('exploration_rates', [])
                    
                    # ตรวจสอบความยาวของ arrays
                    min_length = min(len(train_rewards), len(val_rewards), 
                                   len(train_profits), len(val_profits), 
                                   len(exploration_rates))
                    if min_length > 0:
                        train_rewards = train_rewards[:min_length]
                        val_rewards = val_rewards[:min_length]
                        train_profits = train_profits[:min_length]
                        val_profits = val_profits[:min_length]
                        exploration_rates = exploration_rates[:min_length]
                    
                    # คำนวณ best_val_profit
                    if val_profits:
                        best_val_profit = max(val_profits)
                    
                    # เริ่มฝึกสอนต่อจาก checkpoint
                    current_episode = latest_episode
                    logger.info(f"พบ checkpoint ล่าสุดที่รอบ {latest_episode} กำลังฝึกสอนต่อ...")
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการโหลด checkpoint: {str(e)}")
                logger.info("เริ่มฝึกสอนใหม่ตั้งแต่รอบแรก")
                current_episode = 0
        
        for episode in range(current_episode, episodes):
            try:
                current_episode = episode
                
                # ตรวจสอบการยกเลิก
                if training_cancelled:
                    logger.info("กำลังบันทึกสถานะก่อนยกเลิก...")
                    save_training_state(agent, run_dir, episode, {
                        'train_rewards': train_rewards,
                        'val_rewards': val_rewards,
                        'train_profits': train_profits,
                        'val_profits': val_profits,
                        'exploration_rates': exploration_rates
                    })
                    logger.info("ยกเลิกการฝึกสอนเรียบร้อย")
                    return None
                
                # 9.1 ฝึกสอน
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                done = False
                total_reward = 0
                
                while not done:
                    action_idx = agent.act(state[0])
                    action = convert_discrete_to_continuous_action(action_idx)
                    
                    next_state, reward, done, info = env.step(action)
                    next_state = np.reshape(next_state, [1, state_size])
                    
                    agent.remember(state[0], action_idx, reward, next_state[0], done)
                    state = next_state
                    total_reward += reward
                
                loss = agent.replay()
                
                # 9.2 บันทึกผลลัพธ์การฝึกสอน
                train_rewards.append(total_reward)
                train_profits.append(info['total_profit'])
                exploration_rates.append(agent.exploration_rate)
                
                # 9.3 ตรวจสอบผลลัพธ์
                if episode % 10 == 0:
                    val_result = validate_episode(val_env, agent, state_size)
                    val_rewards.append(val_result['reward'])
                    val_profits.append(val_result['profit'])
                    
                    if val_result['profit'] > best_val_profit:
                        best_val_profit = val_result['profit']
                        agent.save(os.path.join(run_dir, 'best_model.h5'))
                        save_validation_plot(val_env, run_dir)
                    
                    log_progress(episode, episodes, info, val_result, best_val_profit, agent.exploration_rate)
                    
                    # บันทึกสถานะทุกๆ 10 รอบ
                    if episode % 10 == 0:
                        save_training_state(agent, run_dir, episode, {
                            'train_rewards': train_rewards,
                            'val_rewards': val_rewards,
                            'train_profits': train_profits,
                            'val_profits': val_profits,
                            'exploration_rates': exploration_rates
                        })
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในรอบ {episode}: {str(e)}")
                # บันทึกสถานะเมื่อเกิดข้อผิดพลาด
                save_training_state(agent, run_dir, episode, {
                    'train_rewards': train_rewards,
                    'val_rewards': val_rewards,
                    'train_profits': train_profits,
                    'val_profits': val_profits,
                    'exploration_rates': exploration_rates
                })
                # ดำเนินการต่อในรอบถัดไป
                continue
        
        # 10. บันทึกผลลัพธ์สุดท้าย
        save_training_results(agent, run_dir, history={
            'train_rewards': train_rewards,
            'val_rewards': val_rewards,
            'train_profits': train_profits,
            'val_profits': val_profits,
            'exploration_rates': exploration_rates
        })
        
        logger.info(f"การฝึกสอนเสร็จสิ้น กำไรสูงสุดในการตรวจสอบ: {best_val_profit:.4f}")
        logger.info(f"ผลลัพธ์ถูกบันทึกไว้ที่: {run_dir}")
        
        return {
            'best_model_path': os.path.join(run_dir, 'best_model.h5'),
            'final_model_path': os.path.join(run_dir, 'final_model.h5'),
            'history': {
                'train_rewards': train_rewards,
                'val_rewards': val_rewards,
                'train_profits': train_profits,
                'val_profits': val_profits,
                'exploration_rates': exploration_rates
            },
            'best_validation_profit': best_val_profit
        }
        
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการฝึกสอน: {str(e)}")
        # บันทึกสถานะเมื่อเกิดข้อผิดพลาด
        if current_episode > 0:
            save_training_state(agent, current_run_dir, current_episode, {
                'train_rewards': train_rewards,
                'val_rewards': val_rewards,
                'train_profits': train_profits,
                'val_profits': val_profits,
                'exploration_rates': exploration_rates
            })
        return None

def convert_discrete_to_continuous_action(action_idx: int) -> np.ndarray:
    """แปลง discrete action เป็น continuous action"""
    if action_idx == 0:  # ขายหนัก
        return np.array([-1.0, 1.0])
    elif action_idx == 1:  # ขายปานกลาง
        return np.array([-0.66, 0.66])
    elif action_idx == 2:  # ขายเบา
        return np.array([-0.33, 0.33])
    elif action_idx == 3:  # ถือครอง
        return np.array([0.0, 0.0])
    elif action_idx == 4:  # ซื้อเบา
        return np.array([0.33, 0.33])
    elif action_idx == 5:  # ซื้อปานกลาง
        return np.array([0.66, 0.66])
    elif action_idx == 6:  # ซื้อหนัก
        return np.array([1.0, 1.0])
    else:
        raise ValueError(f"ไม่รองรับ action index: {action_idx}")

def validate_episode(env, agent, state_size: int) -> dict:
    """ตรวจสอบผลลัพธ์ในรอบการตรวจสอบ"""
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    total_reward = 0
    
    while not done:
        action_idx = agent.act(state[0], training=False)
        action = convert_discrete_to_continuous_action(action_idx)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        total_reward += reward
    
    return {
        'reward': total_reward,
        'profit': info['total_profit']
    }

def save_validation_plot(env, run_dir: str):
    """บันทึกกราฟการเทรดที่ดีที่สุด"""
    plt.figure(figsize=(15, 10))
    env.render(mode='human')
    plt.savefig(os.path.join(run_dir, 'best_validation_trades.png'))
    plt.close()

def log_progress(episode: int, total_episodes: int, train_info: dict, 
                val_result: dict, best_val_profit: float, exploration_rate: float):
    """แสดงความคืบหน้าของการฝึกสอน"""
    logger.info(f"Episode: {episode+1}/{total_episodes}, "
               f"Train Profit: {train_info['total_profit']:.4f}, "
               f"Val Profit: {val_result['profit']:.4f}, "
               f"Best Val Profit: {best_val_profit:.4f}, "
               f"Epsilon: {exploration_rate:.4f}")

def save_training_results(agent, run_dir: str, history: dict):
    """บันทึกผลลัพธ์การฝึกสอน"""
    # บันทึกโมเดลสุดท้าย
    agent.save(os.path.join(run_dir, 'final_model.h5'))
    
    # บันทึกประวัติการฝึกสอน
    pd.DataFrame(history).to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)
    
    # สร้างกราฟแสดงประวัติการฝึกสอน
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_rewards'])
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_profits'])
    plt.title('Training Profits')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    
    plt.subplot(2, 2, 3)
    val_episodes = list(range(0, len(history['train_rewards']), 10))
    plt.plot(val_episodes, history['val_rewards'])
    plt.title('Validation Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(2, 2, 4)
    plt.plot(val_episodes, history['val_profits'])
    plt.title('Validation Profits')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_history.png'))
    plt.close()
    
    # สร้างกราฟแสดงอัตราการสำรวจ
    plt.figure(figsize=(10, 5))
    plt.plot(history['exploration_rates'])
    plt.title('Exploration Rate (Epsilon)')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.savefig(os.path.join(run_dir, 'exploration_rate.png'))
    plt.close()

def parse_args(args=None):
    """
    แปลง command line arguments
    
    Args:
        args (list): รายการ arguments (ถ้าเป็น None จะใช้ sys.argv)
        
    Returns:
        argparse.Namespace: arguments ที่แปลงแล้ว
    """
    parser = argparse.ArgumentParser(description='ฝึกสอน Crypto Trading Bot ด้วย DQN')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='สัญลักษณ์คู่เหรียญ')
    parser.add_argument('--timeframe', type=str, default='1h', help='กรอบเวลา')
    parser.add_argument('--start_date', type=str, help='วันที่เริ่มต้น (เช่น 2020-01-01)')
    parser.add_argument('--end_date', type=str, help='วันที่สิ้นสุด (เช่น 2022-12-31)')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='เงินทุนเริ่มต้น')
    parser.add_argument('--window_size', type=int, default=10, help='ขนาดหน้าต่างข้อมูลย้อนหลัง')
    parser.add_argument('--batch_size', type=int, default=64, help='ขนาด batch สำหรับการฝึกสอน')
    parser.add_argument('--episodes', type=int, default=1000, help='จำนวนรอบการฝึกสอนทั้งหมด')
    parser.add_argument('--output_dir', type=str, default='outputs', help='โฟลเดอร์สำหรับบันทึกผลลัพธ์')
    
    if args is None:
        return parser.parse_args()
    return parser.parse_args(args)

def main(args=None):
    """
    ฟังก์ชันหลักสำหรับการรันสคริปต์
    
    Args:
        args (list): รายการ arguments (ถ้าเป็น None จะใช้ sys.argv)
    """
    parsed_args = parse_args(args)
    
    train_dqn_agent(
        symbol=parsed_args.symbol,
        timeframe=parsed_args.timeframe,
        start_date=parsed_args.start_date,
        end_date=parsed_args.end_date,
        initial_balance=parsed_args.initial_balance,
        window_size=parsed_args.window_size,
        batch_size=parsed_args.batch_size,
        episodes=parsed_args.episodes,
        output_dir=parsed_args.output_dir
    )

if __name__ == '__main__':
    main()