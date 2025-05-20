"""
สคริปต์สำหรับเก็บรวบรวมข้อมูล
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# เพิ่ม path สำหรับ import โมดูลจากโฟลเดอร์อื่น
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collector import BinanceDataCollector
from utils.logger import setup_logger

# ตั้งค่า logger
logger = setup_logger('collect_data')

def main():
    """
    ฟังก์ชันหลักสำหรับเก็บรวบรวมข้อมูล
    """
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
    
    # เรียกใช้ฟังก์ชันหลักจาก BinanceDataCollector
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

def log_progress(episode: int, total_episodes: int, train_info: dict, 
                val_result: dict, best_val_profit: float, exploration_rate: float):
    """Show training progress"""
    logger.info(f"Episode: {episode+1}/{total_episodes}, "
               f"Train Profit: {train_info['total_profit']:.4f}, "
               f"Val Profit: {val_result['profit']:.4f}, "
               f"Best Val Profit: {best_val_profit:.4f}, "
               f"Epsilon: {exploration_rate:.4f}")

if __name__ == "__main__":
    main()
