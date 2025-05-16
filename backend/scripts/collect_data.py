"""
สคริปต์สำหรับเก็บรวบรวมข้อมูล
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# เพิ่ม path สำหรับ import โมดูลจากโฟลเดอร์อื่น
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_collector import BinanceDataCollector
from data.feature_engineering import FeatureEngineering
from utils.logger import setup_logger

# ตั้งค่า logger
logger = setup_logger('collect_data')


def main(args):
    """
    ฟังก์ชันหลักสำหรับเก็บรวบรวมข้อมูล
    
    Args:
        args: พารามิเตอร์จากคำสั่ง
    """
    # คำนวณวันที่เริ่มต้นจากจำนวนวัน
    start_date = None
    if args.days:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
    elif args.start_date:
        start_date = args.start_date
    
    # สร้าง DataCollector
    collector = BinanceDataCollector(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=args.end_date,
        testnet=args.testnet
    )
    
    # ถ้าระบุให้อัพเดทข้อมูล
    if args.update:
        logger.info("กำลังอัพเดทข้อมูลให้เป็นปัจจุบัน")
        df = collector.update_historical_data()
    else:
        # เก็บข้อมูลใหม่
        logger.info("กำลังเก็บข้อมูลใหม่")
        df = collector.get_historical_klines()
    
    logger.info(f"เก็บข้อมูลเสร็จสิ้น ได้ข้อมูลทั้งหมด {len(df)} แถว")
    
    # ถ้าระบุให้ประมวลผลข้อมูล
    if args.process:
        logger.info("กำลังประมวลผลข้อมูล")
        # สร้าง features
        fe = FeatureEngineering(df=df)
        processed_df = fe.process_data_pipeline(
            normalize_method=args.normalize,
            missing_method=args.missing_method
        )
        
        logger.info(f"ประมวลผลข้อมูลเสร็จสิ้น ได้ข้อมูลทั้งหมด {len(processed_df)} แถว และ {len(processed_df.columns)} คอลัมน์")


if __name__ == "__main__":
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
    
    main(args)
