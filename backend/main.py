"""
จุดเริ่มต้นของโปรแกรม Crypto Trading Bot ที่ใช้ Reinforcement Learning
"""

import argparse
import os
import sys
from typing import Optional


def setup_project():
    """
    ตั้งค่าโครงสร้างโปรเจกต์
    """
    directories = [
        'data',
        'models',
        'outputs',
        'backtest_results',
        'live_trading_logs',
        'environment',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("สร้างโครงสร้างโปรเจกต์เรียบร้อยแล้ว")


def main():
    """
    ฟังก์ชันหลักสำหรับการรันโปรแกรม
    """
    parser = argparse.ArgumentParser(description='Crypto Trading Bot ที่ใช้ Reinforcement Learning')
    parser.add_argument('command', choices=['setup', 'train', 'backtest', 'live'], help='คำสั่งที่ต้องการรัน')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='สัญลักษณ์คู่เหรียญ')
    parser.add_argument('--timeframe', type=str, default='1h', help='กรอบเวลา')
    parser.add_argument('--start_date', type=str, help='วันที่เริ่มต้น (เช่น 2020-01-01)')
    parser.add_argument('--end_date', type=str, help='วันที่สิ้นสุด (เช่น 2022-12-31)')
    parser.add_argument('--model_path', type=str, help='เส้นทางไปยังไฟล์โมเดล (สำหรับการทดสอบหรือเทรดจริง)')
    parser.add_argument('--episodes', type=int, default=1000, help='จำนวนรอบการฝึกสอนทั้งหมด')
    
    # พารามิเตอร์สำหรับการเทรดแบบเรียลไทม์
    parser.add_argument('--exchange', type=str, default='binance', help='ID ของ Exchange')
    parser.add_argument('--api_key', type=str, help='API Key')
    parser.add_argument('--api_secret', type=str, help='API Secret')
    parser.add_argument('--duration', type=float, help='ระยะเวลาที่ต้องการให้บอททำงาน (ชั่วโมง)')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_project()
    
    elif args.command == 'train':
        from train import train_dqn_agent
        
        train_dqn_agent(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            episodes=args.episodes
        )
    
    elif args.command == 'backtest':
        if not args.model_path:
            print("กรุณาระบุ --model_path สำหรับการทดสอบย้อนหลัง")
            sys.exit(1)
            
        from backend.backtest import backtest_model
        
        backtest_model(
            model_path=args.model_path,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )
    
    elif args.command == 'live':
        if not args.model_path:
            print("กรุณาระบุ --model_path สำหรับการเทรดแบบเรียลไทม์")
            sys.exit(1)
            
        if not args.api_key or not args.api_secret:
            print("กรุณาระบุ --api_key และ --api_secret สำหรับการเทรดแบบเรียลไทม์")
            sys.exit(1)
            
        from live_trading import LiveTradingBot
        
        bot = LiveTradingBot(
            model_path=args.model_path,
            exchange_id=args.exchange,
            api_key=args.api_key,
            api_secret=args.api_secret,
            symbol=args.symbol.replace('USDT', '/USDT'),  # ปรับให้เข้ากับรูปแบบของ ccxt
            timeframe=args.timeframe
        )
        
        bot.run(duration_hours=args.duration)


if __name__ == '__main__':
    main()
    