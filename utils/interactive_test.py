#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
import requests
from datetime import datetime
import time

# เพิ่ม path ของ root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from config.config import BINANCE_CONFIG, BASE_URL, STREAM_URL, TESTNET
from utils.APIKeyValidator import APIKeyValidator

# ตั้งค่า logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("interactive_test.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("interactive_test")

class InteractiveBinanceClient:
    """
    คลาสสำหรับการทำงานแบบ interactive กับ Binance API
    """
    
    def __init__(self):
        """
        เริ่มต้นคลาสด้วยค่าเริ่มต้นจาก Config
        """
        self.validator = APIKeyValidator(
            api_key=BINANCE_CONFIG['apiKey'],
            api_secret=BINANCE_CONFIG['secret'],
            base_url=BASE_URL
        )
        self.base_url = self.validator.base_url
        self.last_sync_time = 0
        self.sync_interval = 30000  # 30 วินาที
        
        # ซิงค์เวลาครั้งแรก
        if not self._sync_time_if_needed():
            logger.error("ไม่สามารถซิงค์เวลาได้")
        logger.info("กำลังใช้ Binance Testnet")
        
    def _sync_time_if_needed(self):
        """
        ตรวจสอบและซิงค์เวลาถ้าจำเป็น
        
        Returns:
            bool: True หากซิงค์สำเร็จหรือไม่จำเป็นต้องซิงค์, False หากซิงค์ล้มเหลว
        """
        current_time = int(time.time() * 1000)
        if current_time - self.last_sync_time > self.sync_interval:
            if self.validator.sync_time():
                self.last_sync_time = current_time
                return True
            return False
        return True
        
    def get_account_info(self):
        """
        ดึงข้อมูลบัญชี
        
        Returns:
            dict: ข้อมูลบัญชี
        """
        try:
            if not self._sync_time_if_needed():
                logger.error("ไม่สามารถซิงค์เวลาได้")
                return None
                
            data = {
                'timestamp': self.validator.get_timestamp()
            }
            
            signature, query_string = self.validator.create_signature(data)
            if not signature:
                logger.error("ไม่สามารถสร้าง signature ได้")
                return None
                
            data['signature'] = signature
            url = f"{self.base_url}/v3/account"
            headers = {
                'X-MBX-APIKEY': self.validator.api_key
            }
            
            response = requests.get(url, params=data, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"เกิดข้อผิดพลาด: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลบัญชี: {str(e)}")
            return None
            
    def get_portfolio_balance(self):
        """
        คำนวณยอดรวมของพอร์ต
        
        Returns:
            float: ยอดรวมของพอร์ตใน USDT
        """
        try:
            if not self._sync_time_if_needed():
                logger.error("ไม่สามารถซิงค์เวลาได้")
                return 0.0
                
            account_info = self.get_account_info()
            if not account_info:
                return 0.0
                
            total_balance = 0.0
            
            # ดึงราคาปัจจุบันของทุกเหรียญ
            prices_url = f"{self.base_url}/v3/ticker/price"
            prices_response = requests.get(prices_url)
            
            if prices_response.status_code != 200:
                logger.error("ไม่สามารถดึงราคาปัจจุบันได้")
                return 0.0
                
            prices = {item['symbol']: float(item['price']) for item in prices_response.json()}
            
            # คำนวณยอดรวม
            for asset in account_info['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                total = free + locked
                
                if total > 0:
                    symbol = asset['asset']
                    if symbol == 'USDT':
                        total_balance += total
                    else:
                        # ค้นหาราคาใน USDT
                        price_key = f"{symbol}USDT"
                        if price_key in prices:
                            total_balance += total * prices[price_key]
                            
            return total_balance
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณยอดรวม: {str(e)}")
            return 0.0

def main():
    """
    ฟังก์ชันหลักสำหรับการทำงานแบบ interactive
    """
    try:
        # สร้าง client โดยใช้ค่าเริ่มต้นจาก Config
        client = InteractiveBinanceClient()
        
        while True:
            print("\n=== Binance Testnet Interactive Menu ===")
            print("1. แสดงข้อมูลบัญชี")
            print("2. แสดงยอดรวมพอร์ต")
            print("0. ออกจากโปรแกรม")
            
            choice = input("\nกรุณาเลือกเมนู (0-2): ")
            
            if choice == '1':
                account_info = client.get_account_info()
                if account_info:
                    print("\n=== ข้อมูลบัญชี ===")
                    print(f"สถานะ: {account_info.get('accountType', 'N/A')}")
                    print(f"สิทธิ์การเทรด: {account_info.get('canTrade', False)}")
                    print(f"สิทธิ์การถอน: {account_info.get('canWithdraw', False)}")
                    print(f"สิทธิ์การฝาก: {account_info.get('canDeposit', False)}")
                    
                    print("\n=== ยอดคงเหลือ ===")
                    for asset in account_info['balances']:
                        free = float(asset['free'])
                        locked = float(asset['locked'])
                        if free > 0 or locked > 0:
                            print(f"{asset['asset']}:")
                            print(f"  - ใช้ได้: {free}")
                            print(f"  - ถูกล็อค: {locked}")
                
            elif choice == '2':
                total_balance = client.get_portfolio_balance()
                print(f"\nยอดรวมพอร์ต: {total_balance:.2f} USDT")
                
            elif choice == '0':
                print("\nขอบคุณที่ใช้งานโปรแกรม")
                break
                
            else:
                print("\nกรุณาเลือกเมนูที่ถูกต้อง (0-2)")
                
    except KeyboardInterrupt:
        print("\n\nกำลังปิดโปรแกรม...")
        print("ขอบคุณที่ใช้งานโปรแกรม")
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดที่ไม่คาดคิด: {str(e)}")
        print("\nเกิดข้อผิดพลาดที่ไม่คาดคิด กรุณาตรวจสอบ log file")
    finally:
        print("\nปิดโปรแกรมเรียบร้อยแล้ว")

if __name__ == "__main__":
    main() 