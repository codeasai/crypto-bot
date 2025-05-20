#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import hmac
import hashlib
import time
import requests
import logging
from urllib.parse import urlencode

# เพิ่ม path ของ root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class APIKeyValidator:
    """
    คลาสสำหรับตรวจสอบและจัดการ API key ของ Binance
    """
    
    def __init__(self, api_key, api_secret, base_url):
        """
        เริ่มต้นคลาสด้วย API key และ secret
        
        Args:
            api_key (str): Binance API Key
            api_secret (str): Binance API Secret
            base_url (str): Base URL สำหรับ API
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.time_offset = 0
        
        # ตั้งค่า logging
        self.logger = logging.getLogger("APIKeyValidator")
        self.logger.info("กำลังใช้ Binance Testnet")
        
    def create_signature(self, data):
        """
        สร้าง signature สำหรับการยืนยันตัวตน
        
        Args:
            data (dict): ข้อมูลที่จะส่งไปยัง API
            
        Returns:
            tuple: (signature, query_string)
        """
        try:
            # แปลงข้อมูลเป็น query string
            query_string = urlencode(data)
            
            # สร้าง signature ด้วย HMAC SHA256
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature, query_string
            
        except Exception as e:
            self.logger.error(f"เกิดข้อผิดพลาดในการสร้าง signature: {str(e)}")
            return None, None
            
    def sync_time(self):
        """
        ซิงค์เวลากับ server
        
        Returns:
            bool: True หากสำเร็จ, False หากล้มเหลว
        """
        try:
            # ดึงเวลาของ server
            url = f"{self.base_url}/v3/time"
            response = requests.get(url)
            
            if response.status_code == 200:
                server_time = response.json().get('serverTime', 0)
                local_time = int(time.time() * 1000)
                self.time_offset = server_time - local_time
                self.logger.info(f"ปรับเวลาท้องถิ่นให้ตรงกับ server: {self.time_offset} ms")
                return True
            else:
                self.logger.error(f"ไม่สามารถดึงเวลาของ server ได้: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"เกิดข้อผิดพลาดในการซิงค์เวลา: {str(e)}")
            return False
            
    def get_timestamp(self):
        """
        ดึง timestamp ที่ปรับแล้ว
        
        Returns:
            int: timestamp ที่ปรับแล้ว
        """
        return int(time.time() * 1000) + self.time_offset
        
    def validate_api_key(self):
        """
        ตรวจสอบความถูกต้องของ API key
        
        Returns:
            bool: True หาก API key ถูกต้อง, False หากไม่ถูกต้อง
        """
        try:
            # ซิงค์เวลาก่อน
            if not self.sync_time():
                return False
                
            # ใช้เวลาที่ปรับแล้ว
            data = {
                'timestamp': self.get_timestamp()
            }
            
            # สร้าง signature
            signature, query_string = self.create_signature(data)
            if not signature:
                return False
                
            # เพิ่ม signature ไปใน data
            data['signature'] = signature
            
            # สร้าง URL สำหรับ request
            url = f"{self.base_url}/v3/account"
            
            # สร้าง headers
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            # ส่ง request
            response = requests.get(url, params=data, headers=headers)
            
            if response.status_code == 200:
                self.logger.info("API key ถูกต้อง")
                return True
            else:
                self.logger.error(f"API key ไม่ถูกต้อง: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบ API key: {str(e)}")
            return False
            
    def get_api_info(self):
        """
        ดึงข้อมูล API key
        
        Returns:
            dict: ข้อมูล API key หรือ None หากเกิดข้อผิดพลาด
        """
        try:
            # ซิงค์เวลาก่อน
            if not self.sync_time():
                return None
                
            # ใช้เวลาที่ปรับแล้ว
            data = {
                'timestamp': self.get_timestamp()
            }
            
            # สร้าง signature
            signature, query_string = self.create_signature(data)
            if not signature:
                return None
                
            # เพิ่ม signature ไปใน data
            data['signature'] = signature
            
            # สร้าง URL สำหรับ request
            url = f"{self.base_url}/v3/account"
            
            # สร้าง headers
            headers = {
                'X-MBX-APIKEY': self.api_key
            }
            
            # ส่ง request
            response = requests.get(url, params=data, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"ไม่สามารถดึงข้อมูล API key ได้: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล API key: {str(e)}")
            return None 