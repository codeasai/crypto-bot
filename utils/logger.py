"""
จัดการการบันทึก log สำหรับระบบ
"""

import logging
import os
import sys
from datetime import datetime

def setup_logger(name: str, log_dir: str = 'logs') -> logging.Logger:
    """
    ตั้งค่า logger
    
    Args:
        name (str): ชื่อของ logger
        log_dir (str): โฟลเดอร์สำหรับเก็บไฟล์ log
        
    Returns:
        logging.Logger: logger ที่ตั้งค่าแล้ว
    """
    # สร้างโฟลเดอร์สำหรับเก็บ log
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # สร้าง logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # ตรวจสอบว่ามี handler อยู่แล้วหรือไม่
    if logger.handlers:
        return logger
        
    # สร้าง formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # สร้าง file handler
    log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # สร้าง console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # เพิ่ม handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    ดึง logger ที่มีอยู่แล้วหรือสร้างใหม่ถ้ายังไม่มี
    
    Args:
        name (str): ชื่อของโมดูล
        
    Returns:
        logging.Logger: logger ที่ตั้งค่าแล้ว
    """
    return setup_logger(name)

if __name__ == "__main__":
    # ทดสอบการใช้งาน logger
    logger = setup_logger('test_logger')
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
