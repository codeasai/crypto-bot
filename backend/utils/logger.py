"""
ระบบบันทึกประวัติการทำงาน (Logger)
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys

# เพิ่ม path ของ backend เข้าไปใน sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
sys.path.insert(0, backend_dir)

from config.config import LOGGER_CONFIG


def setup_logger(name):
    """
    ตั้งค่า logger
    
    Args:
        name (str): ชื่อของ logger
        
    Returns:
        logging.Logger: Logger object
    """
    # สร้าง logger
    logger = logging.getLogger(name)
    
    # ตั้งค่าระดับการ log
    level = getattr(logging, LOGGER_CONFIG['level'])
    logger.setLevel(level)
    
    # ตรวจสอบว่ามี handler อยู่แล้วหรือไม่
    if logger.handlers:
        return logger
    
    # สร้าง formatter
    formatter = logging.Formatter(LOGGER_CONFIG['format'])
    
    # สร้าง console handler พร้อมกำหนด encoding เป็น utf-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.stream.reconfigure(encoding='utf-8')
    logger.addHandler(console_handler)
    
    # สร้าง file handler
    # สร้างโฟลเดอร์สำหรับเก็บ log ถ้ายังไม่มี
    log_dir = LOGGER_CONFIG['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # สร้าง log file ตามชื่อ logger
    log_file = os.path.join(log_dir, f"{name}.log")
    
    # สร้าง rotating file handler (จำกัดขนาดไฟล์และจำนวนไฟล์)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


if __name__ == "__main__":
    # ทดสอบการใช้งาน logger
    logger = setup_logger('test_logger')
    logger.debug('This is a debug message')
    logger.info('This is an info message')
    logger.warning('This is a warning message')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
