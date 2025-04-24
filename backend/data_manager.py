import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataManager:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        
        # สร้างโฟลเดอร์ถ้ายังไม่มี
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
    def _get_user_dir(self, user_id):
        """สร้างและส่งคืน path ของโฟลเดอร์ user"""
        user_dir = os.path.join(self.data_dir, str(user_id))
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        return user_dir
        
    def _get_user_files(self, user_id):
        """ส่งคืน path ของไฟล์ข้อมูลของ user"""
        user_dir = self._get_user_dir(user_id)
        return {
            'profile': os.path.join(user_dir, 'profile.json'),
            'order_history': os.path.join(user_dir, 'order_history.json'),
            'portfolio': os.path.join(user_dir, 'portfolio.json')
        }
        
    def _init_user_files(self, user_id):
        """สร้างไฟล์ข้อมูลเริ่มต้นสำหรับ user"""
        files = self._get_user_files(user_id)
        
        # สร้าง profile.json ถ้ายังไม่มี
        if not os.path.exists(files['profile']):
            self.save_profile(user_id, {
                'user_id': user_id,
                'created_at': datetime.now().isoformat(),
                'last_login': datetime.now().isoformat(),
                'settings': {}
            })
            
        # สร้าง order_history.json ถ้ายังไม่มี
        if not os.path.exists(files['order_history']):
            self.save_order_history(user_id, [])
            
        # สร้าง portfolio.json ถ้ายังไม่มี
        if not os.path.exists(files['portfolio']):
            self.save_portfolio(user_id, {
                'last_update': datetime.now().isoformat(),
                'total_value': 0,
                'assets': []
            })
            
    def save_profile(self, user_id, profile_data):
        """บันทึกข้อมูล profile"""
        try:
            files = self._get_user_files(user_id)
            with open(files['profile'], 'w') as f:
                json.dump(profile_data, f, indent=2)
            logger.info(f"บันทึกข้อมูล profile ของ user {user_id} สำเร็จ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกข้อมูล profile ของ user {user_id}: {str(e)}")
            
    def get_profile(self, user_id):
        """ดึงข้อมูล profile"""
        try:
            files = self._get_user_files(user_id)
            if not os.path.exists(files['profile']):
                self._init_user_files(user_id)
            with open(files['profile'], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูล profile ของ user {user_id}: {str(e)}")
            return None
            
    def save_portfolio(self, user_id, portfolio_data):
        """บันทึกข้อมูลพอร์ตโฟลิโอ"""
        try:
            files = self._get_user_files(user_id)
            with open(files['portfolio'], 'w') as f:
                json.dump(portfolio_data, f, indent=2)
            logger.info(f"บันทึกข้อมูลพอร์ตโฟลิโอของ user {user_id} สำเร็จ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกข้อมูลพอร์ตโฟลิโอของ user {user_id}: {str(e)}")
            
    def get_portfolio(self, user_id):
        """ดึงข้อมูลพอร์ตโฟลิโอ"""
        try:
            files = self._get_user_files(user_id)
            if not os.path.exists(files['portfolio']):
                self._init_user_files(user_id)
            with open(files['portfolio'], 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลพอร์ตโฟลิโอของ user {user_id}: {str(e)}")
            return {
                'last_update': datetime.now().isoformat(),
                'total_value': 0,
                'assets': []
            }
            
    def save_order_history(self, user_id, orders):
        """บันทึกประวัติออเดอร์"""
        try:
            files = self._get_user_files(user_id)
            with open(files['order_history'], 'w') as f:
                json.dump({'orders': orders}, f, indent=2)
            logger.info(f"บันทึกประวัติออเดอร์ของ user {user_id} สำเร็จ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกประวัติออเดอร์ของ user {user_id}: {str(e)}")
            
    def get_order_history(self, user_id):
        """ดึงประวัติออเดอร์"""
        try:
            files = self._get_user_files(user_id)
            if not os.path.exists(files['order_history']):
                self._init_user_files(user_id)
            with open(files['order_history'], 'r') as f:
                return json.load(f)['orders']
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงประวัติออเดอร์ของ user {user_id}: {str(e)}")
            return []
            
    def add_order(self, user_id, order):
        """เพิ่มออเดอร์ใหม่"""
        try:
            orders = self.get_order_history(user_id)
            order['timestamp'] = datetime.now().isoformat()
            orders.append(order)
            self.save_order_history(user_id, orders)
            logger.info(f"เพิ่มออเดอร์ใหม่ของ user {user_id} สำเร็จ: {order['id']}")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเพิ่มออเดอร์ของ user {user_id}: {str(e)}")
            
    def update_portfolio(self, user_id, assets, total_value):
        """อัพเดทข้อมูลพอร์ตโฟลิโอ"""
        try:
            portfolio_data = {
                'last_update': datetime.now().isoformat(),
                'total_value': total_value,
                'assets': assets
            }
            self.save_portfolio(user_id, portfolio_data)
            logger.info(f"อัพเดทข้อมูลพอร์ตโฟลิโอของ user {user_id} สำเร็จ")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอัพเดทข้อมูลพอร์ตโฟลิโอของ user {user_id}: {str(e)}")
            
    def get_all_users(self):
        """ดึงรายการ user ทั้งหมด"""
        try:
            return [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดึงรายการ user: {str(e)}")
            return [] 