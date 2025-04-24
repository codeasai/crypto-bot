from typing import Dict, List, Optional
import json
import os
import logging
from src.bot import CryptoTradingBot
import ccxt # เพิ่ม ccxt

logger = logging.getLogger(__name__)

class BotManager:
    """
    คลาสสำหรับจัดการบอทหลายตัว
    """
    def __init__(self, config_dir: str = 'configs'):
        self.config_dir = os.path.abspath(config_dir) # ใช้ absolute path
        self.bots: Dict[str, CryptoTradingBot] = {}
        self._ensure_config_dir()
        self.load_existing_bots() # โหลดบอทที่มีอยู่ตอนเริ่มต้น
        
    def _ensure_config_dir(self) -> None:
        """สร้างโฟลเดอร์ configs ถ้ายังไม่มี"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            logger.info(f"Created config directory: {self.config_dir}")
            
    def load_existing_bots(self) -> None:
        """โหลดบอทจากไฟล์ config ที่มีอยู่"""
        logger.info(f"Loading existing bots from {self.config_dir}...")
        loaded_count = 0
        for filename in os.listdir(self.config_dir):
            if filename.endswith('.json') and "_history" not in filename:
                bot_id = filename[:-5] # ลบ .json
                try:
                    if bot_id not in self.bots:
                        # สร้าง bot instance โดยใช้ config_dir ที่ถูกต้อง
                        bot = CryptoTradingBot(bot_id=bot_id, config_dir=self.config_dir)
                        self.bots[bot_id] = bot
                        logger.info(f"Loaded bot: {bot_id}")
                        loaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to load bot {bot_id} from {filename}: {e}")
        logger.info(f"Finished loading {loaded_count} existing bots.")
        
    def create_bot(self, bot_id: str, config_data: Dict) -> bool:
        """สร้างบอทใหม่"""
        if bot_id in self.bots:
             logger.error(f"Bot creation failed: Bot ID '{bot_id}' already exists.")
             return False
        try:
            config_file_path = os.path.join(self.config_dir, f"{bot_id}.json")
            
            # รวม config ที่รับมากับ default (ถ้าจำเป็น)
            # bot_config = {**default_bot_config, **config_data} # อาจจะต้องมี default config
            bot_config = config_data # สมมติว่า frontend ส่ง config ที่ครบถ้วนมา

            with open(config_file_path, 'w') as f:
                json.dump(bot_config, f, indent=4)

            # สร้าง bot instance และเพิ่มเข้า dictionary
            # ส่ง config_dir ไปให้ CryptoTradingBot รู้ว่า config อยู่ไหน
            bot = CryptoTradingBot(bot_id=bot_id, config_dir=self.config_dir)
            self.bots[bot_id] = bot
            logger.info(f"Bot '{bot_id}' created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating bot '{bot_id}': {str(e)}", exc_info=True)
            # ลบไฟล์ config ที่อาจสร้างไปแล้วถ้าเกิด error
            if os.path.exists(config_file_path):
                 try: os.remove(config_file_path)
                 except: pass
            return False
            
    def delete_bot(self, bot_id: str) -> bool:
        if bot_id not in self.bots:
            logger.warning(f"Delete failed: Bot ID '{bot_id}' not found.")
            return False
        try:
            bot = self.bots[bot_id]
            if bot.is_running:
                logger.info(f"Stopping bot '{bot_id}' before deletion...")
                bot.stop_bot()

            # ลบไฟล์ config
            config_file = bot.config_file # ใช้ path จาก instance
            if os.path.exists(config_file):
                os.remove(config_file)
                logger.info(f"Deleted config file: {config_file}")

            # ลบไฟล์ history (ถ้าต้องการ)
            history_file = os.path.join(self.config_dir, f"{bot_id}_history.json")
            if os.path.exists(history_file):
                os.remove(history_file)
                logger.info(f"Deleted history file: {history_file}")

            del self.bots[bot_id]
            logger.info(f"Bot '{bot_id}' deleted successfully.")
            return True
        except Exception as e:
            logger.error(f"Error deleting bot '{bot_id}': {str(e)}")
            return False
            
    def start_bot(self, bot_id: str) -> bool:
        if bot_id not in self.bots:
            logger.warning(f"Start failed: Bot ID '{bot_id}' not found.")
            return False
        try:
            bot = self.bots[bot_id]
            if not bot.is_running:
                 bot.run_bot()
                 logger.info(f"Bot '{bot_id}' started.")
                 return True
            else:
                 logger.info(f"Bot '{bot_id}' is already running.")
                 return True # ถือว่าสำเร็จถ้ามันรันอยู่แล้ว
        except Exception as e:
            logger.error(f"Error starting bot '{bot_id}': {str(e)}")
            return False
            
    def stop_bot(self, bot_id: str) -> bool:
        if bot_id not in self.bots:
            logger.warning(f"Stop failed: Bot ID '{bot_id}' not found.")
            return False
        try:
            bot = self.bots[bot_id]
            if bot.is_running:
                bot.stop_bot()
                logger.info(f"Bot '{bot_id}' stopped.")
                return True
            else:
                 logger.info(f"Bot '{bot_id}' is already stopped.")
                 return True # ถือว่าสำเร็จถ้ามันหยุดอยู่แล้ว
        except Exception as e:
            logger.error(f"Error stopping bot '{bot_id}': {str(e)}")
            return False
            
    def stop_all_bots(self) -> None:
         """หยุดการทำงานของบอททุกตัว"""
         logger.info("Stopping all running bots...")
         for bot_id in list(self.bots.keys()): # ใช้ list() เพื่อให้ลบ key ได้ขณะวน loop
             self.stop_bot(bot_id)
         logger.info("All bots stopped.")
            
    def get_bot(self, bot_id: str) -> Optional[CryptoTradingBot]:
        """ดึง instance ของ bot ตาม ID"""
        return self.bots.get(bot_id)
            
    def get_bot_status(self, bot_id: str) -> Dict:
        bot = self.get_bot(bot_id)
        if bot:
             # ใช้ get_status() method จาก CryptoTradingBot ถ้ามี
             if hasattr(bot, 'get_status') and callable(getattr(bot, 'get_status')):
                  return bot.get_status()
             else:
                  # Fallback ถ้า bot ไม่มี get_status()
                  return {
                     'id': bot_id,
                     'is_running': bot.is_running,
                     'config': bot.config,
                     # อาจจะเพิ่มข้อมูลอื่น ๆ ที่จำเป็น
                  }
        return {}
            
    def get_all_bots_status(self) -> List[Dict]:
        """ดึงสถานะของบอททั้งหมด"""
        return [self.get_bot_status(bot_id) for bot_id in self.bots.keys()]
            
    def update_bot_config(self, bot_id: str, config_updates: Dict) -> bool:
        if bot_id not in self.bots:
            logger.warning(f"Update config failed: Bot ID '{bot_id}' not found.")
            return False
        try:
            bot = self.bots[bot_id]
            was_running = bot.is_running
            if was_running:
                logger.info(f"Stopping bot '{bot_id}' temporarily to update config...")
                bot.stop_bot()

            # อัพเดท config ใน instance
            # ควรมี validation หรือ merge logic ที่ดีกว่านี้
            bot.config.update(config_updates)
            # บันทึก config ลงไฟล์
            bot.save_config()
            logger.info(f"Config for bot '{bot_id}' updated successfully.")

            # รีโหลด exchange settings หรือ strategy ถ้าจำเป็น
            bot.setup_exchange() # อาจจะต้อง setup ใหม่ถ้า API key เปลี่ยน
            # โหลด Strategy ใหม่
            strategy_name = bot.config.get('strategy')
            strategy_params = bot.config.get('strategy_params', {})
            try:
                from src.strategies import get_strategy # Import get_strategy ที่นี่
                bot.strategy = get_strategy(strategy_name, strategy_params)
                logger.info(f"[{bot.bot_id}] Reloaded strategy '{strategy_name}' after config update.")
            except Exception as e:
                 logger.error(f"[{bot.bot_id}] Error reloading strategy '{strategy_name}' after config update: {e}")
                 bot.strategy = None # หรือใช้ fallback

            if was_running:
                logger.info(f"Restarting bot '{bot_id}' after config update...")
                bot.run_bot()

            return True
        except Exception as e:
            logger.error(f"Error updating config for bot '{bot_id}': {str(e)}", exc_info=True)
            return False

    # ---- Methods for Background Thread ----
    # Methods เหล่านี้อาจจะดึงข้อมูลจาก bot instances โดยตรง

    def get_all_balances(self) -> Dict[str, Dict]:
         """ดึง balance จากทุก bot (อาจจะช้า)"""
         all_balances = {}
         for bot_id, bot in self.bots.items():
             if bot.exchange:
                 try:
                     all_balances[bot_id] = bot.get_balance() # เรียก method ของ bot
                 except Exception as e:
                     logger.error(f"Error getting balance for bot {bot_id}: {e}")
                     all_balances[bot_id] = {"error": str(e)}
         return all_balances

    def get_all_open_orders(self) -> Dict[str, List[Dict]]:
        """ดึง open orders จากทุก bot (อาจจะช้า)"""
        all_orders = {}
        for bot_id, bot in self.bots.items():
            if bot.exchange:
                bot_orders = []
                try:
                    for symbol in bot.config.get('symbols', []):
                        orders = bot.exchange.fetch_open_orders(symbol)
                        bot_orders.extend(orders)
                    all_orders[bot_id] = bot_orders
                except Exception as e:
                    logger.error(f"Error getting open orders for bot {bot_id}: {e}")
                    all_orders[bot_id] = [{"error": str(e)}]
            else:
                all_orders[bot_id] = []
        return all_orders

    # Method get_ticker อาจจะไม่จำเป็นต้องมีใน manager
    # background thread สามารถเรียก get_current_price จาก bot instance ได้โดยตรง
    # def get_ticker(self, symbol: str) -> Optional[Dict]:
    #     # หา bot ตัวแรกที่มี exchange และลอง fetch ticker
    #     for bot in self.bots.values():
    #         if bot.exchange:
    #             try:
    #                 return bot.exchange.fetch_ticker(symbol)
    #             except Exception as e:
    #                 logger.warning(f"Could not fetch ticker {symbol} using bot {bot.bot_id}: {e}")
    #     logger.error(f"Could not fetch ticker {symbol} from any bot.")
    #     return None 