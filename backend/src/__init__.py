# ไฟล์นี้ทำให้โฟลเดอร์ src เป็น Python package
# เพื่อให้สามารถ import โมดูลต่างๆ ได้

from src.bot import CryptoTradingBot
from src.agent import TradingDecisionMaker
from src.utils import CostBasisCalculator, RiskManager
from src.bot_manager import BotManager

__all__ = [
    'CryptoTradingBot',
    'TradingDecisionMaker',
    'CostBasisCalculator',
    'RiskManager',
    'BotManager'
]

def create_bot(config_file: str = 'bot_config.json') -> CryptoTradingBot:
    """
    สร้าง instance ของ CryptoTradingBot
    
    Args:
        config_file (str): path ของไฟล์ config
        
    Returns:
        CryptoTradingBot: instance ของ bot
    """
    return CryptoTradingBot(config_file=config_file)

def create_bot_manager(config_dir: str = 'configs') -> BotManager:
    """
    สร้าง instance ของ BotManager
    
    Args:
        config_dir (str): path ของโฟลเดอร์ config
        
    Returns:
        BotManager: instance ของ bot manager
    """
    return BotManager(config_dir=config_dir)
