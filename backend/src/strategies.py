from typing import Dict, List, Optional
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class TradingStrategy(ABC):
    """
    คลาสพื้นฐานสำหรับกลยุทธ์การเทรด
    """
    def __init__(self, params: Dict = None):
        self.params = params or {}
    
    @abstractmethod
    def analyze(self, data: Dict) -> Dict:
        """
        วิเคราะห์ข้อมูลและตัดสินใจซื้อขาย
        
        Args:
            data: ข้อมูลสำหรับการวิเคราะห์
            
        Returns:
            Dict: ผลการวิเคราะห์
        """
        pass

class CostBasisStrategy(TradingStrategy):
    """
    กลยุทธ์ราคาต้นทุน
    """
    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.profit_target_pct = self.params.get('profit_target_percentage', 0.05)
        self.buy_dip_pct = self.params.get('buy_dip_percentage', 0.10)
    
    def analyze(self, data: Dict) -> Dict:
        current_price = data['current_price']
        cost_basis = data['cost_basis']['weighted_average']
        holdings = data['current_holdings']
        
        if cost_basis <= 0:
            return {
                'action': "ENTER_NEW" if holdings <= 0 else "HOLD",
                'reason': "No cost basis available"
            }
        
        profit_pct = (current_price / cost_basis) - 1
        
        if profit_pct >= self.profit_target_pct:
            return {
                'action': "SELL_PROFIT",
                'reason': f"Profit target reached: {profit_pct:.2%}"
            }
        elif profit_pct <= -self.buy_dip_pct:
            return {
                'action': "BUY_MORE",
                'reason': f"Price dropped: {profit_pct:.2%}"
            }
        else:
            return {
                'action': "HOLD",
                'reason': f"Current profit: {profit_pct:.2%}"
            }

class MovingAverageStrategy(TradingStrategy):
    """
    กลยุทธ์ Moving Average
    """
    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.short_period = self.params.get('short_period', 20)
        self.long_period = self.params.get('long_period', 50)
    
    def analyze(self, data: Dict) -> Dict:
        # ตัวอย่างการวิเคราะห์ด้วย MA
        # ในความเป็นจริงควรคำนวณ MA จากข้อมูลราคาย้อนหลัง
        current_price = data['current_price']
        holdings = data['current_holdings']
        
        # จำลองการคำนวณ MA
        short_ma = current_price * 1.02  # จำลอง MA สั้น
        long_ma = current_price * 0.98   # จำลอง MA ยาว
        
        if short_ma > long_ma:
            return {
                'action': "BUY_MORE" if holdings <= 0 else "HOLD",
                'reason': "Short MA above Long MA"
            }
        else:
            return {
                'action': "SELL_PROFIT" if holdings > 0 else "HOLD",
                'reason': "Short MA below Long MA"
            }

class RSIStrategy(TradingStrategy):
    """
    กลยุทธ์ RSI
    """
    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.overbought = self.params.get('overbought', 70)
        self.oversold = self.params.get('oversold', 30)
    
    def analyze(self, data: Dict) -> Dict:
        # ตัวอย่างการวิเคราะห์ด้วย RSI
        # ในความเป็นจริงควรคำนวณ RSI จากข้อมูลราคาย้อนหลัง
        current_price = data['current_price']
        holdings = data['current_holdings']
        
        # จำลองค่า RSI
        rsi = 45  # จำลองค่า RSI
        
        if rsi <= self.oversold:
            return {
                'action': "BUY_MORE" if holdings <= 0 else "HOLD",
                'reason': f"RSI oversold: {rsi}"
            }
        elif rsi >= self.overbought:
            return {
                'action': "SELL_PROFIT" if holdings > 0 else "HOLD",
                'reason': f"RSI overbought: {rsi}"
            }
        else:
            return {
                'action': "HOLD",
                'reason': f"RSI neutral: {rsi}"
            }

# Dictionary สำหรับเก็บกลยุทธ์ที่มี
STRATEGIES = {
    'cost_basis': CostBasisStrategy,
    'moving_average': MovingAverageStrategy,
    'rsi': RSIStrategy
}

def get_strategy(strategy_name: str, params: Dict = None) -> TradingStrategy:
    """
    สร้าง instance ของกลยุทธ์ที่ระบุ
    
    Args:
        strategy_name: ชื่อกลยุทธ์
        params: พารามิเตอร์สำหรับกลยุทธ์
        
    Returns:
        TradingStrategy: instance ของกลยุทธ์
        
    Raises:
        ValueError: ถ้าไม่พบกลยุทธ์ที่ระบุ
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(f"ไม่พบกลยุทธ์ {strategy_name}")
    
    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(params)

def list_available_strategies() -> List[str]:
    """
    ดึงรายการกลยุทธ์ที่มีทั้งหมด
    
    Returns:
        List[str]: รายการชื่อกลยุทธ์
    """
    return list(STRATEGIES.keys()) 