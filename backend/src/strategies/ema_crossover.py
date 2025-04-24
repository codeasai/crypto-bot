from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class Strategy(BaseStrategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.short_period = self.config.get('short_period', 12)
        self.long_period = self.config.get('long_period', 26)
    
    def analyze(self, data: dict) -> dict:
        """
        วิเคราะห์ข้อมูลและตัดสินใจซื้อขาย
        """
        df = pd.DataFrame(data['candles'])
        df['short_ema'] = df['close'].ewm(span=self.short_period, adjust=False).mean()
        df['long_ema'] = df['close'].ewm(span=self.long_period, adjust=False).mean()
        
        current_price = df['close'].iloc[-1]
        short_ema = df['short_ema'].iloc[-1]
        long_ema = df['long_ema'].iloc[-1]
        
        if short_ema > long_ema:
            return {
                'action': 'BUY',
                'reason': 'EMA Crossover (Golden Cross)',
                'price': current_price
            }
        elif short_ema < long_ema:
            return {
                'action': 'SELL',
                'reason': 'EMA Crossover (Death Cross)',
                'price': current_price
            }
        else:
            return {
                'action': 'HOLD',
                'reason': 'No clear signal',
                'price': current_price
            }
    
    def get_name(self) -> str:
        return "EMA Crossover"
    
    def get_description(self) -> str:
        return "กลยุทธ์เทรดตามการตัดกันของเส้น EMA"

    def get_parameters(self):
        return [
            {
                "name": "fast_ema",
                "label": "Fast EMA Period",
                "type": "number",
                "min": 5,
                "max": 20
            },
            {
                "name": "slow_ema",
                "label": "Slow EMA Period",
                "type": "number",
                "min": 21,
                "max": 200
            },
            {
                "name": "investment",
                "label": "Investment Amount (USDT)",
                "type": "number",
                "min": 10
            },
            {
                "name": "stop_loss",
                "label": "Stop Loss (%)",
                "type": "number",
                "min": 0.1,
                "max": 10
            },
            {
                "name": "take_profit",
                "label": "Take Profit (%)",
                "type": "number",
                "min": 0.1,
                "max": 20
            }
        ]
