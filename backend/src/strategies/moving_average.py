from .base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class Strategy(BaseStrategy):
    def __init__(self, config=None):
        super().__init__(config)
        self.short_period = self.config.get('short_period', 20)
        self.long_period = self.config.get('long_period', 50)
    
    def analyze(self, data: dict) -> dict:
        """
        วิเคราะห์ข้อมูลและตัดสินใจซื้อขาย
        """
        df = pd.DataFrame(data['candles'])
        df['short_ma'] = df['close'].rolling(window=self.short_period).mean()
        df['long_ma'] = df['close'].rolling(window=self.long_period).mean()
        
        current_price = df['close'].iloc[-1]
        short_ma = df['short_ma'].iloc[-1]
        long_ma = df['long_ma'].iloc[-1]
        
        if short_ma > long_ma:
            return {
                'action': 'BUY',
                'reason': 'Moving Average Crossover (Golden Cross)',
                'price': current_price
            }
        elif short_ma < long_ma:
            return {
                'action': 'SELL',
                'reason': 'Moving Average Crossover (Death Cross)',
                'price': current_price
            }
        else:
            return {
                'action': 'HOLD',
                'reason': 'No clear signal',
                'price': current_price
            }
    
    def get_name(self) -> str:
        return "Moving Average"
    
    def get_description(self) -> str:
        return "กลยุทธ์เทรดตามการตัดกันของเส้น Moving Average"

    def get_parameters(self):
        return [
            {
                "name": "fast_ma",
                "label": "Fast MA Period",
                "type": "number",
                "min": 5,
                "max": 20
            },
            {
                "name": "slow_ma",
                "label": "Slow MA Period",
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