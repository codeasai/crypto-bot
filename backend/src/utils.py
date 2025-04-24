from typing import Dict, List, Tuple

class CostBasisCalculator:
    """
    คลาสสำหรับคำนวณราคาต้นทุนเฉลี่ยของสินทรัพย์คริปโต
    """
    @staticmethod
    def calculate_weighted_average(trades: List[Dict]) -> float:
        """
        คำนวณราคาต้นทุนเฉลี่ยถ่วงน้ำหนักตามปริมาณ
        """
        if not trades:
            return 0
            
        # กรองเฉพาะคำสั่งซื้อ
        buy_trades = [trade for trade in trades if trade['side'].upper() == 'BUY']
        
        if not buy_trades:
            return 0
            
        total_qty = sum(float(trade['amount']) for trade in buy_trades)
        
        if total_qty == 0:
            return 0
            
        total_cost = sum(float(trade['amount']) * float(trade['price']) for trade in buy_trades)
        return total_cost / total_qty
    
    @staticmethod
    def calculate_with_fees(trades: List[Dict]) -> float:
        """
        คำนวณราคาต้นทุนเฉลี่ยรวมค่าธรรมเนียม
        """
        if not trades:
            return 0
            
        # กรองเฉพาะคำสั่งซื้อ
        buy_trades = [trade for trade in trades if trade['side'].upper() == 'BUY']
        
        if not buy_trades:
            return 0
            
        total_qty = sum(float(trade['amount']) for trade in buy_trades)
        
        if total_qty == 0:
            return 0
            
        total_cost = sum(float(trade['amount']) * float(trade['price']) + (float(trade.get('fee', {}).get('cost', 0)) if trade.get('fee', {}).get('currency') == 'USDT' else 0) for trade in buy_trades)
        return total_cost / total_qty
    
    @staticmethod
    def calculate_fifo_cost_basis(trades: List[Dict]) -> float:
        """
        คำนวณราคาต้นทุนโดยใช้วิธี First-In-First-Out (FIFO)
        """
        if not trades:
            return 0
            
        # เรียงลำดับตามเวลา
        sorted_trades = sorted(trades, key=lambda t: t['datetime'])
        
        buy_queue = []
        remaining_qty = 0
        
        for trade in sorted_trades:
            side = trade['side'].upper()
            qty = float(trade['amount'])
            price = float(trade['price'])
            
            if side == 'BUY':
                buy_queue.append({'amount': qty, 'price': price})
                remaining_qty += qty
            elif side == 'SELL' and remaining_qty > 0:
                sell_qty = qty
                while sell_qty > 0 and buy_queue:
                    oldest_buy = buy_queue[0]
                    if oldest_buy['amount'] <= sell_qty:
                        # ขายหมดการซื้อครั้งแรก
                        sell_qty -= oldest_buy['amount']
                        remaining_qty -= oldest_buy['amount']
                        buy_queue.pop(0)
                    else:
                        # ขายเพียงบางส่วนของการซื้อครั้งแรก
                        oldest_buy['amount'] -= sell_qty
                        remaining_qty -= sell_qty
                        sell_qty = 0
        
        # คำนวณราคาต้นทุนจากการซื้อที่เหลือ
        if remaining_qty == 0:
            return 0
            
        total_cost = sum(trade['amount'] * trade['price'] for trade in buy_queue)
        return total_cost / remaining_qty


class RiskManager:
    """
    คลาสสำหรับจัดการความเสี่ยงในการเทรด
    """
    def __init__(self, max_position_pct: float = 0.20, max_trade_pct: float = 0.05):
        self.max_position_pct = max_position_pct  # สัดส่วนสูงสุดของพอร์ตที่จะลงทุนในสินทรัพย์เดียว
        self.max_trade_pct = max_trade_pct  # สัดส่วนสูงสุดของพอร์ตต่อการเทรดครั้งเดียว
    
    def validate_trade(self, balance: float, position_value: float, trade_amount: float) -> Tuple[bool, float]:
        """
        ตรวจสอบว่าการเทรดอยู่ในขอบเขตความเสี่ยงที่ยอมรับได้หรือไม่
        """
        total_portfolio = balance + position_value
        
        # ตรวจสอบว่าการเทรดนี้จะทำให้ตำแหน่งเกินขีดจำกัดหรือไม่
        new_position_value = position_value + trade_amount
        new_position_pct = new_position_value / total_portfolio if total_portfolio > 0 else 0
        
        if new_position_pct > self.max_position_pct:
            # ปรับขนาดการเทรดให้อยู่ในขีดจำกัด
            max_allowed_position = total_portfolio * self.max_position_pct
            adjusted_trade = max(0, max_allowed_position - position_value)
            return False, adjusted_trade
        
        # ตรวจสอบว่าขนาดการเทรดเกินขีดจำกัดต่อครั้งหรือไม่
        trade_pct = trade_amount / total_portfolio if total_portfolio > 0 else 0
        
        if trade_pct > self.max_trade_pct:
            adjusted_trade = total_portfolio * self.max_trade_pct
            return False, adjusted_trade
        
        return True, trade_amount
