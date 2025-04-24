from typing import Dict, Any, List
from .base_strategy import BaseStrategy

class Strategy(BaseStrategy):
    """
    กลยุทธ์การเทรดตามราคาต้นทุน
    """
    def __init__(self, params: Dict[str, Any] = None):
        # ตั้งค่า default params ที่นี่ ถ้ามี
        default_params = {
            "profit_target_pct": 0.05,
            "buy_dip_pct": 0.10
        }
        merged_params = default_params
        if params:
            merged_params.update(params)
        super().__init__(config=merged_params) # ส่ง params ให้ BaseStrategy

    def get_name(self) -> str:
        return "Cost Basis"

    def get_description(self) -> str:
        return "Buy when price drops below cost basis and sell when profit target is reached"

    def get_parameters(self) -> List[Dict[str, Any]]:
        # กลยุทธ์นี้อาจจะดึง parameter จาก config โดยตรง แต่ควรมี method นี้
        return [
            {"name": "profit_target_pct", "label": "Profit Target %", "type": "number", "min": 1, "max": 50, "default": 5},
            {"name": "buy_dip_pct", "label": "Buy Dip %", "type": "number", "min": 1, "max": 50, "default": 10},
        ]

    def analyze(self, data: Dict) -> Dict:
        current_price = data.get('current_price', 0)
        # ใช้ self.config ที่รับมาจาก __init__
        cost_basis = data.get('cost_basis', {}).get('weighted_average', 0)
        holdings = data.get('current_holdings', 0)

        profit_target = self.config.get('profit_target_pct', 0.05)
        buy_dip = self.config.get('buy_dip_pct', 0.10)

        if cost_basis <= 0:
            return {"action": "ENTER_NEW" if holdings <= 0 else "HOLD", "reason": "No cost basis"}

        profit_pct = (current_price / cost_basis) - 1

        if profit_pct >= profit_target:
            return {
                "action": "SELL_PROFIT",
                "reason": f"Profit target reached: {profit_pct:.2%} >= {profit_target:.2%}"
            }
        elif profit_pct <= -buy_dip:
            return {
                "action": "BUY_MORE",
                "reason": f"Price dipped: {profit_pct:.2%} <= {-buy_dip:.2%}"
            }
        else:
            return {
                "action": "HOLD",
                "reason": f"Within thresholds: {-buy_dip:.2%} < {profit_pct:.2%} < {profit_target:.2%}"
            }