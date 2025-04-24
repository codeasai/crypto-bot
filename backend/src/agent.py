class TradingDecisionMaker:
    """
    คลาสสำหรับตัดสินใจซื้อขายตามกลยุทธ์ราคาต้นทุน
    """
    def __init__(self, profit_target_pct: float = 0.05, buy_dip_pct: float = 0.10):
        self.profit_target_pct = profit_target_pct
        self.buy_dip_pct = buy_dip_pct
    
    def decide_action(self, current_price: float, cost_basis: float, holdings: float) -> str:
        """
        ตัดสินใจว่าจะซื้อ ขาย หรือถือตามราคาต้นทุนและราคาปัจจุบัน
        """
        if cost_basis <= 0:
            return "ENTER_NEW" if holdings <= 0 else "HOLD"
        
        profit_pct = (current_price / cost_basis) - 1
        
        if profit_pct >= self.profit_target_pct:
            return "SELL_PROFIT"
        elif profit_pct <= -self.buy_dip_pct:
            return "BUY_MORE"
        else:
            return "HOLD"
    
    def adjust_thresholds_based_on_trend(self, trend: str) -> None:
        """
        ปรับเกณฑ์การตัดสินใจตามแนวโน้มตลาด
        """
        if trend == "STRONG_UP":
            # ในตลาดขาขึ้นแรง เพิ่มเป้าหมายกำไรและลดการซื้อเพิ่ม
            self.profit_target_pct = 0.08
            self.buy_dip_pct = 0.15
        elif trend == "MODERATE_UP":
            self.profit_target_pct = 0.05
            self.buy_dip_pct = 0.12
        elif trend == "NEUTRAL":
            # ในตลาดทรงตัว ใช้ค่าปกติ
            self.profit_target_pct = 0.05
            self.buy_dip_pct = 0.10
        elif trend == "MODERATE_DOWN":
            self.profit_target_pct = 0.03
            self.buy_dip_pct = 0.08
        elif trend == "STRONG_DOWN":
            # ในตลาดขาลงแรง ลดเป้าหมายกำไรและเพิ่มความระมัดระวังในการซื้อเพิ่ม
            self.profit_target_pct = 0.02
            self.buy_dip_pct = 0.20
