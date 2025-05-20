"""
คำนวณเมตริกต่างๆ เช่น Sharpe ratio, Max Drawdown
"""

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, period=252):
    """
    คำนวณ Sharpe Ratio
    
    Args:
        returns (np.array): อาเรย์ของผลตอบแทน
        risk_free_rate (float): อัตราผลตอบแทนปราศจากความเสี่ยง
        period (int): จำนวนช่วงเวลาใน 1 ปี (252 วันทำการ, 52 สัปดาห์, 12 เดือน)
        
    Returns:
        float: Sharpe Ratio
    """
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if len(returns) == 0:
        return 0
    
    # คำนวณ Sharpe Ratio
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    
    # ปรับ Sharpe Ratio ตามช่วงเวลา
    sharpe_annualized = sharpe * np.sqrt(period)
    
    return sharpe_annualized


def calculate_sortino_ratio(returns, risk_free_rate=0.0, period=252):
    """
    คำนวณ Sortino Ratio (Sharpe Ratio ที่พิจารณาเฉพาะความเสี่ยงขาลง)
    
    Args:
        returns (np.array): อาเรย์ของผลตอบแทน
        risk_free_rate (float): อัตราผลตอบแทนปราศจากความเสี่ยง
        period (int): จำนวนช่วงเวลาใน 1 ปี (252 วันทำการ, 52 สัปดาห์, 12 เดือน)
        
    Returns:
        float: Sortino Ratio
    """
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if len(returns) == 0:
        return 0
    
    # แยกเฉพาะผลตอบแทนที่ติดลบ
    negative_returns = returns[returns < 0]
    
    # ถ้าไม่มีผลตอบแทนติดลบ, return Sortino Ratio เป็นอินฟินิตี้
    if len(negative_returns) == 0:
        return float('inf')
    
    # คำนวณ Downside Deviation
    mean_return = np.mean(returns)
    downside_deviation = np.sqrt(np.mean(np.square(negative_returns)))
    
    if downside_deviation == 0:
        return 0
    
    sortino = (mean_return - risk_free_rate) / downside_deviation
    
    # ปรับ Sortino Ratio ตามช่วงเวลา
    sortino_annualized = sortino * np.sqrt(period)
    
    return sortino_annualized


def calculate_max_drawdown(values):
    """
    คำนวณ Maximum Drawdown
    
    Args:
        values (np.array): อาเรย์ของมูลค่าพอร์ตโฟลิโอ
        
    Returns:
        float: Maximum Drawdown (%)
    """
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if len(values) == 0:
        return 0
    
    # คำนวณ Maximum Drawdown
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    max_drawdown = np.max(drawdown)
    
    return max_drawdown


def calculate_win_rate(trades_df):
    """
    คำนวณอัตราส่วนชนะ (Win Rate)
    
    Args:
        trades_df (pd.DataFrame): DataFrame ของประวัติการเทรด
        
    Returns:
        float: อัตราส่วนชนะ (%)
    """
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if len(trades_df) == 0:
        return 0
    
    # ถ้า trades_df ไม่มีคอลัมน์ profit, คำนวณกำไร/ขาดทุนจากราคาซื้อและขาย
    if 'profit' not in trades_df.columns:
        # ต้องมีคอลัมน์ type และ price
        if 'type' not in trades_df.columns or 'price' not in trades_df.columns:
            return 0
        
        # แยกเป็นการซื้อและขาย
        buys = trades_df[trades_df['type'] == 'buy']
        sells = trades_df[trades_df['type'] == 'sell']
        
        # ถ้าไม่มีการขาย, return 0
        if len(sells) == 0:
            return 0
        
        # คำนวณกำไร/ขาดทุนสำหรับแต่ละการขาย
        profits = []
        for idx, sell in sells.iterrows():
            # หาการซื้อที่เกี่ยวข้อง (FIFO)
            # ในตัวอย่างนี้, ใช้วิธีง่ายๆ โดยหาการซื้อล่าสุดก่อนการขาย
            prev_buys = buys[buys.index < idx]
            if len(prev_buys) == 0:
                continue
            
            # ใช้ราคาซื้อล่าสุด
            last_buy = prev_buys.iloc[-1]
            profit = sell['price'] - last_buy['price']
            profits.append(profit)
        
        # คำนวณอัตราส่วนชนะ
        win_count = sum(1 for p in profits if p > 0)
        win_rate = win_count / len(profits) if len(profits) > 0 else 0
    
    else:
        # ถ้ามีคอลัมน์ profit อยู่แล้ว, ใช้โดยตรง
        win_count = len(trades_df[trades_df['profit'] > 0])
        win_rate = win_count / len(trades_df)
    
    return win_rate


def calculate_profit_factor(trades_df):
    """
    คำนวณ Profit Factor
    
    Args:
        trades_df (pd.DataFrame): DataFrame ของประวัติการเทรด
        
    Returns:
        float: Profit Factor
    """
    # ตรวจสอบว่ามีข้อมูลหรือไม่
    if len(trades_df) == 0:
        return 0
    
    # ถ้า trades_df ไม่มีคอลัมน์ profit, คำนวณกำไร/ขาดทุนจากราคาซื้อและขาย
    if 'profit' not in trades_df.columns:
        # ต้องมีคอลัมน์ type และ price
        if 'type' not in trades_df.columns or 'price' not in trades_df.columns:
            return 0
        
        # แยกเป็นการซื้อและขาย
        buys = trades_df[trades_df['type'] == 'buy']
        sells = trades_df[trades_df['type'] == 'sell']
        
        # ถ้าไม่มีการขาย, return 0
        if len(sells) == 0:
            return 0
        
        # คำนวณกำไร/ขาดทุนสำหรับแต่ละการขาย
        profits = []
        for idx, sell in sells.iterrows():
            # หาการซื้อที่เกี่ยวข้อง (FIFO)
            # ในตัวอย่างนี้, ใช้วิธีง่ายๆ โดยหาการซื้อล่าสุดก่อนการขาย
            prev_buys = buys[buys.index < idx]
            if len(prev_buys) == 0:
                continue
            
            # ใช้ราคาซื้อล่าสุด
            last_buy = prev_buys.iloc[-1]
            profit = sell['price'] - last_buy['price']
            profits.append(profit)
        
        # แยกเป็นกำไรและขาดทุน
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
    
    else:
        # ถ้ามีคอลัมน์ profit อยู่แล้ว, ใช้โดยตรง
        gross_profit = trades_df[trades_df['profit'] > 0]['profit'].sum()
        gross_loss = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
    
    # คำนวณ Profit Factor
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    return profit_factor


if __name__ == "__main__":
    # ทดสอบการใช้งาน
    # สร้างข้อมูลตัวอย่าง
    returns = np.array([0.01, -0.02, 0.03, 0.01, -0.01, 0.02, -0.03])
    values = np.array([1000, 1010, 990, 1020, 1030, 1020, 1040, 1010])
    
    trades = pd.DataFrame({
        'type': ['buy', 'sell', 'buy', 'sell', 'buy', 'sell'],
        'price': [100, 105, 98, 102, 100, 103],
        'amount': [1, 1, 1, 1, 1, 1]
    })
    
    # คำนวณเมตริก
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(values)
    win_rate = calculate_win_rate(trades)
    profit_factor = calculate_profit_factor(trades)
    
    print(f"Sharpe Ratio: {sharpe:.4f}")
    print(f"Sortino Ratio: {sortino:.4f}")
    print(f"Maximum Drawdown: {max_dd:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.4f}")
