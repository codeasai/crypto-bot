"""
สภาพแวดล้อมการเทรด (Trading Environment) สำหรับ Reinforcement Learning Crypto Trading Bot
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class CryptoTradingEnv(gym.Env):
    """
    คลาสสภาพแวดล้อมการเทรดสินทรัพย์คริปโต สำหรับการเรียนรู้แบบเสริมแรง
    ใช้ OpenAI Gym interface สำหรับการจำลองการเทรด
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 window_size: int = 10,
                 initial_balance: float = 10000.0,
                 commission_fee: float = 0.001,
                 max_position: float = 1.0,
                 reward_scaling: float = 0.01,
                 use_risk_adjusted_rewards: bool = True):
        """
        กำหนดค่าเริ่มต้นสำหรับสภาพแวดล้อมการเทรด
        
        Args:
            df (pd.DataFrame): DataFrame ที่มีข้อมูลราคาและตัวชี้วัดทางเทคนิค
            window_size (int): ขนาดหน้าต่างข้อมูลย้อนหลังที่จะใช้เป็น state
            initial_balance (float): เงินทุนเริ่มต้น
            commission_fee (float): ค่าธรรมเนียมการซื้อขาย (เป็นเศษส่วน เช่น 0.001 = 0.1%)
            max_position (float): ตำแหน่งสูงสุดที่อนุญาตให้ถือครอง (1.0 = 100% ของพอร์ต)
            reward_scaling (float): ตัวคูณสำหรับปรับขนาดของรางวัล
            use_risk_adjusted_rewards (bool): ใช้ผลตอบแทนที่ปรับด้วยความเสี่ยงหรือไม่
        """
        super(CryptoTradingEnv, self).__init__()
        
        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_fee = commission_fee
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        self.use_risk_adjusted_rewards = use_risk_adjusted_rewards
        
        # ตัวแปรสถานะภายใน
        self.current_step = 0
        self.account_history = []
        self.trades_history = []
        self.total_reward = 0
        self.total_profit = 0
        
        # กำหนดขนาดของ state space
        self.num_features = len(self.df.columns) - 1  # ลบคอลัมน์ timestamp ออก
        
        # กำหนด action space
        # action[0] คือการตัดสินใจซื้อ/ขาย (ช่วง -1 ถึง 1)
        # action[1] คือขนาดของการเทรด (0 ถึง 1, เป็นเปอร์เซ็นต์ของพอร์ต)
        self.action_space = spaces.Box(
            low=np.array([-1, 0]), 
            high=np.array([1, 1]), 
            dtype=np.float32
        )
        
        # กำหนด observation space
        # รวมข้อมูลตลาดและข้อมูลพอร์ตโฟลิโอ
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size * self.num_features + 2,),  # +2 สำหรับ balance และ position
            dtype=np.float32
        )
        
        # ตั้งค่าสถานะเริ่มต้น
        self.reset()
        
    def _get_observation(self) -> np.ndarray:
        """
        สร้าง observation (state) ปัจจุบันสำหรับตัวแทน (agent)
        
        Returns:
            np.ndarray: สถานะปัจจุบันประกอบด้วยข้อมูลตลาดและพอร์ตโฟลิโอ
        """
        # ตัวแปรของพอร์ตโฟลิโอ
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # เงินสดที่มีเทียบกับเงินทุนเริ่มต้น
            self.current_position                 # ตำแหน่งปัจจุบัน (สัดส่วนของพอร์ตที่ลงทุนในสินทรัพย์)
        ])
        
        # ข้อมูลตลาดย้อนหลัง (ตัวชี้วัดทางเทคนิคและราคา)
        market_data = []
        for i in range(self.current_step - self.window_size + 1, self.current_step + 1):
            market_data.extend(self.df.iloc[i, 1:].values)  # เริ่มจาก 1 เพื่อข้ามคอลัมน์ timestamp
            
        # รวมข้อมูลทั้งหมด
        observation = np.concatenate([portfolio_features, market_data])
        return observation
    
    def _calculate_reward(self) -> float:
        """
        คำนวณค่ารางวัล (reward) สำหรับการกระทำปัจจุบัน
        
        Returns:
            float: ค่ารางวัลสำหรับการกระทำ
        """
        # คำนวณการเปลี่ยนแปลงของมูลค่าพอร์ตโฟลิโอ
        current_value = self.balance + self.current_position * self.current_price
        previous_value = self.account_history[-2]['total_value'] if len(self.account_history) > 1 else self.initial_balance
        
        # คำนวณผลตอบแทน
        pnl = (current_value - previous_value) / previous_value
        
        if self.use_risk_adjusted_rewards:
            # คำนวณ Sharpe Ratio ในระยะสั้น
            if len(self.account_history) >= 30:  # ต้องมีข้อมูลอย่างน้อย 30 วัน
                returns = [
                    (record['total_value'] - prev_record['total_value']) / prev_record['total_value']
                    for record, prev_record in zip(self.account_history[-30:], self.account_history[-31:-1])
                ]
                
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns)
                    # ปรับ reward ด้วย Sharpe Ratio
                    reward = pnl * (1 + sharpe) * self.reward_scaling
                else:
                    reward = pnl * self.reward_scaling
            else:
                reward = pnl * self.reward_scaling
        else:
            reward = pnl * self.reward_scaling
            
        # ใส่บทลงโทษสำหรับการเทรดที่บ่อยเกินไป
        if len(self.trades_history) > 0 and self.trades_history[-1]['step'] == self.current_step - 1:
            reward -= 0.0005  # บทลงโทษสำหรับการเทรดถี่เกินไป
            
        return reward
    
    def _update_account_history(self):
        """
        บันทึกประวัติสถานะบัญชีในแต่ละขั้นตอน
        """
        self.account_history.append({
            'step': self.current_step,
            'timestamp': self.df.iloc[self.current_step, 0],
            'price': self.current_price,
            'balance': self.balance,
            'position': self.current_position,
            'total_value': self.balance + self.current_position * self.current_price
        })
    
    def _take_action(self, action: np.ndarray):
        """
        ดำเนินการตามการกระทำที่ได้รับจากตัวแทน (agent)
        
        Args:
            action (np.ndarray): การกระทำที่ต้องการดำเนินการ
                action[0] = ทิศทาง (-1 ถึง 1, ขาย/ซื้อ)
                action[1] = ขนาด (0 ถึง 1, เปอร์เซ็นต์ของพอร์ต)
        """
        current_price = self.df.iloc[self.current_step, 1]  # ราคาปัจจุบัน
        self.current_price = current_price
        
        action_type = action[0]  # ทิศทาง
        action_size = action[1]  # ขนาด
        
        # คำนวณมูลค่าพอร์ตโฟลิโอปัจจุบัน
        current_value = self.balance + self.current_position * current_price
        
        # กำหนดเงินทุนสูงสุดที่จะใช้ในการเทรดครั้งนี้
        max_trade_value = current_value * action_size * self.max_position
        
        # ตัดสินใจว่าจะซื้อหรือขาย
        if action_type > 0:  # ซื้อ
            if action_type * action_size > 0.05:  # ถ้าการกระทำมีขนาดใหญ่พอ
                # คำนวณจำนวนสินทรัพย์ที่จะซื้อ
                available_amount = min(max_trade_value, self.balance)
                amount_to_buy = available_amount / current_price
                
                # หักค่าธรรมเนียม
                fee = available_amount * self.commission_fee
                amount_to_buy = (available_amount - fee) / current_price
                
                # อัพเดทสถานะ
                self.balance -= (available_amount)
                self.current_position += amount_to_buy
                
                # บันทึกประวัติการเทรด
                self.trades_history.append({
                    'step': self.current_step,
                    'timestamp': self.df.iloc[self.current_step, 0],
                    'type': 'buy',
                    'price': current_price,
                    'amount': amount_to_buy,
                    'fee': fee
                })
                
        elif action_type < 0:  # ขาย
            if abs(action_type) * action_size > 0.05:  # ถ้าการกระทำมีขนาดใหญ่พอ
                # คำนวณจำนวนสินทรัพย์ที่จะขาย
                position_value = self.current_position * current_price
                amount_to_sell_value = min(max_trade_value, position_value)
                amount_to_sell = amount_to_sell_value / current_price
                
                # ได้รับเงิน
                received_amount = amount_to_sell_value
                
                # หักค่าธรรมเนียม
                fee = received_amount * self.commission_fee
                received_amount -= fee
                
                # อัพเดทสถานะ
                self.balance += received_amount
                self.current_position -= amount_to_sell
                
                # บันทึกประวัติการเทรด
                self.trades_history.append({
                    'step': self.current_step,
                    'timestamp': self.df.iloc[self.current_step, 0],
                    'type': 'sell',
                    'price': current_price,
                    'amount': amount_to_sell,
                    'fee': fee
                })
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        ดำเนินการตามการกระทำและเลื่อนไปยังสถานะถัดไป
        
        Args:
            action (np.ndarray): การกระทำที่ต้องการดำเนินการ
        
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: สถานะใหม่, รางวัล, สถานะการจบ, ข้อมูลเพิ่มเติม
        """
        # ตรวจสอบว่าจบการจำลองหรือยัง
        done = self.current_step >= len(self.df) - 1
        
        if done:
            return self._get_observation(), 0, done, {}
        
        # ดำเนินการตามการกระทำ
        self._take_action(action)
        
        # บันทึกประวัติบัญชี
        self._update_account_history()
        
        # เลื่อนไปยังขั้นตอนถัดไป
        self.current_step += 1
        
        # คำนวณรางวัล
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # คำนวณกำไรขาดทุนโดยรวม
        self.total_profit = (self.balance + self.current_position * self.current_price) / self.initial_balance - 1
        
        # ตรวจสอบเงื่อนไขการจบพิเศษ
        # 1. ถ้าเงินในบัญชีหมด
        if self.balance + self.current_position * self.current_price <= 0:
            done = True
            reward = -1  # บทลงโทษสำหรับการล้มละลาย
            
        # 2. ถ้าขาดทุนเกินเกณฑ์ (Maximum Drawdown)
        max_value = max([record['total_value'] for record in self.account_history])
        current_value = self.balance + self.current_position * self.current_price
        drawdown = (max_value - current_value) / max_value
        
        if drawdown > 0.2:  # Maximum Drawdown 20%
            reward -= drawdown * 0.1  # เพิ่มบทลงโทษตามระดับของ drawdown
        
        # ข้อมูลเพิ่มเติม
        info = {
            'step': self.current_step,
            'timestamp': self.df.iloc[self.current_step, 0] if self.current_step < len(self.df) else self.df.iloc[-1, 0],
            'price': self.current_price,
            'balance': self.balance,
            'position': self.current_position,
            'total_value': self.balance + self.current_position * self.current_price,
            'total_profit': self.total_profit,
            'total_reward': self.total_reward,
            'drawdown': drawdown
        }
        
        return self._get_observation(), reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        รีเซ็ตสภาพแวดล้อมและเริ่มต้นการจำลองใหม่
        
        Returns:
            np.ndarray: สถานะเริ่มต้น
        """
        # รีเซ็ตตัวแปรสถานะภายใน
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.current_position = 0
        self.current_price = self.df.iloc[self.current_step, 1]  # ราคาเริ่มต้น
        self.account_history = []
        self.trades_history = []
        self.total_reward = 0
        self.total_profit = 0
        
        # บันทึกสถานะเริ่มต้น
        self._update_account_history()
        
        return self._get_observation()
    
    def render(self, mode='human'):
        """
        แสดงผลสถานะปัจจุบันของสภาพแวดล้อม
        
        Args:
            mode (str): โหมดการแสดงผล ('human' หรือ 'rgb_array')
            
        Returns:
            np.ndarray หรือ None: ภาพสำหรับ 'rgb_array' หรือ None สำหรับ 'human'
        """
        if len(self.account_history) < 2:
            return
            
        # สร้างข้อมูลสำหรับการพล็อต
        steps = [record['step'] for record in self.account_history]
        timestamps = [record['timestamp'] for record in self.account_history]
        prices = [record['price'] for record in self.account_history]
        total_values = [record['total_value'] for record in self.account_history]
        
        # สร้างจุดซื้อและขาย
        buy_steps = [trade['step'] for trade in self.trades_history if trade['type'] == 'buy']
        buy_prices = [trade['price'] for trade in self.trades_history if trade['type'] == 'buy']
        sell_steps = [trade['step'] for trade in self.trades_history if trade['type'] == 'sell']
        sell_prices = [trade['price'] for trade in self.trades_history if trade['type'] == 'sell']
        
        # สร้างกราฟ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # กราฟราคา
        ax1.plot(steps, prices, label='Price', color='blue')
        ax1.scatter(buy_steps, buy_prices, marker='^', color='green', label='Buy')
        ax1.scatter(sell_steps, sell_prices, marker='v', color='red', label='Sell')
        ax1.set_ylabel('Price')
        ax1.set_title('Price and Trades')
        ax1.legend()
        ax1.grid(True)
        
        # กราฟมูลค่าพอร์ตโฟลิโอ
        ax2.plot(steps, total_values, label='Portfolio Value', color='purple')
        ax2.set_ylabel('Value')
        ax2.set_title('Portfolio Value')
        ax2.legend()
        ax2.grid(True)
        
        # ปรับแต่งกราฟ
        plt.xlabel('Step')
        plt.tight_layout()
        
        if mode == 'human':
            plt.show()
        elif mode == 'rgb_array':
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return img
            
    def close(self):
        """
        ปิดสภาพแวดล้อม
        """
        plt.close()
        
    def get_performance_metrics(self) -> Dict:
        """
        คำนวณและคืนค่าเมตริกประสิทธิภาพของกลยุทธ์การเทรด
        
        Returns:
            Dict: เมตริกประสิทธิภาพต่างๆ
        """
        if len(self.account_history) < 2:
            return {}
            
        # ข้อมูลกำไรขาดทุน
        initial_value = self.account_history[0]['total_value']
        final_value = self.account_history[-1]['total_value']
        total_return = (final_value / initial_value) - 1
        
        # คำนวณผลตอบแทนรายวัน
        daily_returns = []
        for i in range(1, len(self.account_history)):
            prev_value = self.account_history[i-1]['total_value']
            curr_value = self.account_history[i]['total_value']
            daily_return = (curr_value / prev_value) - 1
            daily_returns.append(daily_return)
        
        # คำนวณ Sharpe Ratio
        sharpe_ratio = 0
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # ปรับเป็นรายปี
        
        # คำนวณ Maximum Drawdown
        peak = self.account_history[0]['total_value']
        max_drawdown = 0
        
        for record in self.account_history:
            if record['total_value'] > peak:
                peak = record['total_value']
            drawdown = (peak - record['total_value']) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # คำนวณอัตราการชนะ
        win_count = 0
        trade_count = len(self.trades_history)
        total_profit = 0
        total_loss = 0
        
        for i in range(1, len(self.trades_history)):
            prev_trade = self.trades_history[i-1]
            curr_trade = self.trades_history[i]
            
            if prev_trade['type'] == 'buy' and curr_trade['type'] == 'sell':
                trade_profit = (curr_trade['price'] - prev_trade['price']) / prev_trade['price']
                
                if trade_profit > 0:
                    win_count += 1
                    total_profit += trade_profit
                else:
                    total_loss += abs(trade_profit)
        
        win_rate = win_count / (trade_count / 2) if trade_count > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # รวมเมตริก
        metrics = {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(self.account_history)) if len(self.account_history) > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trade_count': trade_count,
            'final_balance': self.balance,
            'final_position': self.current_position,
            'final_value': final_value
        }
        
        return metrics