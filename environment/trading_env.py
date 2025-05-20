"""
สภาพแวดล้อมการเทรด (Trading Environment) สำหรับ Reinforcement Learning Crypto Trading Bot
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import logging

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class CryptoTradingEnv(gym.Env):
    """
    สภาพแวดล้อมการเทรดคริปโตสำหรับการฝึกสอนตัวแทน DQN
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 10, initial_balance: float = 10000.0,
                 commission_fee: float = 0.001, use_risk_adjusted_rewards: bool = True):
        """
        กำหนดค่าเริ่มต้นของสภาพแวดล้อม
        
        Args:
            df (pd.DataFrame): ข้อมูลราคาและตัวชี้วัดทางเทคนิค
            window_size (int): ขนาดหน้าต่างข้อมูลย้อนหลัง
            initial_balance (float): เงินทุนเริ่มต้น
            commission_fee (float): ค่าธรรมเนียมการเทรด
            use_risk_adjusted_rewards (bool): ใช้การคำนวณรางวัลที่ปรับตามความเสี่ยงหรือไม่
        """
        super(CryptoTradingEnv, self).__init__()
        
        # ตรวจสอบข้อมูล
        if df is None or len(df) == 0:
            raise ValueError("ไม่พบข้อมูล")
            
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"ไม่พบคอลัมน์ {col} ในข้อมูล")
        
        # ตรวจสอบและแปลงข้อมูล
        self.df = df.copy()
        for col in self.df.columns:
            if col not in ['timestamp', 'date']:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # แทนที่ค่า NaN ด้วยค่าเฉลี่ย
        self.df = self.df.fillna(self.df.mean())
        
        # ตรวจสอบและจัดการค่า inf
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(self.df.mean())
        
        # ตรวจสอบและจัดการค่าลบ
        for col in self.df.select_dtypes(include=np.number).columns:
            if (self.df[col] < 0).any():
                self.df[col] = self.df[col].abs()
        
        # ตรวจสอบและจัดการค่า 0
        for col in self.df.select_dtypes(include=np.number).columns:
            if (self.df[col] == 0).any():
                mean_value = self.df[col].mean()
                self.df[col] = self.df[col].replace(0, mean_value)
        
        # ตรวจสอบความยาวของข้อมูล
        if len(self.df) < window_size:
            raise ValueError(f"ข้อมูลไม่เพียงพอ ต้องการอย่างน้อย {window_size} แท่ง แต่มีเพียง {len(self.df)} แท่ง")
        
        # ตรวจสอบการเรียงลำดับข้อมูล
        if not self.df.index.is_monotonic_increasing:
            self.df = self.df.sort_index()
        
        # กำหนดค่าตัวแปร
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_fee = commission_fee
        self.use_risk_adjusted_rewards = use_risk_adjusted_rewards
        
        # กำหนดค่าสำหรับการเทรด
        self.current_step = window_size
        self.balance = initial_balance
        self.position = 0
        self.trades = []
        self.account_history = []
        
        # กำหนดค่าสำหรับการคำนวณรางวัล
        self.returns = []
        self.volatility = []
        
        # กำหนดค่าสำหรับการแสดงผล
        self.render_mode = None
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.price_line = None
        self.balance_line = None
        self.buy_markers = None
        self.sell_markers = None
        
        # กำหนดค่าสำหรับการคำนวณ state
        self.state_size = len(self.df.columns) * window_size + 2  # +2 สำหรับ balance และ position
        
        # กำหนดค่าสำหรับการคำนวณ action
        self.action_space = gym.spaces.Box(
            low=np.array([-1, 0]),  # [position, leverage]
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # กำหนดค่าสำหรับการคำนวณ observation
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_size,),
            dtype=np.float32
        )
    
    def reset(self):
        """
        รีเซ็ตสภาพแวดล้อม
        
        Returns:
            np.ndarray: สถานะเริ่มต้น
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0
        self.trades = []
        self.account_history = []
        self.returns = []
        self.volatility = []
        
        return self._get_state()
    
    def step(self, action):
        """
        ดำเนินการตาม action ที่ได้รับ
        
        Args:
            action (np.ndarray): [position, leverage]
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # ตรวจสอบ action
        position, leverage = action
        position = np.clip(position, -1, 1)
        leverage = np.clip(leverage, 0, 1)
        
        # คำนวณราคาปัจจุบัน
        current_price = self.df.iloc[self.current_step]['close']
        
        # คำนวณค่าธรรมเนียม
        commission = abs(position - self.position) * current_price * self.commission_fee
        
        # อัพเดทตำแหน่ง
        self.position = position
        
        # อัพเดทยอดเงิน
        self.balance -= commission
        
        # คำนวณกำไร/ขาดทุน
        if self.position != 0:
            price_change = (current_price - self.df.iloc[self.current_step-1]['close']) / self.df.iloc[self.current_step-1]['close']
            profit = self.balance * self.position * price_change * leverage
            self.balance += profit
        
        # บันทึกประวัติ
        self.trades.append({
            'step': self.current_step,
            'price': current_price,
            'position': self.position,
            'leverage': leverage,
            'balance': self.balance,
            'commission': commission
        })
        
        self.account_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'leverage': leverage
        })
        
        # คำนวณรางวัล
        if self.use_risk_adjusted_rewards:
            # คำนวณ return
            if len(self.account_history) > 1:
                prev_balance = self.account_history[-2]['balance']
                current_return = (self.balance - prev_balance) / prev_balance
                self.returns.append(current_return)
                
                # คำนวณ volatility
                if len(self.returns) > 1:
                    current_volatility = np.std(self.returns)
                    self.volatility.append(current_volatility)
                    
                    # คำนวณ Sharpe ratio
                    if current_volatility > 0:
                        sharpe_ratio = np.mean(self.returns) / current_volatility
                        reward = sharpe_ratio
                    else:
                        reward = 0
                else:
                    reward = current_return
            else:
                reward = 0
        else:
            # ใช้ return เป็นรางวัล
            if len(self.account_history) > 1:
                prev_balance = self.account_history[-2]['balance']
                reward = (self.balance - prev_balance) / prev_balance
            else:
                reward = 0
        
        # อัพเดทขั้นตอน
        self.current_step += 1
        
        # ตรวจสอบการจบรอบ
        done = self.current_step >= len(self.df)
        
        # สร้างข้อมูลเพิ่มเติม
        info = {
            'total_profit': self.balance - self.initial_balance,
            'total_trades': len(self.trades),
            'current_price': current_price,
            'position': self.position,
            'leverage': leverage,
            'balance': self.balance
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """
        สร้างสถานะปัจจุบัน
        
        Returns:
            np.ndarray: สถานะปัจจุบัน
        """
        # ดึงข้อมูลย้อนหลัง
        window_data = self.df.iloc[self.current_step-self.window_size:self.current_step]
        
        # สร้าง state
        state = []
        
        # เพิ่มข้อมูลราคาและตัวชี้วัด
        for col in window_data.columns:
            if col not in ['timestamp', 'date']:
                state.extend(window_data[col].values)
        
        # เพิ่มข้อมูลบัญชี
        state.append(self.balance)
        state.append(self.position)
        
        return np.array(state, dtype=np.float32)
    
    def render(self, mode='human'):
        """
        แสดงผลการเทรด
        
        Args:
            mode (str): โหมดการแสดงผล
        """
        if mode == 'human':
            try:
                # สร้าง figure ถ้ายังไม่มี
                if self.fig is None:
                    self.fig, self.ax = plt.subplots(figsize=(8, 6))
                    self.fig.canvas.manager.set_window_title('Trading Environment')
                    self.ax2 = self.ax.twinx()
                    
                    # ตั้งค่าการแสดงผล
                    self.ax.set_title('Trading Environment', fontsize=10)
                    self.ax.set_xlabel('Step', fontsize=8)
                    self.ax.set_ylabel('Price', fontsize=8)
                    self.ax2.set_ylabel('Balance', fontsize=8)
                    
                    # ตั้งค่า grid
                    self.ax.grid(True, linestyle='--', alpha=0.3)
                    
                    # ตั้งค่า legend
                    self.ax.legend(loc='upper left', fontsize=8)
                    self.ax2.legend(loc='upper right', fontsize=8)
                    
                    # สร้างเส้นและ markers
                    self.price_line, = self.ax.plot([], [], label='Price', color='black', linewidth=1)
                    self.balance_line, = self.ax2.plot([], [], label='Balance', color='blue', linewidth=1)
                    self.buy_markers = self.ax.scatter([], [], color='green', marker='^', s=50, label='Buy')
                    self.sell_markers = self.ax.scatter([], [], color='red', marker='v', s=50, label='Sell')
                    
                    plt.tight_layout()
                
                # อัพเดทข้อมูลราคา
                price_data = self.df.iloc[:self.current_step]['close']
                self.price_line.set_data(range(len(price_data)), price_data.values)
                
                # อัพเดทข้อมูลบัญชี
                balance_data = [h['balance'] for h in self.account_history]
                self.balance_line.set_data(range(len(balance_data)), balance_data)
                
                # อัพเดท markers การเทรด
                buy_steps = [t['step'] for t in self.trades if t['position'] > 0]
                buy_prices = [t['price'] for t in self.trades if t['position'] > 0]
                sell_steps = [t['step'] for t in self.trades if t['position'] < 0]
                sell_prices = [t['price'] for t in self.trades if t['position'] < 0]
                
                self.buy_markers.set_offsets(np.c_[buy_steps, buy_prices])
                self.sell_markers.set_offsets(np.c_[sell_steps, sell_prices])
                
                # ปรับแกนให้เหมาะสม
                self.ax.relim()
                self.ax.autoscale_view()
                self.ax2.relim()
                self.ax2.autoscale_view()
                
                # แสดงผล
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)
                
            except Exception as e:
                logger.error(f"เกิดข้อผิดพลาดในการแสดงผล: {str(e)}")
    
    def close(self):
        """
        ปิดการแสดงผล
        """
        try:
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax = None
                self.ax2 = None
                self.price_line = None
                self.balance_line = None
                self.buy_markers = None
                self.sell_markers = None
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการปิดการแสดงผล: {str(e)}")