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
        
        # ตรวจสอบข้อมูล
        if df.empty:
            raise ValueError("DataFrame ไม่สามารถเป็นค่าว่างได้")
            
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame ต้องมีคอลัมน์ต่อไปนี้: {required_columns}")
        
        self.df = df.copy()
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
        self.win_rate = 0.0
        
        # กำหนดขนาดของ state space
        self.num_features = len(self.df.columns) - 1  # ลบคอลัมน์ timestamp ออก
        
        # กำหนด action space ด้วย float64
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float64), 
            high=np.array([1.0, 1.0], dtype=np.float64), 
            dtype=np.float64
        )
        
        # กำหนด observation space ด้วย float64
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.window_size * self.num_features + 2,),
            dtype=np.float64
        )
        
        # ตั้งค่าสำหรับการแสดงผล
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.is_rendering = False
        
        # ตั้งค่าสถานะเริ่มต้น
        self.reset()
        
    def _get_observation(self) -> np.ndarray:
        """
        สร้าง observation (state) ปัจจุบันสำหรับตัวแทน (agent)
        
        Returns:
            np.ndarray: สถานะปัจจุบันประกอบด้วยข้อมูลตลาดและพอร์ตโฟลิโอ
        """
        try:
            # ตัวแปรของพอร์ตโฟลิโอ
            portfolio_features = np.array([
                self.balance / self.initial_balance,  # เงินสดที่มีเทียบกับเงินทุนเริ่มต้น
                self.current_position                 # ตำแหน่งปัจจุบัน (สัดส่วนของพอร์ตที่ลงทุนในสินทรัพย์)
            ], dtype=np.float64)
            
            # ข้อมูลตลาดย้อนหลัง (ตัวชี้วัดทางเทคนิคและราคา)
            market_data = []
            for i in range(self.current_step - self.window_size + 1, self.current_step + 1):
                if i >= 0 and i < len(self.df):
                    market_data.extend(self.df.iloc[i, 1:].values)  # เริ่มจาก 1 เพื่อข้ามคอลัมน์ timestamp
                else:
                    # ถ้าไม่มีข้อมูลย้อนหลังเพียงพอ ให้ใช้ค่า 0
                    market_data.extend([0.0] * (len(self.df.columns) - 1))
            
            # รวมข้อมูลทั้งหมด
            observation = np.concatenate([portfolio_features, market_data])
            return observation.astype(np.float64)
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้าง observation: {str(e)}")
            raise
        
    def _calculate_reward(self, profit: float) -> float:
        """
        คำนวณ reward จากกำไร/ขาดทุน
        
        Args:
            profit (float): กำไร/ขาดทุน
            
        Returns:
            float: reward
        """
        try:
            if self.use_risk_adjusted_rewards:
                # คำนวณ Sharpe Ratio
                returns = self.df['close'].pct_change().dropna()
                if len(returns) > 0 and returns.std() > 0:
                    sharpe = np.sqrt(252) * (returns.mean() / returns.std())
                    return profit * (1 + sharpe) * self.reward_scaling
            return profit * self.reward_scaling
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณ reward: {str(e)}")
            return 0.0
    
    def _update_account_history(self):
        """
        บันทึกประวัติสถานะบัญชีในแต่ละขั้นตอน
        """
        try:
            current_value = self.balance + self.current_position * self.current_price
            if not np.isfinite(current_value):
                current_value = 0.0
                
            self.account_history.append({
                'step': self.current_step,
                'timestamp': self.df.index[self.current_step],
                'price': self.current_price,
                'balance': self.balance,
                'position': self.current_position,
                'total_value': current_value
            })
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอัพเดทประวัติบัญชี: {str(e)}")
    
    def _take_action(self, action: np.ndarray):
        """
        ดำเนินการตามการกระทำที่ได้รับจากตัวแทน (agent)
        """
        try:
            current_price = self.df.iloc[self.current_step, 1]  # ราคาปัจจุบัน
            if not np.isfinite(current_price) or current_price <= 0:
                logger.warning(f"ราคาไม่ถูกต้อง: {current_price}")
                return
                
            self.current_price = current_price
            
            action_type = action[0]  # ทิศทาง
            action_size = action[1]  # ขนาด
            
            # คำนวณมูลค่าพอร์ตโฟลิโอปัจจุบัน
            current_value = self.balance + self.current_position * current_price
            
            # ตรวจสอบค่าไม่ถูกต้อง
            if not np.isfinite(current_value) or current_value <= 0:
                logger.warning(f"มูลค่าพอร์ตโฟลิโอไม่ถูกต้อง: {current_value}")
                return
                
            # กำหนดเงินทุนสูงสุดที่จะใช้ในการเทรดครั้งนี้
            max_trade_value = current_value * action_size * self.max_position
            
            # ตรวจสอบค่าไม่ถูกต้องและขนาดการเทรด
            if not np.isfinite(max_trade_value) or max_trade_value <= 0:
                return  # ไม่ต้องแสดง log ถ้าค่าเป็น 0 หรือไม่ถูกต้อง
                
            # ตัดสินใจว่าจะซื้อหรือขาย
            if action_type > 0:  # ซื้อ
                if action_type * action_size > 0.05:  # ถ้าการกระทำมีขนาดใหญ่พอ
                    # คำนวณจำนวนสินทรัพย์ที่จะซื้อ
                    available_amount = min(max_trade_value, self.balance)
                    if available_amount <= 0:
                        return
                        
                    # หักค่าธรรมเนียม
                    fee = available_amount * self.commission_fee
                    amount_to_buy = (available_amount - fee) / current_price
                    
                    if np.isfinite(amount_to_buy) and amount_to_buy > 0:
                        # อัพเดทสถานะ
                        self.balance -= (available_amount)
                        self.current_position += amount_to_buy
                        
                        # บันทึกประวัติการเทรด
                        self.trades_history.append({
                            'step': self.current_step,
                            'timestamp': self.df.index[self.current_step],
                            'type': 'buy',
                            'price': current_price,
                            'amount': amount_to_buy,
                            'fee': fee,
                            'value': available_amount
                        })
                    
            elif action_type < 0:  # ขาย
                if abs(action_type) * action_size > 0.05:  # ถ้าการกระทำมีขนาดใหญ่พอ
                    # คำนวณจำนวนสินทรัพย์ที่จะขาย
                    position_value = self.current_position * current_price
                    if position_value <= 0:
                        return
                        
                    amount_to_sell_value = min(max_trade_value, position_value)
                    amount_to_sell = amount_to_sell_value / current_price
                    
                    if np.isfinite(amount_to_sell) and amount_to_sell > 0:
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
                            'timestamp': self.df.index[self.current_step],
                            'type': 'sell',
                            'price': current_price,
                            'amount': amount_to_sell,
                            'fee': fee,
                            'value': amount_to_sell_value
                        })
                        
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการดำเนินการ: {str(e)}")
            logger.error(f"สถานะปัจจุบัน: step={self.current_step}, balance={self.balance}, position={self.current_position}")
            logger.error(f"การกระทำ: type={action[0]}, size={action[1]}")
            raise
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        ดำเนินการตามการกระทำและเลื่อนไปยังสถานะถัดไป
        """
        try:
            # ตรวจสอบว่าจบการจำลองหรือยัง
            done = self.current_step >= len(self.df) - 1
            
            if done:
                # คำนวณ win rate ก่อนจบ
                trades = [(b['price'], s['price']) for b, s in zip(
                    [t for t in self.trades_history if t['type'] == 'buy'],
                    [t for t in self.trades_history if t['type'] == 'sell']
                )]
                wins = sum(1 for b, s in trades if s > b)
                self.win_rate = wins / len(trades) if trades else 0.0
                
                info = {
                    'total_profit': self.total_profit,
                    'total_trades': len(self.trades_history),
                    'win_rate': self.win_rate,
                    'final_balance': self.balance,
                    'final_position': self.current_position
                }
                return self._get_observation(), 0, done, info
            
            # ดำเนินการตามการกระทำ
            self._take_action(action)
            
            # บันทึกประวัติบัญชี
            self._update_account_history()
            
            # เลื่อนไปยังขั้นตอนถัดไป
            self.current_step += 1
            
            # คำนวณรางวัลและกำไร
            reward = self._calculate_reward(self.total_profit)
            self.total_reward += reward
            
            # คำนวณกำไรโดยรวม
            current_value = self.balance + self.current_position * self.current_price
            if current_value > 0 and np.isfinite(current_value):
                self.total_profit = (current_value / self.initial_balance) - 1
            else:
                self.total_profit = -1.0
            
            # ตรวจสอบเงื่อนไขการจบพิเศษ
            if current_value <= 0 or not np.isfinite(current_value):
                done = True
                reward = -1
            
            # คำนวณ drawdown
            max_value = max([record['total_value'] for record in self.account_history])
            if max_value > 0 and np.isfinite(max_value):
                drawdown = (max_value - current_value) / max_value
            else:
                drawdown = 0.0
            
            if drawdown > 0.2:
                reward -= drawdown * 0.1
            
            # สรุปข้อมูล
            info = {
                'step': self.current_step,
                'total_profit': self.total_profit,
                'reward': self.total_reward,
                'drawdown': drawdown,
                'balance': self.balance,
                'position': self.current_position
            }
            
            return self._get_observation(), reward, done, info
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในขั้นตอนการทำงาน: {str(e)}")
            return self._get_observation(), -1, True, {}
    
    def reset(self) -> np.ndarray:
        """
        รีเซ็ตสภาพแวดล้อมและเริ่มต้นการจำลองใหม่
        
        Returns:
            np.ndarray: สถานะเริ่มต้น
        """
        try:
            # รีเซ็ตตัวแปรสถานะภายใน
            self.current_step = self.window_size
            self.balance = self.initial_balance
            self.current_position = 0
            self.current_price = self.df.iloc[self.current_step, 1]  # ราคาเริ่มต้น
            self.account_history = []
            self.trades_history = []
            self.total_reward = 0
            self.total_profit = 0
            
            # รีเซ็ตการแสดงผล
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax1 = None
                self.ax2 = None
                self.is_rendering = False
            
            # บันทึกสถานะเริ่มต้น
            self._update_account_history()
            
            return self._get_observation()
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการรีเซ็ต: {str(e)}")
            raise
    
    def render(self, mode='human'):
        """
        แสดงผลสถานะปัจจุบันของสภาพแวดล้อม
        
        Args:
            mode (str): โหมดการแสดงผล ('human' หรือ 'rgb_array')
            
        Returns:
            np.ndarray หรือ None: ภาพสำหรับ 'rgb_array' หรือ None สำหรับ 'human'
        """
        # ยกเลิกการแสดงผล chart
        return None
    
    def close(self):
        """
        ปิดสภาพแวดล้อม
        """
        try:
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                self.ax1 = None
                self.ax2 = None
                self.is_rendering = False
            plt.ioff()  # ปิด interactive mode
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการปิดกราฟ: {str(e)}")
        
    def get_performance_metrics(self) -> Dict:
        """
        คำนวณเมตริกประสิทธิภาพ
        """
        try:
            if len(self.account_history) < 2:
                return {}
            
            # คำนวณผลตอบแทน
            initial = self.account_history[0]['total_value']
            final = self.account_history[-1]['total_value']
            returns = [(r['total_value'] / p['total_value'] - 1) 
                      for r, p in zip(self.account_history[1:], self.account_history[:-1])]
            
            # คำนวณ Sharpe Ratio
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
            
            # คำนวณ Drawdown
            peak = initial
            max_dd = 0
            for r in self.account_history:
                if r['total_value'] > peak:
                    peak = r['total_value']
                dd = (peak - r['total_value']) / peak
                max_dd = max(max_dd, dd)
            
            # คำนวณ Win Rate
            trades = [(b['price'], s['price']) for b, s in zip(
                [t for t in self.trades_history if t['type'] == 'buy'],
                [t for t in self.trades_history if t['type'] == 'sell']
            )]
            wins = sum(1 for b, s in trades if s > b)
            win_rate = wins / len(trades) if trades else 0
            
            return {
                'return': (final / initial - 1),
                'sharpe': sharpe,
                'drawdown': max_dd,
                'win_rate': win_rate,
                'trades': len(trades)
            }
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการคำนวณเมตริก: {str(e)}")
            return {}