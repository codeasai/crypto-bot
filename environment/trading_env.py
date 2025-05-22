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
        
        # Input DataFrame 'df' is expected to be pre-processed by DataProcessor.
        # This includes normalization of features, handling of NaNs, infs, sorting by time,
        # and ensuring all feature columns are numeric.
        if df is None or len(df) == 0:
            logger.error("Input DataFrame is empty or None.")
            raise ValueError("DataFrame cannot be empty or None.")
            
        # Verify essential columns for environment operation (e.g., 'close' for price, 'timestamp' or index for time)
        if 'close' not in df.columns:
            logger.error("Essential 'close' column is missing in DataFrame.")
            raise ValueError("DataFrame must contain a 'close' column.")
        if not isinstance(df.index, pd.DatetimeIndex) and 'timestamp' not in df.columns:
            logger.error("DataFrame must have a DatetimeIndex or a 'timestamp' column.")
            raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column for time reference.")

        self.df = df.copy()
        
        # Ensure data length is sufficient for the window size
        if len(self.df) < window_size:
            logger.error(f"Data length ({len(self.df)}) is less than window_size ({window_size}).")
            raise ValueError(f"Data length must be at least window_size. Got: {len(self.df)}, Window: {window_size}")
        
        # Define instance variables
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.commission_fee = commission_fee # Transaction fee as a fraction (e.g., 0.001 for 0.1%)
        self.use_risk_adjusted_rewards = use_risk_adjusted_rewards
        
        # Trading state variables
        self.current_step = self.window_size # Start after the first window
        self.balance = self.initial_balance # Current account balance
        self.position = 0 # Current position: -1 (short), 0 (neutral), 1 (long) - can be fractional
        self.trades = [] # List to store trade details
        self.account_history = [] # List to store balance history for performance tracking
        
        # Variables for reward calculation
        self.returns = [] # Stores per-step returns for Sharpe/Sortino calculation within an episode
        self.volatility = [] # Stores per-step volatility (std of returns) if needed for reward shaping
        
        # Rendering variables (if applicable)
        self.render_mode = None
        self.fig = None
        self.ax = None
        self.ax2 = None
        self.price_line = None
        self.balance_line = None
        self.buy_markers = None
        self.sell_markers = None
        
        # กำหนดค่าสำหรับการคำนวณ state
        # Define feature columns for state construction (all columns except timestamp and date if present)
        # Assuming 'df' contains normalized features + 'close' price for calculations.
        # Other non-feature columns like raw OHLC should ideally be removed by DataProcessor before passing to env.
        self.feature_columns = [col for col in self.df.columns if col not in ['timestamp', 'date']]
        
        # State size: (number of features * window_size) + 2 (for current balance and position)
        self.state_size = len(self.feature_columns) * window_size + 2
        
        # Action space: [target_position_proportion, leverage]
        # target_position_proportion: -1 (full short) to 1 (full long)
        # leverage: 0 (no leverage) to 1 (max allowed leverage, e.g. 0.5 maps to 50% of what broker allows)
        self.action_space = gym.spaces.Box(
            low=np.array([-1, 0]),  
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # Observation space: Based on the state size.
        # Values can be anything if not strictly normalized in a fixed range (e.g. z-score).
        # If MinMax normalized to [0,1], then low=0, high=1.
        self.observation_space = gym.spaces.Box(
            low=-np.inf, # Using -np.inf, np.inf as features might not be strictly bounded after all operations
            high=np.inf,
            shape=(self.state_size,),
            dtype=np.float32
        )
        logger.info(f"CryptoTradingEnv initialized. State size: {self.state_size}, Window: {self.window_size}")
        logger.info(f"Feature columns for state: {self.feature_columns}")
    
    def reset(self) -> np.ndarray:
        """
        Resets the environment to its initial state.
        This involves resetting the trading step, balance, position, and clearing logs/history.
        
        Returns:
            np.ndarray: The initial state observation.
        """
        self.current_step = self.window_size # Reset step to the start of the data after the initial window
        self.balance = self.initial_balance # Reset balance
        self.position = 0 # Reset position (neutral)
        self.trades = [] # Clear trade log
        self.account_history = [{'step': self.current_step -1, 'balance': self.initial_balance, 'position': 0, 'leverage': 0}] # Initialize account history
        self.returns = [] # Clear returns history for Sharpe ratio calculation
        self.volatility = [] # Clear volatility history
        
        logger.debug(f"Environment reset. Initial balance: {self.balance}")
        return self._get_state()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executes one time step within the environment.
        
        Args:
            action (np.ndarray): A 2-element array: [target_position_proportion, leverage].
                                 target_position_proportion: Desired position (-1 to 1).
                                 leverage: Desired leverage (0 to 1, scaled by env).
            
        Returns:
            Tuple[np.ndarray, float, bool, Dict]: 
                - next_state: The agent's observation of the current environment.
                - reward: Amount of reward returned after previous action.
                - done: Whether the episode has ended.
                - info: Contains auxiliary diagnostic information.
        """
        # Validate and clip action
        target_position, leverage = action
        target_position = np.clip(target_position, -1, 1) # Proportion of balance to allocate
        leverage = np.clip(leverage, 0, 1) # Max leverage proportion (e.g., if 1 means 10x, then 0.5 means 5x)
        
        current_price = self.df.iloc[self.current_step]['close']
        previous_price = self.df.iloc[self.current_step - 1]['close']
        
        # Store previous balance for reward calculation
        prev_balance_for_reward_calc = self.balance 

        # Simulate trade execution
        # Calculate how much the position changes
        position_change = target_position - self.position 
        
        # Calculate transaction costs (commission)
        # Cost is proportional to the change in position size (absolute) multiplied by price and fee rate
        transaction_cost = abs(position_change) * current_price * self.commission_fee * leverage # Leverage amplifies exposure, thus cost
        self.balance -= transaction_cost # Deduct commission from balance

        # Calculate PnL from the previous position held (if any) based on price change
        # This assumes the previous position was held with its associated leverage
        # This is a simplified model. A more complex one would track entry prices.
        price_diff_ratio = (current_price - previous_price) / previous_price
        # PnL from previous position. self.position is from previous step here. Leverage is also from previous step.
        # For simplicity, assume previous leverage was similar or implicitly managed by position size.
        # This part of PnL is tricky if leverage changes dynamically or is not tied to position.
        # The current code structure's PnL is based on the new position and current price change.
        # Let's refine PnL:
        # PnL = units_of_asset * price_change_per_unit
        # units_of_asset = (self.position * self.balance_before_commission_and_new_pnl) / previous_price * leverage_applied_to_position

        # Simplified PnL for the step based on the NEW position and leverage applied to it:
        # This assumes the 'profit' is realized/unrealized based on the new position over the current step's price change.
        # A more standard way:
        # 1. Close previous position: PnL = units * (current_price - entry_price_of_prev_pos) - fees
        # 2. Open new position: Cost = units * current_price + fees
        # The current code is more of a mark-to-market on the new position for the current step's price movement.

        # The original code's PnL: self.balance * self.position * price_change * leverage
        # This implies self.position is the new target_position, and PnL is calculated on it.
        # Let's stick to this logic for now and clarify with comments.
        
        # Profit/Loss calculation for the current step based on the *newly adopted* position and leverage
        # This means the agent decides on a position, and we see how that position would have performed during this step.
        if target_position != 0: # If a non-neutral position is taken
            # Calculate profit/loss based on the change in price during this step,
            # applied to the new position size and leverage.
            # (current_price - previous_price) is the price movement in this step.
            profit_or_loss = target_position * self.balance * price_diff_ratio * leverage
            self.balance += profit_or_loss
        
        # Update current position and leverage for the environment state
        self.position = target_position 
        # self.current_leverage = leverage # If we need to store current leverage explicitly

        # Log trade details
        self.trades.append({
            'step': self.current_step,
            'price': current_price,
            'position_action': target_position, # The position decided by the action
            'leverage_action': leverage, # The leverage decided by the action
            'balance': self.balance,
            'commission_paid': transaction_cost
        })
        
        # Update account history (balance after all operations in the step)
        self.account_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position, # Current holding position
            'leverage': leverage # Leverage applied for this step
        })
        
        # --- Reward Calculation ---
        # Calculate simple percentage return for the current step based on balance change.
        # prev_balance_for_reward_calc holds the balance *before* PnL and commissions for this step.
        step_return = (self.balance - prev_balance_for_reward_calc) / prev_balance_for_reward_calc if prev_balance_for_reward_calc != 0 else 0
        
        if self.use_risk_adjusted_rewards:
            # self.returns stores the history of step_return values for the current episode.
            self.returns.append(step_return) 
            
            # For Sharpe-like reward, calculate mean return and std dev of returns over the episode so far.
            # A minimum number of returns are needed for a meaningful standard deviation.
            if len(self.returns) > 5: # Example: require at least 5 returns for calculation
                current_episode_volatility = np.std(self.returns)
                # self.volatility list is not actively used here but could store current_episode_volatility if needed elsewhere.
                
                if current_episode_volatility > 1e-8: # Avoid division by zero or very small std dev
                    # Sharpe-like ratio for the episode's returns up to this point.
                    # Assumes a risk-free rate of 0, which is common for crypto.
                    sharpe_ratio_episode = np.mean(self.returns) / current_episode_volatility
                    reward = sharpe_ratio_episode # The reward becomes the Sharpe ratio.
                else:
                    # If volatility is zero/too low (e.g., all returns are same, possibly zero),
                    # reward is the simple return for the step. If mean return is also 0, reward is 0.
                    reward = step_return if np.mean(self.returns) != 0 else 0.0
            else:
                # Not enough history for a meaningful volatility/Sharpe calculation, use simple step_return.
                reward = step_return 
        else:
            # Simple reward: the percentage return for the current step.
            reward = step_return
        
        # Advance to the next step in the data
        self.current_step += 1
        
        # Check if the episode is done
        done = (self.balance <= 0) or (self.current_step >= len(self.df))
        
        # Compile additional information
        info = {
            'total_profit': self.balance - self.initial_balance,
            'total_trades': len(self.trades), # Number of times a decision was made (could be refined to actual trades)
            'current_price': current_price,
            'current_position_held': self.position,
            'current_leverage_applied': leverage,
            'current_balance': self.balance,
            'trade_executed': position_change != 0, # True if position was changed
            'trade_profit': self.balance - prev_balance_for_reward_calc if position_change !=0 else 0 # Profit of this specific trade action
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Constructs the current state observation for the agent.
        The state is composed of three main parts:
        1. Windowed Market Features: A flattened array of historical market data
           (e.g., normalized prices, technical indicators) over `self.window_size` previous time steps.
           These features are selected from `self.feature_columns`.
        2. Normalized Account Balance: The current account balance normalized by the initial balance.
           This provides a relative measure of performance. Clipped to avoid extreme values.
        3. Normalized Position: The current trading position (e.g., -1 for short, 0 for neutral, 1 for long).
           This is already in a normalized range.
        
        Returns:
            np.ndarray: A 1D NumPy array representing the current state, flattened and cast to float32.
                        The length of this array must match `self.state_size`.
        """
        # Get windowed data: features from (current_step - window_size) up to (current_step - 1).
        # This represents the market data leading up to the current decision point at `self.current_step`.
        start_idx = self.current_step - self.window_size
        end_idx = self.current_step # iloc is exclusive for the end index, so it takes up to current_step - 1
        window_data_features = self.df.iloc[start_idx:end_idx][self.feature_columns]
        
        # Flatten the windowed market features into a 1D array.
        # DataProcessor should have ensured these features are numeric and appropriately normalized (e.g., MinMax, Z-score).
        state_features = window_data_features.values.flatten()
        
        # Normalize the current account balance relative to the initial balance.
        # This gives a sense of profit/loss. It can exceed 1.0 if profitable.
        # Clipping is applied to prevent extremely large or small values from destabilizing the agent.
        normalized_balance = self.balance / self.initial_balance
        normalized_balance = np.clip(normalized_balance, 0, 10) # Example: Cap at 10x profit, min 0 for state representation
        
        # The current position is typically already scaled (e.g., between -1 and 1),
        # representing the proportion of the portfolio allocated or direction (short/long).
        normalized_position = self.position
        
        # Concatenate all parts of the state: historical market features, normalized balance, and current position.
        state = np.concatenate((state_features, [normalized_balance, normalized_position])).astype(np.float32)
        
        # Critical Sanity Check: Ensure the constructed state's length matches the predefined self.state_size.
        # A mismatch here indicates a severe problem, likely in feature selection during DataFrame processing
        # or in the calculation of self.state_size in the __init__ method.
        if state.shape[0] != self.state_size:
             logger.error(f"CRITICAL STATE SHAPE MISMATCH: Expected state size {self.state_size}, but got {state.shape[0]}. "
                          f"Number of features processed: {len(state_features)} from {len(self.feature_columns)} feature columns over window {self.window_size}.")
             # Handling mismatch: Padding or truncating can hide the root cause.
             # It's often better to raise an error during development to force a fix.
             # For robustness in production (if absolutely necessary and understood), one might pad/truncate.
             if state.shape[0] > self.state_size:
                 state = state[:self.state_size]
             else:
                 state = np.pad(state, (0, self.state_size - state.shape[0]), 'constant')
        return state
    
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