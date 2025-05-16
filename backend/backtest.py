"""
สคริปต์สำหรับการทดสอบย้อนหลัง (backtesting) โมเดล Reinforcement Learning สำหรับ Crypto Trading Bot
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import argparse
from typing import Dict, List, Tuple, Optional

# นำเข้าโมดูลที่เราสร้างขึ้น
from utils.data_processor import DataProcessor
from environment.trading_env import CryptoTradingEnv
from models.dqn_agent import DQNAgent


def backtest_model(model_path: str,
                  symbol: str,
                  timeframe: str,
                  start_date: Optional[str],
                  end_date: Optional[str],
                  initial_balance: float = 10000.0,
                  window_size: int = 10,
                  commission_fee: float = 0.001,
                  output_dir: str = 'backtest_results'):
    """
    ทดสอบย้อนหลังโมเดล RL ที่ฝึกสอนแล้ว
    
    Args:
        model_path (str): เส้นทางไปยังไฟล์โมเดล
        symbol (str): สัญลักษณ์คู่เหรียญ
        timeframe (str): กรอบเวลา
        start_date (Optional[str]): วันที่เริ่มต้นสำหรับการทดสอบ
        end_date (Optional[str]): วันที่สิ้นสุดสำหรับการทดสอบ
        initial_balance (float): เงินทุนเริ่มต้น
        window_size (int): ขนาดหน้าต่างข้อมูลย้อนหลัง
        commission_fee (float): ค่าธรรมเนียมการซื้อขาย
        output_dir (str): โฟลเดอร์สำหรับบันทึกผลลัพธ์
    """
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # สร้างโฟลเดอร์ย่อยสำหรับผลลัพธ์การทดสอบนี้
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"backtest_{symbol}_{timeframe}_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir)
    
    # โหลดและเตรียมข้อมูล
    print(f"กำลังโหลดและเตรียมข้อมูลสำหรับการทดสอบ {symbol} ที่กรอบเวลา {timeframe}...")
    data_processor = DataProcessor()
    test_data = data_processor.create_features_for_backtesting(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"ข้อมูลทดสอบ: {len(test_data)} แถว")
    
    # สร้างสภาพแวดล้อมการเทรด
    env = CryptoTradingEnv(
        df=test_data,
        window_size=window_size,
        initial_balance=initial_balance,
        commission_fee=commission_fee,
        use_risk_adjusted_rewards=True
    )
    
    # โหลดโมเดล
    print(f"กำลังโหลดโมเดลจาก {model_path}...")
    model = keras.models.load_model(model_path)
    
    # สร้างตัวแทน DQN
    state_size = env.observation_space.shape[0]
    discrete_action_size = 7  # เหมือนกับตอนฝึกสอน
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=discrete_action_size,
        learning_rate=0.001,  # ไม่สำคัญเพราะเราไม่ได้ฝึกสอน
        batch_size=32
    )
    
    # โหลดน้ำหนักของโมเดล
    agent.model = model
    
    # ทดสอบย้อนหลัง
    print(f"เริ่มการทดสอบย้อนหลัง...")
    
    # รีเซ็ตสภาพแวดล้อม
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    
    # ข้อมูลการเทรด
    trades = []
    portfolio_values = []
    actions_taken = []
    
    while not done:
        # เลือกการกระทำ
        action_idx = agent.act(state[0], training=False)
        
        # แปลง discrete action เป็น continuous action
        if action_idx == 0:  # ขายหนัก
            action = np.array([-1.0, 1.0])
        elif action_idx == 1:  # ขายปานกลาง
            action = np.array([-0.66, 0.66])
        elif action_idx == 2:  # ขายเบา
            action = np.array([-0.33, 0.33])
        elif action_idx == 3:  # ถือครอง
            action = np.array([0.0, 0.0])
        elif action_idx == 4:  # ซื้อเบา
            action = np.array([0.33, 0.33])
        elif action_idx == 5:  # ซื้อปานกลาง
            action = np.array([0.66, 0.66])
        elif action_idx == 6:  # ซื้อหนัก
            action = np.array([1.0, 1.0])
        
        # ดำเนินการตามการกระทำ
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        
        # บันทึกข้อมูล
        portfolio_values.append({
            'timestamp': test_data.iloc[info['step']]['timestamp'] if 'timestamp' in test_data.columns else info['step'],
            'price': info['price'],
            'total_value': info['total_value'],
            'balance': info['balance'],
            'position': info['position']
        })
        
        actions_taken.append({
            'timestamp': test_data.iloc[info['step']]['timestamp'] if 'timestamp' in test_data.columns else info['step'],
            'action_idx': action_idx,
            'action_x': action[0],  # ทิศทาง
            'action_y': action[1],  # ขนาด
            'price': info['price']
        })
        
        # อัพเดทสถานะ
        state = next_state
    
    # บันทึกประวัติการเทรด
    trades = env.trades_history
    
    # คำนวณเมตริกประสิทธิภาพ
    performance_metrics = env.get_performance_metrics()
    
    # เปรียบเทียบกับกลยุทธ์ HODL
    hodl_returns = (test_data.iloc[-1]['close'] / test_data.iloc[0]['close']) - 1
    performance_metrics['hodl_returns'] = hodl_returns
    performance_metrics['vs_hodl'] = performance_metrics['total_return'] - hodl_returns
    
    # แสดงผลลัพธ์
    print("\n====== ผลลัพธ์การทดสอบย้อนหลัง ======")
    print(f"ผลตอบแทนรวม: {performance_metrics['total_return']:.4f} ({performance_metrics['total_return']*100:.2f}%)")
    print(f"ผลตอบแทนรายปี: {performance_metrics['annualized_return']:.4f} ({performance_metrics['annualized_return']*100:.2f}%)")
    print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {performance_metrics['max_drawdown']:.4f} ({performance_metrics['max_drawdown']*100:.2f}%)")
    print(f"Win Rate: {performance_metrics['win_rate']:.4f} ({performance_metrics['win_rate']*100:.2f}%)")
    print(f"Profit Factor: {performance_metrics['profit_factor']:.4f}")
    print(f"จำนวนการเทรด: {performance_metrics['trade_count']}")
    print(f"กลยุทธ์ HODL: {hodl_returns:.4f} ({hodl_returns*100:.2f}%)")
    print(f"เทียบกับ HODL: {performance_metrics['vs_hodl']:.4f} ({performance_metrics['vs_hodl']*100:.2f}%)")
    
    # บันทึกผลลัพธ์
    with open(os.path.join(run_dir, 'performance_metrics.txt'), 'w') as f:
        for key, value in performance_metrics.items():
            f.write(f"{key}: {value}\n")
    
    # บันทึกประวัติพอร์ตโฟลิโอ
    pd.DataFrame(portfolio_values).to_csv(os.path.join(run_dir, 'portfolio_values.csv'), index=False)
    
    # บันทึกประวัติการกระทำ
    pd.DataFrame(actions_taken).to_csv(os.path.join(run_dir, 'actions_taken.csv'), index=False)
    
    # บันทึกประวัติการเทรด
    pd.DataFrame(trades).to_csv(os.path.join(run_dir, 'trades.csv'), index=False)
    
    # สร้างกราฟแสดงผลการเทรด
    plt.figure(figsize=(15, 10))
    env.render(mode='human')
    plt.savefig(os.path.join(run_dir, 'backtest_trades.png'))
    plt.close()
    
    # สร้างกราฟมูลค่าพอร์ตโฟลิโอเทียบกับเวลา
    portfolio_df = pd.DataFrame(portfolio_values)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_df['timestamp'], portfolio_df['price'], label='Price', color='gray')
    plt.title('Price History')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(portfolio_df['timestamp'], portfolio_df['total_value'], label='Portfolio Value', color='green')
    
    # เพิ่มเส้น HODL
    hodl_values = [initial_balance * (1 + (portfolio_df.iloc[i]['price'] / portfolio_df.iloc[0]['price'] - 1)) 
                   for i in range(len(portfolio_df))]
    plt.plot(portfolio_df['timestamp'], hodl_values, label='HODL Strategy', color='blue', linestyle='--')
    
    plt.title('Portfolio Value vs HODL Strategy')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'portfolio_value.png'))
    plt.close()
    
    # สร้างกราฟแสดงการกระทำที่เลือก
    actions_df = pd.DataFrame(actions_taken)
    
    plt.figure(figsize=(15, 5))
    plt.plot(actions_df['timestamp'], actions_df['action_idx'], marker='o', linestyle='-', markersize=2)
    plt.yticks(range(7), ['Strong Sell', 'Medium Sell', 'Light Sell', 'HODL', 'Light Buy', 'Medium Buy', 'Strong Buy'])
    plt.title('Actions Taken')
    plt.xlabel('Time')
    plt.ylabel('Action')
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, 'actions_taken.png'))
    plt.close()
    
    print(f"ผลลัพธ์การทดสอบย้อนหลังถูกบันทึกไว้ที่: {run_dir}")
    
    return {
        'performance_metrics': performance_metrics,
        'portfolio_values': portfolio_values,
        'trades': trades,
        'actions_taken': actions_taken,
        'output_dir': run_dir
    }


def main():
    """
    ฟังก์ชันหลักสำหรับการรันสคริปต์
    """
    parser = argparse.ArgumentParser(description='ทดสอบย้อนหลังโมเดล Crypto Trading')
    parser.add_argument('--model_path', type=str, required=True, help='เส้นทางไปยังไฟล์โมเดล')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='สัญลักษณ์คู่เหรียญ')
    parser.add_argument('--timeframe', type=str, default='1h', help='กรอบเวลา')
    parser.add_argument('--start_date', type=str, help='วันที่เริ่มต้น (เช่น 2020-01-01)')
    parser.add_argument('--end_date', type=str, help='วันที่สิ้นสุด (เช่น 2022-12-31)')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='เงินทุนเริ่มต้น')
    parser.add_argument('--window_size', type=int, default=10, help='ขนาดหน้าต่างข้อมูลย้อนหลัง')
    parser.add_argument('--commission_fee', type=float, default=0.001, help='ค่าธรรมเนียมการซื้อขาย')
    parser.add_argument('--output_dir', type=str, default='backtest_results', help='โฟลเดอร์สำหรับบันทึกผลลัพธ์')
    
    args = parser.parse_args()
    
    backtest_model(
        model_path=args.model_path,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        window_size=args.window_size,
        commission_fee=args.commission_fee,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()