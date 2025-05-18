"""
ฝึกสอนตัวแทน DQN สำหรับการเทรดสินทรัพย์คริปโต
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# เพิ่ม path ของ root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from data.data_processor import DataProcessor
from environment.trading_env import CryptoTradingEnv
from agents.dqn_agent import DQNAgent
from utils.logger import setup_logger

# ตั้งค่า logger
logger = setup_logger('train')

def preview_data_loading(symbol: str, timeframe: str, start_date: str, end_date: str) -> bool:
    """
    แสดงตัวอย่างข้อมูลที่จะโหลดและขอการยืนยัน
    
    Args:
        symbol (str): สัญลักษณ์คู่เหรียญ
        timeframe (str): กรอบเวลา
        start_date (str): วันที่เริ่มต้น
        end_date (str): วันที่สิ้นสุด
        
    Returns:
        bool: True ถ้าผู้ใช้ยืนยัน, False ถ้าผู้ใช้ยกเลิก
    """
    logger.info("\n=== ข้อมูลที่จะโหลด ===")
    logger.info(f"สัญลักษณ์: {symbol}")
    logger.info(f"กรอบเวลา: {timeframe}")
    logger.info(f"วันที่เริ่มต้น: {start_date}")
    logger.info(f"วันที่สิ้นสุด: {end_date}")
    
    # คำนวณจำนวนแท่งข้อมูลโดยประมาณ
    bars_per_day = 0
    if timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        bars_per_day = int(24 * 60 / minutes)
    elif timeframe == '1h':
        bars_per_day = 24
    elif timeframe == '4h':
        bars_per_day = 6
    elif timeframe == '1d':
        bars_per_day = 1
    else:
        logger.error(f"ไม่รองรับกรอบเวลา {timeframe}")
        return False
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end - start).days
    estimated_bars = days * bars_per_day
    
    logger.info(f"จำนวนวัน: {days} วัน")
    logger.info(f"จำนวนแท่งข้อมูลโดยประมาณ: {estimated_bars} แท่ง")
    
    # ดึงข้อมูลตัวอย่างจริงจาก Binance
    try:
        from data.data_collector import BinanceDataCollector
        # ใช้ช่วงเวลาปัจจุบันสำหรับตัวอย่าง
        current_time = datetime.now()
        sample_start = (current_time - timedelta(days=1)).strftime('%Y-%m-%d')
        sample_end = current_time.strftime('%Y-%m-%d')
        
        collector = BinanceDataCollector(
            symbol=symbol,
            interval=timeframe,
            start_date=sample_start,
            end_date=sample_end,
            testnet=True
        )
        
        # ดึงข้อมูลตัวอย่าง
        sample_data = collector.get_historical_klines()
        if len(sample_data) > 0:
            sample_data = sample_data.head(5)  # เลือก 5 แท่งแรก
            logger.info("\nตัวอย่างข้อมูลล่าสุด (5 แท่ง):")
            logger.info(f"\n{sample_data.to_string()}")
            logger.info("\nหมายเหตุ: ข้อมูลตัวอย่างเป็นข้อมูลล่าสุด 5 แท่ง")
        else:
            logger.warning("ไม่สามารถดึงข้อมูลตัวอย่างได้")
            
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการดึงข้อมูลตัวอย่าง: {str(e)}")
    
    while True:
        choice = input("\nDo you want to proceed with data loading? (Y/n): ").lower()
        if choice in ['', 'y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            logger.info("Data loading cancelled by user")
            return False
        else:
            print("Please press Enter for Yes or type 'n' for No")

def validate_and_confirm_data(df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
    """
    ตรวจสอบและยืนยันข้อมูลก่อนการฝึกสอน
    
    Args:
        df (pd.DataFrame): ข้อมูลที่ต้องการตรวจสอบ
        symbol (str): สัญลักษณ์คู่เหรียญ
        timeframe (str): กรอบเวลา
        
    Returns:
        bool: True ถ้าผู้ใช้ยืนยัน, False ถ้าผู้ใช้ยกเลิก
    """
    if len(df) == 0:
        logger.error("ไม่พบข้อมูล กรุณาตรวจสอบพารามิเตอร์และลองใหม่อีกครั้ง")
        return False
        
    logger.info(f"\nข้อมูลที่ได้สำหรับ {symbol} ({timeframe}):")
    logger.info(f"จำนวนข้อมูลทั้งหมด: {len(df)} แท่ง")
    logger.info(f"ช่วงเวลา: {df.index[0]} ถึง {df.index[-1]}")
    
    # แสดงตัวอย่างข้อมูล 5 แท่งแรก
    logger.info("\nตัวอย่างข้อมูล 5 แท่งแรก:")
    logger.info(f"\n{df.head().to_string()}")
    
    # แสดงสถิติพื้นฐาน
    logger.info("\nสถิติพื้นฐาน:")
    logger.info(f"\n{df.describe().to_string()}")
    
    while True:
        choice = input("\nDo you want to proceed with training? (y/n): ").lower()
        if choice in ['y', 'yes']:
            return True
        elif choice in ['n', 'no']:
            logger.info("Training cancelled by user")
            return False
        else:
            print("Please enter 'y' or 'n'")

def train_dqn_agent(
    symbol: str,
    timeframe: str,
    start_date: str = None,
    end_date: str = None,
    initial_balance: float = 10000.0,
    window_size: int = 10,
    batch_size: int = 64,
    episodes: int = 1000,
    output_dir: str = 'outputs'
):
    """
    ฝึกสอนตัวแทน DQN สำหรับการเทรดสินทรัพย์คริปโต
    
    Args:
        symbol (str): สัญลักษณ์คู่เหรียญ เช่น 'BTCUSDT'
        timeframe (str): กรอบเวลา เช่น '1h'
        start_date (Optional[str]): วันที่เริ่มต้นสำหรับข้อมูลฝึกสอน
        end_date (Optional[str]): วันที่สิ้นสุดสำหรับข้อมูลฝึกสอน
        initial_balance (float): เงินทุนเริ่มต้น
        window_size (int): ขนาดหน้าต่างข้อมูลย้อนหลัง
        batch_size (int): ขนาด batch สำหรับการฝึกสอน
        episodes (int): จำนวนรอบการฝึกสอนทั้งหมด
        output_dir (str): โฟลเดอร์สำหรับบันทึกผลลัพธ์
    """
    try:
        # แสดงตัวอย่างข้อมูลที่จะโหลดและขอการยืนยัน
        if not preview_data_loading(symbol, timeframe, start_date, end_date):
            sys.exit(0)
            
        # สร้างโฟลเดอร์ output ถ้ายังไม่มี
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # สร้างโฟลเดอร์ย่อยสำหรับผลลัพธ์การฝึกสอนนี้
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{symbol}_{timeframe}_{timestamp}"
        run_dir = os.path.join(output_dir, run_name)
        os.makedirs(run_dir)
        
        # โหลดและเตรียมข้อมูล
        logger.info(f"กำลังโหลดและเตรียมข้อมูล {symbol} ที่กรอบเวลา {timeframe}...")
        data_processor = DataProcessor()
        
        # ตรวจสอบไฟล์ข้อมูลที่มีอยู่
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        datasets_dir = os.path.join(parent_dir, 'data', 'datasets')
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
            
        data_file = os.path.join(datasets_dir, f"{symbol}_{timeframe}.csv")
        
        if os.path.exists(data_file):
            logger.info(f"กำลังโหลดข้อมูลจาก {data_file}")
            try:
                test_data = pd.read_csv(data_file, index_col=0, parse_dates=True)
                if len(test_data) > 0:
                    logger.info(f"โหลดข้อมูลสำเร็จ ได้ข้อมูล {len(test_data)} แท่ง")
                    logger.info(f"ช่วงเวลาข้อมูล: {test_data.index[0]} ถึง {test_data.index[-1]}")
                else:
                    raise ValueError("ไฟล์ข้อมูลว่างเปล่า")
            except Exception as e:
                logger.warning(f"ไม่สามารถโหลดข้อมูลจากไฟล์ได้: {str(e)}")
                test_data = None
        else:
            logger.info(f"ไม่พบไฟล์ข้อมูลที่ {data_file}")
            test_data = None
            
        # ถ้าไม่มีข้อมูลหรือข้อมูลไม่ถูกต้อง ให้ดึงข้อมูลใหม่
        if test_data is None or len(test_data) == 0:
            logger.info(f"กำลังดึงข้อมูลใหม่สำหรับ {symbol} ที่กรอบเวลา {timeframe}...")
            from data.data_collector import BinanceDataCollector
            collector = BinanceDataCollector(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                testnet=True
            )
            
            # พยายามอัพเดทข้อมูลก่อน
            logger.info("กำลังอัพเดทข้อมูลให้เป็นปัจจุบัน...")
            test_data = collector.update_historical_data()
            
            # ถ้าไม่มีข้อมูลใหม่ ให้ดึงข้อมูลใหม่ทั้งหมด
            if len(test_data) == 0:
                logger.info("ไม่พบข้อมูลที่อัพเดท กำลังดึงข้อมูลใหม่ทั้งหมด...")
                test_data = collector.get_historical_klines()
                
            if len(test_data) > 0:
                # สร้างชื่อไฟล์ตามช่วงเวลา
                file_name = f"{symbol}_{timeframe}_{test_data.index[0].strftime('%Y%m%d')}_{test_data.index[-1].strftime('%Y%m%d')}.csv"
                data_file = os.path.join(datasets_dir, file_name)
                test_data.to_csv(data_file)
                logger.info(f"บันทึกข้อมูล {len(test_data)} แท่ง ไปที่ {data_file}")
                logger.info(f"ช่วงเวลาข้อมูล: {test_data.index[0]} ถึง {test_data.index[-1]}")
        
        if len(test_data) == 0:
            logger.error("ไม่พบข้อมูลในช่วงเวลาที่ระบุ กรุณาตรวจสอบช่วงเวลาและลองใหม่อีกครั้ง")
            return None
            
        # ตรวจสอบและแปลงข้อมูลให้ถูกต้อง
        if not isinstance(test_data.index, pd.DatetimeIndex):
            test_data.index = pd.to_datetime(test_data.index)
            
        # ตรวจสอบจำนวนข้อมูล
        if len(test_data) < 100:
            logger.error(f"ข้อมูลไม่เพียงพอ ต้องการอย่างน้อย 100 แท่ง แต่มีเพียง {len(test_data)} แท่ง")
            return None
            
        # โหลดข้อมูลด้วยช่วงเวลาที่มีอยู่จริง
        actual_start = test_data.index[0].strftime('%Y-%m-%d')
        actual_end = test_data.index[-1].strftime('%Y-%m-%d')
        logger.info(f"กำลังเตรียมข้อมูลสำหรับการฝึกสอนในช่วงเวลา {actual_start} ถึง {actual_end}")
        
        train_data, val_data, _ = data_processor.prepare_data_for_training(
            symbol=symbol,
            timeframe=timeframe,
            start_date=actual_start,
            end_date=actual_end
        )
        
        # ตรวจสอบและยืนยันข้อมูล
        if not validate_and_confirm_data(train_data, symbol, timeframe):
            sys.exit(0)
            
        logger.info(f"ข้อมูลฝึกสอน: {len(train_data)} แถว, ข้อมูลตรวจสอบ: {len(val_data)} แถว")
        
        # สร้างสภาพแวดล้อมการเทรด
        env = CryptoTradingEnv(
            df=train_data,
            window_size=window_size,
            initial_balance=initial_balance,
            commission_fee=0.001,  # 0.1% ค่าธรรมเนียม
            use_risk_adjusted_rewards=True
        )
        
        # สร้างสภาพแวดล้อมสำหรับการตรวจสอบ
        val_env = CryptoTradingEnv(
            df=val_data,
            window_size=window_size,
            initial_balance=initial_balance,
            commission_fee=0.001,
            use_risk_adjusted_rewards=True
        )
        
        # สร้างตัวแทน DQN
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        
        # แปลง continuous action space เป็น discrete
        # เราจะแบ่งการกระทำเป็น 7 ประเภท:
        # 0: ขายหนัก (ขาย 100% ของตำแหน่ง)
        # 1: ขายปานกลาง (ขาย 66% ของตำแหน่ง)
        # 2: ขายเบา (ขาย 33% ของตำแหน่ง)
        # 3: ถือครอง (ไม่ทำอะไร)
        # 4: ซื้อเบา (ใช้ 33% ของเงินทุนที่มี)
        # 5: ซื้อปานกลาง (ใช้ 66% ของเงินทุนที่มี)
        # 6: ซื้อหนัก (ใช้ 100% ของเงินทุนที่มี)
        discrete_action_size = 7
        
        agent = DQNAgent(
            state_size=state_size,
            action_size=discrete_action_size,
            learning_rate=0.001,
            discount_factor=0.95,
            exploration_rate=1.0,
            exploration_decay=0.995,
            exploration_min=0.01,
            batch_size=batch_size,
            memory_size=10000
        )
        
        # ฝึกสอนตัวแทน
        logger.info(f"เริ่มการฝึกสอนตัวแทน DQN สำหรับ {episodes} รอบ...")
        
        # ตัวแปรสำหรับการติดตาม
        train_rewards = []
        val_rewards = []
        train_profits = []
        val_profits = []
        best_val_profit = -np.inf
        exploration_rates = []
        
        for episode in range(episodes):
            # TRAINING PHASE
            state = env.reset()
            state = np.reshape(state, [1, state_size])
            done = False
            total_reward = 0
            
            while not done:
                # เลือกการกระทำ
                action_idx = agent.act(state[0])
                
                # แปลง discrete action เป็น continuous action สำหรับสภาพแวดล้อม
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
                
                # เก็บประสบการณ์ลงในหน่วยความจำ
                agent.remember(state[0], action_idx, reward, next_state[0], done)
                
                # อัพเดทสถานะ
                state = next_state
                total_reward += reward
            
            # ฝึกสอนตัวแทนด้วย experience replay
            loss = agent.replay()
            
            # บันทึกผลลัพธ์
            train_rewards.append(total_reward)
            train_profits.append(info['total_profit'])
            exploration_rates.append(agent.exploration_rate)
            
            # VALIDATION PHASE
            if episode % 10 == 0:  # ทำการตรวจสอบทุกๆ 10 รอบ
                val_state = val_env.reset()
                val_state = np.reshape(val_state, [1, state_size])
                val_done = False
                val_total_reward = 0
                
                while not val_done:
                    # เลือกการกระทำ (ไม่มีการสำรวจในโหมดตรวจสอบ)
                    val_action_idx = agent.act(val_state[0], training=False)
                    
                    # แปลง discrete action เป็น continuous action
                    if val_action_idx == 0:
                        val_action = np.array([-1.0, 1.0])
                    elif val_action_idx == 1:
                        val_action = np.array([-0.66, 0.66])
                    elif val_action_idx == 2:
                        val_action = np.array([-0.33, 0.33])
                    elif val_action_idx == 3:
                        val_action = np.array([0.0, 0.0])
                    elif val_action_idx == 4:
                        val_action = np.array([0.33, 0.33])
                    elif val_action_idx == 5:
                        val_action = np.array([0.66, 0.66])
                    elif val_action_idx == 6:
                        val_action = np.array([1.0, 1.0])
                    
                    # ดำเนินการตามการกระทำ
                    val_next_state, val_reward, val_done, val_info = val_env.step(val_action)
                    val_next_state = np.reshape(val_next_state, [1, state_size])
                    
                    val_state = val_next_state
                    val_total_reward += val_reward
                
                val_rewards.append(val_total_reward)
                val_profits.append(val_info['total_profit'])
                
                # บันทึกโมเดลที่ดีที่สุด
                if val_info['total_profit'] > best_val_profit:
                    best_val_profit = val_info['total_profit']
                    agent.save(os.path.join(run_dir, 'best_model.h5'))
                    
                    # สร้างภาพแสดงผลการเทรดที่ดีที่สุด
                    plt.figure(figsize=(15, 10))
                    val_env.render(mode='human')
                    plt.savefig(os.path.join(run_dir, 'best_validation_trades.png'))
                    plt.close()
                
                # แสดงความคืบหน้า
                logger.info(f"Episode: {episode+1}/{episodes}, "
                          f"Train Profit: {info['total_profit']:.4f}, "
                          f"Val Profit: {val_info['total_profit']:.4f}, "
                          f"Best Val Profit: {best_val_profit:.4f}, "
                          f"Epsilon: {agent.exploration_rate:.4f}")
        
        # บันทึกโมเดลสุดท้าย
        agent.save(os.path.join(run_dir, 'final_model.h5'))
        
        # บันทึกประวัติการฝึกสอน
        history = {
            'train_rewards': train_rewards,
            'val_rewards': val_rewards,
            'train_profits': train_profits,
            'val_profits': val_profits,
            'exploration_rates': exploration_rates
        }
        
        pd.DataFrame(history).to_csv(os.path.join(run_dir, 'training_history.csv'), index=False)
        
        # สร้างกราฟแสดงประวัติการฝึกสอน
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.plot(train_rewards)
        plt.title('Training Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(2, 2, 2)
        plt.plot(train_profits)
        plt.title('Training Profits')
        plt.xlabel('Episode')
        plt.ylabel('Profit')
        
        plt.subplot(2, 2, 3)
        val_episodes = list(range(0, episodes, 10))
        plt.plot(val_episodes, val_rewards)
        plt.title('Validation Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(2, 2, 4)
        plt.plot(val_episodes, val_profits)
        plt.title('Validation Profits')
        plt.xlabel('Episode')
        plt.ylabel('Profit')
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, 'training_history.png'))
        plt.close()
        
        # สร้างกราฟแสดงอัตราการสำรวจ
        plt.figure(figsize=(10, 5))
        plt.plot(exploration_rates)
        plt.title('Exploration Rate (Epsilon)')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.savefig(os.path.join(run_dir, 'exploration_rate.png'))
        plt.close()
        
        logger.info(f"การฝึกสอนเสร็จสิ้น กำไรสูงสุดในการตรวจสอบ: {best_val_profit:.4f}")
        logger.info(f"ผลลัพธ์ถูกบันทึกไว้ที่: {run_dir}")
        
        return {
            'best_model_path': os.path.join(run_dir, 'best_model.h5'),
            'final_model_path': os.path.join(run_dir, 'final_model.h5'),
            'history': history,
            'best_validation_profit': best_val_profit
        }
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลังในการฝึกสอน: {e}")
        return None

def main():
    """
    ฟังก์ชันหลักสำหรับการรันสคริปต์
    """
    parser = argparse.ArgumentParser(description='ฝึกสอน Crypto Trading Bot ด้วย DQN')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='สัญลักษณ์คู่เหรียญ')
    parser.add_argument('--timeframe', type=str, default='1h', help='กรอบเวลา')
    parser.add_argument('--start_date', type=str, help='วันที่เริ่มต้น (เช่น 2020-01-01)')
    parser.add_argument('--end_date', type=str, help='วันที่สิ้นสุด (เช่น 2022-12-31)')
    parser.add_argument('--initial_balance', type=float, default=10000.0, help='เงินทุนเริ่มต้น')
    parser.add_argument('--window_size', type=int, default=10, help='ขนาดหน้าต่างข้อมูลย้อนหลัง')
    parser.add_argument('--batch_size', type=int, default=64, help='ขนาด batch สำหรับการฝึกสอน')
    parser.add_argument('--episodes', type=int, default=1000, help='จำนวนรอบการฝึกสอนทั้งหมด')
    parser.add_argument('--output_dir', type=str, default='outputs', help='โฟลเดอร์สำหรับบันทึกผลลัพธ์')
    
    args = parser.parse_args()
    
    train_dqn_agent(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_balance=args.initial_balance,
        window_size=args.window_size,
        batch_size=args.batch_size,
        episodes=args.episodes,
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()