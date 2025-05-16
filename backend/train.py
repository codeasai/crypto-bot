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
    # สร้างโฟลเดอร์ output ถ้ายังไม่มี
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # สร้างโฟลเดอร์ย่อยสำหรับผลลัพธ์การฝึกสอนนี้
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{symbol}_{timeframe}_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir)
    
    # โหลดและเตรียมข้อมูล
    print(f"กำลังโหลดและเตรียมข้อมูล {symbol} ที่กรอบเวลา {timeframe}...")
    data_processor = DataProcessor()
    train_data, val_data, _ = data_processor.prepare_data_for_training(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    print(f"ข้อมูลฝึกสอน: {len(train_data)} แถว, ข้อมูลตรวจสอบ: {len(val_data)} แถว")
    
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
    # เราจะแบ่งการกระทำเป็น 9 ประเภท:
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
    print(f"เริ่มการฝึกสอนตัวแทน DQN สำหรับ {episodes} รอบ...")
    
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
            print(f"Episode: {episode+1}/{episodes}, "
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
    
    print(f"การฝึกสอนเสร็จสิ้น กำไรสูงสุดในการตรวจสอบ: {best_val_profit:.4f}")
    print(f"ผลลัพธ์ถูกบันทึกไว้ที่: {run_dir}")
    
    return {
        'best_model_path': os.path.join(run_dir, 'best_model.h5'),
        'final_model_path': os.path.join(run_dir, 'final_model.h5'),
        'history': history,
        'best_validation_profit': best_val_profit
    }


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