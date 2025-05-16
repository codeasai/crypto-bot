"""
ตัวแทน AI (Agent) ที่ใช้อัลกอริทึม Deep Q-Network (DQN) สำหรับการเทรดสินทรัพย์คริปโต
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
from typing import Tuple, List, Dict, Any


class DQNAgent:
    """
    ตัวแทน AI (Agent) ที่ใช้อัลกอริทึม Deep Q-Network (DQN)
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.95,
                 exploration_rate: float = 1.0,
                 exploration_decay: float = 0.995,
                 exploration_min: float = 0.01,
                 batch_size: int = 64,
                 memory_size: int = 10000):
        """
        กำหนดค่าเริ่มต้นของตัวแทน DQN
        
        Args:
            state_size (int): ขนาดของ state vector
            action_size (int): จำนวนการกระทำที่เป็นไปได้
            learning_rate (float): อัตราการเรียนรู้
            discount_factor (float): ค่า discount factor สำหรับผลตอบแทนในอนาคต
            exploration_rate (float): อัตราการสำรวจเริ่มต้น (epsilon)
            exploration_decay (float): อัตราการลดลงของการสำรวจ
            exploration_min (float): ค่าต่ำสุดของอัตราการสำรวจ
            batch_size (int): ขนาด batch สำหรับการฝึกสอน
            memory_size (int): ขนาดของหน่วยความจำสำหรับ experience replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.batch_size = batch_size
        
        # สร้างหน่วยความจำสำหรับ experience replay
        self.memory = deque(maxlen=memory_size)
        
        # สร้างโมเดลหลักและโมเดลเป้าหมาย
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # ตัวแปรเพิ่มเติมสำหรับการติดตาม
        self.train_step_counter = 0
        self.update_target_every = 5  # อัพเดทโมเดลเป้าหมายทุกๆ 5 ขั้นตอนการฝึกสอน
        
    def _build_model(self) -> keras.Model:
        """
        สร้างโครงข่ายประสาทเทียมสำหรับ DQN
        
        Returns:
            keras.Model: โมเดล Keras ที่สร้างขึ้น
        """
        model = keras.Sequential()
        
        # ชั้นข้อมูลเข้า
        model.add(layers.Dense(128, input_dim=self.state_size, activation='relu'))
        
        # ชั้นซ่อน
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        
        # ชั้นข้อมูลออก - Q values สำหรับแต่ละการกระทำ
        model.add(layers.Dense(self.action_size, activation='linear'))
        
        # คอมไพล์โมเดล
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=self.learning_rate))
        
        return model
    
    def update_target_model(self):
        """
        อัพเดทน้ำหนักของโมเดลเป้าหมายให้ตรงกับโมเดลหลัก
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        เก็บประสบการณ์ลงในหน่วยความจำสำหรับ experience replay
        
        Args:
            state (np.ndarray): สถานะปัจจุบัน
            action (int): การกระทำที่เลือก
            reward (float): รางวัลที่ได้รับ
            next_state (np.ndarray): สถานะถัดไป
            done (bool): สถานะการจบ episode
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        เลือกการกระทำตามนโยบาย epsilon-greedy
        
        Args:
            state (np.ndarray): สถานะปัจจุบัน
            training (bool): โหมดการฝึกสอนหรือไม่
            
        Returns:
            int: การกระทำที่เลือก
        """
        if training and np.random.rand() <= self.exploration_rate:
            # สำรวจ - เลือกการกระทำแบบสุ่ม
            return random.randrange(self.action_size)
        
        # ใช้ประโยชน์ - เลือกการกระทำที่ดีที่สุดตามโมเดล
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values)
    
    def replay(self) -> float:
        """
        ฝึกสอนโมเดลด้วย experience replay
        
        Returns:
            float: ค่าความสูญเสีย (loss) จากการฝึกสอน
        """
        if len(self.memory) < self.batch_size:
            return 0
        
        # สุ่มตัวอย่างประสบการณ์จากหน่วยความจำ
        minibatch = random.sample(self.memory, self.batch_size)
        
        # แยกข้อมูลและเตรียมสำหรับการฝึกสอน
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # ใช้โมเดลเป้าหมายในการคำนวณ Q values ของสถานะถัดไป
        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_target_q_values = np.max(target_q_values, axis=1)
        
        # คำนวณ target Q values
        targets = rewards + (1 - dones) * self.discount_factor * max_target_q_values
        
        # ใช้โมเดลหลักในการทำนาย Q values ของสถานะปัจจุบัน
        current_q_values = self.model.predict(states, verbose=0)
        
        # อัพเดท Q values เฉพาะการกระทำที่เลือก
        for i in range(self.batch_size):
            current_q_values[i, actions[i]] = targets[i]
        
        # ฝึกสอนโมเดล
        history = self.model.fit(states, current_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        
        # อัพเดทอัตราการสำรวจ
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
        
        # ตรวจสอบว่าควรอัพเดทโมเดลเป้าหมายหรือไม่
        self.train_step_counter += 1
        if self.train_step_counter % self.update_target_every == 0:
            self.update_target_model()
            
        return loss
    
    def save(self, filepath: str):
        """
        บันทึกโมเดลลงในไฟล์
        
        Args:
            filepath (str): ตำแหน่งที่ต้องการบันทึกโมเดล
        """
        self.model.save(filepath)
    
    def load(self, filepath: str):
        """
        โหลดโมเดลจากไฟล์
        
        Args:
            filepath (str): ตำแหน่งไฟล์โมเดล
        """
        self.model = keras.models.load_model(filepath)
        self.update_target_model()