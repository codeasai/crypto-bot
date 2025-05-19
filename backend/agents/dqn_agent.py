"""
ตัวแทน AI (Agent) ที่ใช้อัลกอริทึม Deep Q-Network (DQN) สำหรับการเทรดสินทรัพย์คริปโต
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from collections import deque
import random
from typing import Tuple, List, Dict, Any, Optional
import os
import logging
from datetime import datetime

# ตั้งค่า logger
logger = logging.getLogger(__name__)

def setup_gpu():
    """
    ตรวจสอบและตั้งค่า GPU สำหรับ TensorFlow
    """
    try:
        # ตรวจสอบ GPU ที่มีอยู่
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            # เปิดใช้งาน memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # ตั้งค่า mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            logger.info(f"พบ GPU: {len(gpus)} เครื่อง")
            for gpu in gpus:
                logger.info(f"ชื่อ GPU: {gpu.name}")
            return True
        else:
            logger.info("ไม่พบ GPU จะใช้ CPU แทน")
            return False
            
    except Exception as e:
        logger.error(f"เกิดข้อผิดพลาดในการตั้งค่า GPU: {str(e)}")
        return False

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
        
        # ตรวจสอบและตั้งค่า GPU
        self.use_gpu = setup_gpu()
        
        # ปรับ batch size ตาม GPU
        if self.use_gpu:
            self.batch_size = min(self.batch_size * 2, 256)  # เพิ่ม batch size ถ้าใช้ GPU
            logger.info(f"ปรับ batch size เป็น {self.batch_size} สำหรับ GPU")
        
        # สร้างหน่วยความจำสำหรับ experience replay
        self.memory = deque(maxlen=memory_size)
        
        # สร้างโมเดลหลักและโมเดลเป้าหมาย
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        # ตัวแปรเพิ่มเติมสำหรับการติดตาม
        self.train_step_counter = 0
        self.update_target_every = 5  # อัพเดทโมเดลเป้าหมายทุกๆ 5 ขั้นตอนการฝึกสอน
        
        # ตัวแปรสำหรับการบันทึกประวัติ
        self.training_history = {
            'loss': [],
            'mae': [],
            'exploration_rate': []
        }
        
    def _build_model(self) -> keras.Model:
        """
        สร้างโครงข่ายประสาทเทียมสำหรับ DQN
        
        Returns:
            keras.Model: โมเดล Keras ที่สร้างขึ้น
        """
        try:
            # สร้าง input layer
            inputs = keras.Input(shape=(self.state_size,))
            
            # ชั้นซ่อน
            x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            
            x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
            x = layers.BatchNormalization()(x)
            
            # ชั้นข้อมูลออก
            outputs = layers.Dense(self.action_size, activation='linear')(x)
            
            # สร้างโมเดล
            model = keras.Model(inputs=inputs, outputs=outputs)
            
            # คอมไพล์โมเดล
            model.compile(
                loss='mse',
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                metrics=['mae'],
                jit_compile=True  # เปิดใช้งาน XLA compilation
            )
            
            return model
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการสร้างโมเดล: {str(e)}")
            raise
    
    def update_target_model(self):
        """
        อัพเดทน้ำหนักของโมเดลเป้าหมายให้ตรงกับโมเดลหลัก
        """
        try:
            self.target_model.set_weights(self.model.get_weights())
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอัพเดทโมเดลเป้าหมาย: {str(e)}")
    
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
        try:
            # ตรวจสอบค่าไม่ถูกต้อง
            if not np.isfinite(reward):
                reward = 0.0
                
            self.memory.append((state, action, reward, next_state, done))
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเก็บประสบการณ์: {str(e)}")
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        เลือกการกระทำตามนโยบาย epsilon-greedy
        
        Args:
            state (np.ndarray): สถานะปัจจุบัน
            training (bool): โหมดการฝึกสอนหรือไม่
            
        Returns:
            int: การกระทำที่เลือก
        """
        try:
            if training and np.random.rand() <= self.exploration_rate:
                # สำรวจ - เลือกการกระทำแบบสุ่ม
                return random.randrange(self.action_size)
            
            # ใช้ประโยชน์ - เลือกการกระทำที่ดีที่สุดตามโมเดล
            state = np.array(state, dtype=np.float32).reshape(1, -1)
            q_values = self.model.predict(state, verbose=0)[0]
            return np.argmax(q_values)
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการเลือกการกระทำ: {str(e)}")
            return random.randrange(self.action_size)
    
    def replay(self) -> float:
        """
        ฝึกสอนโมเดลด้วย experience replay
        
        Returns:
            float: ค่าความสูญเสีย (loss) จากการฝึกสอน
        """
        try:
            if len(self.memory) < self.batch_size:
                return 0
            
            # สุ่มตัวอย่างประสบการณ์จากหน่วยความจำ
            minibatch = random.sample(self.memory, self.batch_size)
            
            # แยกข้อมูลและเตรียมสำหรับการฝึกสอน
            states = np.array([experience[0] for experience in minibatch], dtype=np.float32)
            actions = np.array([experience[1] for experience in minibatch], dtype=np.int32)
            rewards = np.array([experience[2] for experience in minibatch], dtype=np.float32)
            next_states = np.array([experience[3] for experience in minibatch], dtype=np.float32)
            dones = np.array([experience[4] for experience in minibatch], dtype=np.float32)
            
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
            mae = history.history['mae'][0]
            
            # บันทึกประวัติการฝึกสอน
            self.training_history['loss'].append(loss)
            self.training_history['mae'].append(mae)
            self.training_history['exploration_rate'].append(self.exploration_rate)
            
            # อัพเดทอัตราการสำรวจ
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay
            
            # ตรวจสอบว่าควรอัพเดทโมเดลเป้าหมายหรือไม่
            self.train_step_counter += 1
            if self.train_step_counter % self.update_target_every == 0:
                self.update_target_model()
                
            return loss
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการฝึกสอน: {str(e)}")
            return 0.0
    
    def save(self, filepath: str):
        """
        บันทึกโมเดลลงในไฟล์
        
        Args:
            filepath (str): ตำแหน่งที่ต้องการบันทึกโมเดล
        """
        try:
            # สร้างโฟลเดอร์ถ้ายังไม่มี
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # บันทึกโมเดลในรูปแบบ .keras
            model_path = os.path.splitext(filepath)[0] + '.keras'
            self.model.save(model_path)
            
            # บันทึกประวัติการฝึกสอน
            history_path = os.path.splitext(filepath)[0] + '_history.npz'
            np.savez(history_path, **self.training_history)
            
            logger.info(f"บันทึกโมเดลและประวัติการฝึกสอนที่ {filepath}")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการบันทึกโมเดล: {str(e)}")
    
    def load(self, filepath: str):
        """
        โหลดโมเดลจากไฟล์
        
        Args:
            filepath (str): ตำแหน่งไฟล์โมเดล
        """
        try:
            # ลองโหลดจาก .keras ก่อน
            model_path = os.path.splitext(filepath)[0] + '.keras'
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
            else:
                # ถ้าไม่มี .keras ให้ลองโหลดจาก .h5
                self.model = keras.models.load_model(filepath)
                
            self.update_target_model()
            
            # โหลดประวัติการฝึกสอน
            history_path = os.path.splitext(filepath)[0] + '_history.npz'
            if os.path.exists(history_path):
                history = np.load(history_path)
                self.training_history = {key: history[key].tolist() for key in history.files}
                
            logger.info(f"โหลดโมเดลและประวัติการฝึกสอนจาก {filepath}")
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}")
            raise