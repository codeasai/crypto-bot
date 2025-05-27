import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple

class ModelEvaluator:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        
    def predict(self, state: np.ndarray) -> np.ndarray:
        """ทำนาย action จาก state"""
        return self.model.predict(state, verbose=0)
    
    def evaluate_episode(self, states: np.ndarray, actions: np.ndarray, 
                        rewards: np.ndarray) -> Dict[str, float]:
        """ประเมินผลการทำนายในแต่ละ episode"""
        predictions = self.predict(states)
        predicted_actions = np.argmax(predictions, axis=1)
        
        accuracy = np.mean(predicted_actions == actions)
        total_reward = np.sum(rewards)
        
        return {
            "accuracy": float(accuracy),
            "total_reward": float(total_reward)
        }
    
    def get_feature_importance(self, states: np.ndarray, 
                             n_permutations: int = 100) -> np.ndarray:
        """คำนวณ feature importance โดยใช้ permutation importance"""
        baseline_score = self._calculate_baseline_score(states)
        importance_scores = []
        
        for feature_idx in range(states.shape[1]):
            feature_scores = []
            for _ in range(n_permutations):
                permuted_states = states.copy()
                np.random.shuffle(permuted_states[:, feature_idx])
                score = self._calculate_baseline_score(permuted_states)
                feature_scores.append(baseline_score - score)
            
            importance_scores.append(np.mean(feature_scores))
            
        return np.array(importance_scores)
    
    def _calculate_baseline_score(self, states: np.ndarray) -> float:
        """คำนวณ baseline score สำหรับ feature importance"""
        predictions = self.predict(states)
        return float(np.mean(np.max(predictions, axis=1))) 