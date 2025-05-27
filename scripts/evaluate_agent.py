import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from core.evaluator import ModelEvaluator
from evaluation.validation import ValidationMetrics
from evaluation.hyperparameter_optimizer import HyperparameterOptimizer
from evaluation.feature_analyzer import FeatureAnalyzer
from config.config import EVAL_CONFIG

class EvaluationPipeline:
    def __init__(self, model_path: str, validation_data_path: str, output_dir: str):
        self.model_path = model_path
        self.validation_data_path = validation_data_path
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_evaluation(self):
        # สร้างโฟลเดอร์สำหรับผลลัพธ์
        results_dir = Path(self.output_dir) / f"evaluation_{self.timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # โหลดข้อมูล validation
        validation_data = pd.read_csv(self.validation_data_path)
        
        # ประเมินโมเดล
        evaluator = ModelEvaluator(self.model_path)
        validation_metrics = ValidationMetrics(evaluator, validation_data)
        metrics = validation_metrics.calculate_metrics()
        
        # บันทึกผลลัพธ์
        self._save_metrics(metrics, results_dir)
        
        # ทำ hyperparameter optimization
        optimizer = HyperparameterOptimizer(evaluator, validation_data)
        best_params = optimizer.optimize()
        self._save_optimization_results(best_params, results_dir)
        
        # วิเคราะห์ feature importance
        analyzer = FeatureAnalyzer(evaluator, validation_data)
        feature_importance = analyzer.analyze_features()
        self._save_feature_analysis(feature_importance, results_dir)
        
    def _save_metrics(self, metrics: dict, results_dir: Path):
        metrics_file = results_dir / "validation_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
            
    def _save_optimization_results(self, results: dict, results_dir: Path):
        params_file = results_dir / "best_params.json"
        with open(params_file, "w") as f:
            json.dump(results, f, indent=2)
            
    def _save_feature_analysis(self, analysis: dict, results_dir: Path):
        analysis_file = results_dir / "feature_importance_scores.json"
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2)

if __name__ == "__main__":
    model_path = "models/saved_models/dqn_btcusdt_1h_20250522_143052/best_model.keras"
    validation_data_path = "data/datasets/train_test_split/validation_data.csv"
    output_dir = "logs/evaluation"
    
    pipeline = EvaluationPipeline(model_path, validation_data_path, output_dir)
    pipeline.run_evaluation() 