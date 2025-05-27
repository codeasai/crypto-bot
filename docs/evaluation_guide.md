# คู่มือการประเมินผลโมเดล (Model Evaluation Guide)

## 1. การเริ่มต้นใช้งาน

### 1.1 ผ่านหน้า UI (Streamlit)
```bash
# รัน Streamlit dashboard
streamlit run dashboard/app.py
```
1. เลือกเมนู "Model Evaluation" จาก sidebar
2. เลือกโมเดลที่ต้องการประเมินผล
3. ระบบจะแสดงผลการประเมินอัตโนมัติ

### 1.2 ผ่าน Command Line
```bash
# รัน evaluation script
python scripts/evaluate_agent.py
```

## 2. โครงสร้างไฟล์ที่จำเป็น

```
project/
├── models/
│   └── saved_models/
│       └── dqn_btcusdt_1h_20250522_143052/
│           └── best_model.keras
├── data/
│   └── datasets/
│       └── train_test_split/
│           └── validation_data.csv
└── config/
    └── config.py
```

## 3. ขั้นตอนการประเมินผล

### 3.1 การประเมินผลพื้นฐาน
```python
from scripts.evaluate_agent import EvaluationPipeline

# สร้าง pipeline
pipeline = EvaluationPipeline(
    model_path="models/saved_models/dqn_btcusdt_1h_20250522_143052/best_model.keras",
    validation_data_path="data/datasets/train_test_split/validation_data.csv",
    output_dir="logs/evaluation"
)

# รันการประเมินผล
pipeline.run_evaluation()
```

### 3.2 การประเมินผลแบบละเอียด
```python
from core.evaluator import ModelEvaluator
from evaluation.validation import ValidationMetrics
from evaluation.hyperparameter_optimizer import HyperparameterOptimizer
from evaluation.feature_analyzer import FeatureAnalyzer

# โหลดโมเดลและข้อมูล
evaluator = ModelEvaluator("path/to/model.keras")
validation_data = pd.read_csv("path/to/validation_data.csv")

# คำนวณ metrics
metrics = ValidationMetrics(evaluator, validation_data)
results = metrics.calculate_metrics()

# Optimize hyperparameters
optimizer = HyperparameterOptimizer(evaluator, validation_data)
best_params = optimizer.optimize()

# วิเคราะห์ features
analyzer = FeatureAnalyzer(evaluator, validation_data)
feature_importance = analyzer.analyze_features()
```

## 4. ผลลัพธ์ที่ได้

### 4.1 Validation Metrics
```json
{
  "validation_metrics": {
    "total_return": 12.89,
    "sharpe_ratio": 1.67,
    "max_drawdown": -8.45,
    "win_rate": 0.623,
    "profit_factor": 1.89,
    "avg_trade_profit": 0.234,
    "total_trades": 156,
    "winning_trades": 97,
    "losing_trades": 59
  }
}
```

### 4.2 Hyperparameter Optimization
```json
{
  "best_hyperparameters": {
    "learning_rate": 0.0005,
    "gamma": 0.99,
    "hidden_size": 256,
    "batch_size": 128
  }
}
```

### 4.3 Feature Importance
```json
{
  "feature_importance": {
    "price": 0.45,
    "volume": 0.25,
    "rsi": 0.15,
    "macd": 0.10,
    "bollinger_bands": 0.05
  }
}
```

## 5. การตีความผลลัพธ์

### 5.1 Performance Metrics
- **Total Return**: ผลตอบแทนรวม
- **Sharpe Ratio**: อัตราส่วนความเสี่ยงต่อผลตอบแทน
- **Max Drawdown**: การขาดทุนสูงสุด
- **Win Rate**: อัตราการเทรดที่ได้กำไร

### 5.2 Risk Metrics
- **Volatility**: ความผันผวนของผลตอบแทน
- **Sortino Ratio**: อัตราส่วนความเสี่ยงด้านลบต่อผลตอบแทน
- **Calmar Ratio**: อัตราส่วนผลตอบแทนต่อการขาดทุนสูงสุด
- **VaR 95%**: มูลค่าความเสี่ยงที่ระดับความเชื่อมั่น 95%

### 5.3 Feature Importance
- **Permutation Importance**: ความสำคัญของแต่ละ feature
- **SHAP Values**: ผลกระทบของแต่ละ feature ต่อการทำนาย
- **Ablation Study**: ผลกระทบเมื่อลบ feature ออก

## 6. การปรับปรุงโมเดล

### 6.1 จาก Hyperparameter Optimization
1. ใช้ parameter ที่ได้จากการ optimize
2. เทรนโมเดลใหม่ด้วย parameter เหล่านั้น
3. ประเมินผลซ้ำ

### 6.2 จาก Feature Analysis
1. ลบ features ที่มีความสำคัญต่ำ
2. เพิ่ม features ที่มีความสำคัญสูง
3. ปรับปรุงการคำนวณ features

## 7. ข้อควรระวัง

1. ตรวจสอบความถูกต้องของข้อมูล validation
2. ใช้เวลาเพียงพอในการ optimize hyperparameters
3. ตรวจสอบ overfitting จาก validation metrics
4. เก็บ log การทดลองทุกครั้ง

## 8. การแก้ไขปัญหา

### 8.1 Error: Model not found
```python
# ตรวจสอบ path ของโมเดล
model_path = "models/saved_models/your_model.keras"
assert os.path.exists(model_path), "Model file not found"
```

### 8.2 Error: Invalid validation data
```python
# ตรวจสอบข้อมูล validation
validation_data = pd.read_csv("path/to/validation_data.csv")
required_columns = ["close", "volume", "rsi", "macd"]
assert all(col in validation_data.columns for col in required_columns)
```

### 8.3 Error: Memory issues
```python
# ลดขนาด batch size
param_grid = {
    "batch_size": [16, 32, 64]  # ลดจาก [32, 64, 128]
}
```

## 9. การอัพเดท

1. ตรวจสอบการอัพเดทของ dependencies
2. อัพเดท config ตามความเหมาะสม
3. ทดสอบการทำงานหลังอัพเดท

## 10. การติดต่อและรายงานปัญหา

- สร้าง issue ใน GitHub repository
- อธิบายปัญหาและ steps to reproduce
- แนบ log files และ error messages 