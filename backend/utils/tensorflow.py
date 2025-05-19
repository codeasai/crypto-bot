import tensorflow as tf
import numpy as np
from utils.gpu_checker import GPUChecker

def check_tensorflow():
    try:
        # ตรวจสอบ TensorFlow version
        print(f"TensorFlow version: {tf.__version__}")
        
        # ตรวจสอบ GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"พบ GPU: {len(gpus)} เครื่อง")
            for gpu in gpus:
                print(f"ชื่อ GPU: {gpu.name}")
                
            # ตั้งค่า memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            # ตั้งค่า mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("เปิดใช้งาน mixed precision")
            
            return True
        else:
            print("ไม่พบ GPU")
            return False
    except Exception as e:
        print(f"เกิดข้อผิดพลาด: {str(e)}")
        return False

def test_tensorflow():
    try:
        # ทดสอบการคำนวณ
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print("ทดสอบ TensorFlow สำเร็จ")
        return True
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทดสอบ: {str(e)}")
        return False

if __name__ == "__main__":
    # ตรวจสอบ GPU
    if GPUChecker.check_gpu():
        # ทดสอบ TensorFlow
        test_tensorflow()
        # แสดงข้อมูล GPU
        GPUChecker.print_gpu_info()
    else:
        print("ไม่สามารถใช้งาน GPU ได้")
        