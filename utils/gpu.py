import os
import tensorflow as tf

def check_gpu():
    """ตรวจสอบและแสดงข้อมูล GPU และ TensorFlow"""
    # แสดงเวอร์ชัน TensorFlow
    print(f"TensorFlow version: {tf.__version__}")
    
    # ตรวจสอบ CUDA Environment
    print("\nCUDA Environment:")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"CUDA in PATH: {'cuda' in os.environ.get('PATH', '').lower()}")
    
    # ตรวจสอบอุปกรณ์
    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices('GPU')
    cpus = tf.config.list_physical_devices('CPU')
    
    print("\nอุปกรณ์ที่พบ:")
    print(f"GPU: {len(gpus)} ตัว")
    for gpu in gpus:
        print(f"  - {gpu}")
    print(f"CPU: {len(cpus)} ตัว")
    
    # ตรวจสอบ TensorFlow build info
    try:
        build_info = tf.sysconfig.get_build_info()
        print("\nTensorFlow Build Info:")
        print(f"CUDA Version: {build_info.get('cuda_version', 'N/A')}")
        print(f"cuDNN Version: {build_info.get('cudnn_version', 'N/A')}")
        print(f"CUDA Build: {build_info.get('is_cuda_build', False)}")
    except:
        print("\nไม่สามารถแสดง TensorFlow build info ได้")

if __name__ == "__main__":
    check_gpu()