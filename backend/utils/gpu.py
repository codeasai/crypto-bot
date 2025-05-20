import os
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("CUDA Environment Variables:")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"PATH contains CUDA: {'cuda' in os.environ.get('PATH', '').lower()}")

print("\nPhysical Devices:", tf.config.list_physical_devices())
print("GPU Devices:", tf.config.list_physical_devices('GPU'))
print("CPU Devices:", tf.config.list_physical_devices('CPU'))

# ตรวจสอบ TensorFlow build info
try:
    print("\nTensorFlow build information:")
    print(tf.sysconfig.get_build_info())
except:
    print("\nไม่สามารถแสดง TensorFlow build info ได้")