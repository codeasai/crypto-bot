"""
ตรวจสอบและแสดงข้อมูล GPU สำหรับการเทรด
"""

import tensorflow as tf
import logging
import os
import platform
import psutil
from typing import Dict, Any, Optional
import subprocess
import re

# ตั้งค่า logger
logger = logging.getLogger(__name__)

class GPUChecker:
    """
    คลาสสำหรับตรวจสอบและแสดงข้อมูล GPU
    """
    
    @staticmethod
    def get_cuda_info() -> Dict[str, Any]:
        """
        รับข้อมูล CUDA
        
        Returns:
            Dict[str, Any]: ข้อมูล CUDA
        """
        try:
            cuda_info = {
                'available': False,
                'version': None,
                'driver_version': None,
                'devices': []
            }
            
            # ตรวจสอบ CUDA version จาก nvidia-smi
            try:
                nvidia_smi = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
                nvidia_smi_str = nvidia_smi.decode('utf-8')
                
                # หา CUDA version
                cuda_match = re.search(r'CUDA Version: (\d+\.\d+)', nvidia_smi_str)
                if cuda_match:
                    cuda_info['version'] = cuda_match.group(1)
                
                # หา driver version
                driver_match = re.search(r'Driver Version: (\d+\.\d+)', nvidia_smi_str)
                if driver_match:
                    cuda_info['driver_version'] = driver_match.group(1)
                
                # หาข้อมูล GPU
                gpu_matches = re.finditer(r'(\d+)\s+(\w+)\s+(\w+)\s+(\d+)MiB', nvidia_smi_str)
                for match in gpu_matches:
                    gpu_info = {
                        'id': match.group(1),
                        'name': match.group(2),
                        'memory': int(match.group(4))
                    }
                    cuda_info['devices'].append(gpu_info)
                
                cuda_info['available'] = True
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("ไม่พบ nvidia-smi หรือไม่สามารถรันได้")
            
            return cuda_info
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอ่านข้อมูล CUDA: {str(e)}")
            return {'available': False, 'error': str(e)}
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """
        รับข้อมูลระบบ
        
        Returns:
            Dict[str, Any]: ข้อมูลระบบ
        """
        try:
            return {
                'os': platform.system(),
                'os_version': platform.version(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'total_memory': psutil.virtual_memory().total / (1024**3),  # GB
                'available_memory': psutil.virtual_memory().available / (1024**3)  # GB
            }
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอ่านข้อมูลระบบ: {str(e)}")
            return {}
    
    @staticmethod
    def check_gpu() -> Dict[str, Any]:
        """
        ตรวจสอบและแสดงข้อมูล GPU ที่มีอยู่
        
        Returns:
            Dict[str, Any]: ข้อมูล GPU
        """
        try:
            # ตั้งค่า logging
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            tf.get_logger().setLevel('ERROR')
            
            # ตรวจสอบ CUDA
            cuda_info = GPUChecker.get_cuda_info()
            
            # ตรวจสอบ GPU
            gpus = tf.config.list_physical_devices('GPU')
            
            if not gpus:
                logger.info("ไม่พบ GPU จะใช้ CPU แทน")
                return {
                    'available': False,
                    'count': 0,
                    'devices': [],
                    'memory_growth': False,
                    'mixed_precision': False,
                    'tensorflow_version': tf.__version__,
                    'system_info': GPUChecker.get_system_info(),
                    'cuda_info': cuda_info
                }
            
            # ตั้งค่า memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # ตั้งค่า mixed precision
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # เก็บข้อมูล GPU
            gpu_info = {
                'available': True,
                'count': len(gpus),
                'devices': [gpu.name for gpu in gpus],
                'memory_growth': True,
                'mixed_precision': True,
                'tensorflow_version': tf.__version__,
                'system_info': GPUChecker.get_system_info(),
                'cuda_info': cuda_info
            }
            
            # แสดงข้อมูล
            logger.info(f"พบ GPU: {len(gpus)} เครื่อง")
            for gpu in gpus:
                logger.info(f"ชื่อ GPU: {gpu.name}")
            logger.info("เปิดใช้งาน memory growth")
            logger.info("เปิดใช้งาน mixed precision")
            
            if cuda_info['available']:
                logger.info(f"CUDA Version: {cuda_info['version']}")
                logger.info(f"Driver Version: {cuda_info['driver_version']}")
                for device in cuda_info['devices']:
                    logger.info(f"GPU {device['id']}: {device['name']} ({device['memory']}MiB)")
            
            return gpu_info
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการตรวจสอบ GPU: {str(e)}")
            return {
                'available': False,
                'count': 0,
                'devices': [],
                'memory_growth': False,
                'mixed_precision': False,
                'error': str(e),
                'tensorflow_version': tf.__version__,
                'system_info': GPUChecker.get_system_info(),
                'cuda_info': cuda_info
            }
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, Any]:
        """
        แสดงข้อมูลหน่วยความจำของ GPU
        
        Returns:
            Dict[str, Any]: ข้อมูลหน่วยความจำ GPU
        """
        try:
            if not tf.config.list_physical_devices('GPU'):
                return {'error': 'ไม่พบ GPU'}
                
            memory_info = {}
            for gpu in tf.config.list_physical_devices('GPU'):
                try:
                    # แก้ไขการอ่านข้อมูลหน่วยความจำ
                    device_name = gpu.name.split('/')[-1]  # เอาเฉพาะส่วน GPU:0
                    info = tf.config.experimental.get_memory_info(device_name)
                    memory_info[device_name] = {
                        'current': info['current'] / 1024**2,  # MB
                        'peak': info['peak'] / 1024**2,  # MB
                        'total': info['total'] / 1024**2  # MB
                    }
                except Exception as e:
                    memory_info[gpu.name] = {'error': f'ไม่สามารถอ่านข้อมูลหน่วยความจำ: {str(e)}'}
                    
            return memory_info
            
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการอ่านข้อมูลหน่วยความจำ GPU: {str(e)}")
            return {'error': str(e)}
    
    @staticmethod
    def print_gpu_info() -> None:
        """
        แสดงข้อมูล GPU ทั้งหมด
        """
        try:
            # ตรวจสอบ GPU
            gpu_info = GPUChecker.check_gpu()
            
            print("\n=== ข้อมูลระบบ ===")
            system_info = gpu_info.get('system_info', {})
            print(f"ระบบปฏิบัติการ: {system_info.get('os', 'N/A')} {system_info.get('os_version', '')}")
            print(f"Python: {system_info.get('python_version', 'N/A')}")
            print(f"CPU: {system_info.get('cpu_count', 'N/A')} cores")
            print(f"หน่วยความจำทั้งหมด: {system_info.get('total_memory', 0):.2f} GB")
            print(f"หน่วยความจำที่เหลือ: {system_info.get('available_memory', 0):.2f} GB")
            
            print("\n=== ข้อมูล TensorFlow ===")
            print(f"เวอร์ชัน: {gpu_info.get('tensorflow_version', 'N/A')}")
            
            print("\n=== ข้อมูล CUDA ===")
            cuda_info = gpu_info.get('cuda_info', {})
            if cuda_info.get('available'):
                print(f"CUDA Version: {cuda_info.get('version', 'N/A')}")
                print(f"Driver Version: {cuda_info.get('driver_version', 'N/A')}")
                print("\nรายชื่อ GPU:")
                for device in cuda_info.get('devices', []):
                    print(f"- GPU {device['id']}: {device['name']} ({device['memory']}MiB)")
            else:
                print("ไม่พบ CUDA หรือไม่สามารถเข้าถึงได้")
            
            if not gpu_info['available']:
                print("\n=== ข้อมูล GPU ===")
                print("ไม่พบ GPU จะใช้ CPU แทน")
                if 'error' in gpu_info:
                    print(f"ข้อผิดพลาด: {gpu_info['error']}")
                return
            
            print("\n=== ข้อมูล GPU ===")
            print(f"จำนวน GPU: {gpu_info['count']}")
            print("\nรายชื่อ GPU:")
            for device in gpu_info['devices']:
                print(f"- {device}")
            
            print("\nข้อมูลหน่วยความจำ:")
            memory_info = GPUChecker.get_gpu_memory_info()
            for device, info in memory_info.items():
                if 'error' in info:
                    print(f"{device}: {info['error']}")
                else:
                    print(f"{device}:")
                    print(f"  - หน่วยความจำที่ใช้: {info['current']:.2f} MB")
                    print(f"  - หน่วยความจำสูงสุด: {info['peak']:.2f} MB")
                    print(f"  - หน่วยความจำทั้งหมด: {info['total']:.2f} MB")
            
            print("\nการตั้งค่า:")
            print(f"- Memory Growth: {'เปิดใช้งาน' if gpu_info['memory_growth'] else 'ปิดใช้งาน'}")
            print(f"- Mixed Precision: {'เปิดใช้งาน' if gpu_info['mixed_precision'] else 'ปิดใช้งาน'}")
            
        except Exception as e:
            print(f"\nเกิดข้อผิดพลาดในการแสดงข้อมูล GPU: {str(e)}")

if __name__ == '__main__':
    # ตั้งค่า logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # แสดงข้อมูล GPU
    GPUChecker.print_gpu_info() 