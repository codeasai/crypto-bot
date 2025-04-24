from .base_strategy import BaseStrategy
# ไม่จำเป็นต้อง import strategy อื่นๆ ที่นี่ ถ้าจะโหลดแบบ dynamic
# from .ema_crossover import Strategy as EMACrossoverStrategy
# from .moving_average import Strategy as MovingAverageStrategy

# ---- ลบ get_strategy ถ้าไม่ใช้แล้ว หรือปรับปรุง ----
# def get_strategy(strategy_name: str) -> BaseStrategy:
#     ...

import os
import importlib
import inspect # เพิ่ม inspect
from typing import List, Dict, Any # เพิ่ม typing

def get_strategy(strategy_name: str, params: Dict[str, Any] = None) -> BaseStrategy:
    """
    สร้าง instance ของกลยุทธ์ตามชื่อที่ระบุ และส่งพารามิเตอร์ให้
    """
    try:
        module_name = f".{strategy_name}" # ชื่อไฟล์กลยุทธ์
        module = importlib.import_module(module_name, package=__name__)

        # หาคลาสที่สืบทอดจาก BaseStrategy ใน module นั้น
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                # สร้าง instance ของ strategy โดยส่ง params ไปเป็น config
                return obj(config=params)

        # ถ้าไม่พบคลาสที่เหมาะสมในไฟล์
        raise ImportError(f"Could not find a valid Strategy class in {module_name}")

    except ImportError as e:
        raise ValueError(f"Strategy '{strategy_name}' not found or failed to load: {e}")
    except Exception as e:
        # จัดการ error อื่นๆ ที่อาจเกิดขึ้นตอนสร้าง instance
        raise ValueError(f"Error initializing strategy '{strategy_name}': {e}")

def list_available_strategies() -> List[Dict[str, Any]]:
    """
    แสดงรายการกลยุทธ์ที่มีทั้งหมด พร้อมพารามิเตอร์
    """
    strategies_list = []
    current_dir = os.path.dirname(__file__)

    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and not filename.startswith('__') and filename != 'base_strategy.py':
            module_name = filename[:-3]
            try:
                # โหลด module
                module = importlib.import_module(f'.{module_name}', package=__name__)

                # หาคลาสที่สืบทอดจาก BaseStrategy ใน module นั้น
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                        try:
                            # สร้าง instance ของ strategy
                            strategy_instance = obj()
                            # ดึงข้อมูลที่ต้องการ
                            strategy_info = {
                                # ใช้ id เป็นชื่อ module (หรือชื่อไฟล์) เพื่อความง่าย
                                'id': module_name,
                                'name': strategy_instance.get_name(),
                                'description': strategy_instance.get_description(),
                                'parameters': strategy_instance.get_parameters() # เรียก get_parameters()
                            }
                            strategies_list.append(strategy_info)
                            # เจอคลาสแรกแล้ว หยุดหาในไฟล์นี้
                            break
                        except Exception as e:
                            print(f"Error instantiating or getting info from {name} in {filename}: {e}")
                        break # ออกจาก loop หลังจากเจอ class แรก

            except Exception as e:
                print(f"ไม่สามารถโหลดกลยุทธ์ {module_name} จาก {filename}: {str(e)}")

    return strategies_list 