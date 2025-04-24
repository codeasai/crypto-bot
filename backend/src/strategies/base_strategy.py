from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseStrategy(ABC):
    """
    คลาสพื้นฐานสำหรับกลยุทธ์การเทรด
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        วิเคราะห์ข้อมูลและตัดสินใจซื้อขาย
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        รับชื่อกลยุทธ์
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        รับคำอธิบายกลยุทธ์
        """
        pass 