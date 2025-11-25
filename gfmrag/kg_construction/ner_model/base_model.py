"""
base_model.py - Abstract Base Class cho NER Models

Định nghĩa interface chung cho tất cả Named Entity Recognition models.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseNERModel(ABC):
    """
    Abstract base class cho NER models.
    
    Định nghĩa phương thức bắt buộc:
    - __call__(): Trích xuất named entities từ text
    """
    
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Khởi tạo NER model với các tham số tùy chỉnh."""
        pass

    @abstractmethod
    def __call__(self, text: str) -> list:
        """
        Trích xuất named entities từ text.
        
        Args:
            text: Input text cần extract entities
                 Ví dụ: "Bill Gates founded Microsoft in 1975"
        
        Returns:
            list: Danh sách entities đã extract và clean
                 Ví dụ: ["bill gates", "microsoft", "1975"]
        
        Note:
            - Output đã được processed (lowercase, cleaned)
            - Abstract method - subclass phải implement
        """
        pass