"""
base_model.py - Abstract Base Class cho OpenIE Models

Định nghĩa interface chung cho tất cả Open Information Extraction models.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseOPENIEModel(ABC):
    """
    Abstract base class cho OpenIE models.
    
    Định nghĩa phương thức bắt buộc:
    - __call__(): Trích xuất entities và triples từ text
    """
    
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Khởi tạo OpenIE model với các tham số tùy chỉnh."""
        pass

    @abstractmethod
    def __call__(self, text: str) -> dict:
        """
        Thực hiện Open Information Extraction trên text.
        
        Args:
            text: Input text cần extract
                 Ví dụ: "Emmanuel Macron is the president of France"
        
        Returns:
            dict với 3 fields:
            {
                "passage": str,              # Input text gốc
                "extracted_entities": list,  # Entities đã extract
                "extracted_triples": list    # Triples (subject, relation, object)
            }
            
            Ví dụ output:
            {
                "passage": "Emmanuel Macron is the president of France",
                "extracted_entities": ["Emmanuel Macron", "France"],
                "extracted_triples": [
                    ["Emmanuel Macron", "president of", "France"]
                ]
            }
        """
        pass