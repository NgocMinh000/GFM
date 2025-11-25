"""
base_model.py - Abstract Base Class cho Entity Linking Models

Định nghĩa interface chung cho tất cả Entity Linking models.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseELModel(ABC):
    """
    Abstract base class cho Entity Linking models.
    
    Định nghĩa 2 phương thức bắt buộc:
    - index(): Tạo index từ danh sách entities
    - __call__(): Link entities từ NER với KB entities
    """
    
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Khởi tạo model với các tham số tùy chỉnh."""
        pass

    @abstractmethod
    def index(self, entity_list: list) -> None:
        """
        Tạo index từ danh sách entities để search nhanh.
        
        Args:
            entity_list: Danh sách entities cần index
                        Ví dụ: ["Paris", "France", "Eiffel Tower"]
        
        Returns:
            None (tạo internal index)
        
        Raises:
            ValueError: Nếu entity_list rỗng hoặc format không hợp lệ
            TypeError: Nếu entity_list không phải list
        """
        pass

    @abstractmethod
    def __call__(self, ner_entity_list: list, topk: int = 1) -> dict:
        """
        Link NER entities với KB entities.
        
        Args:
            ner_entity_list: Danh sách entities từ NER
                            Ví dụ: ["paris city", "france country"]
            topk: Số lượng candidates trả về cho mỗi entity
        
        Returns:
            dict: Mapping entity → candidates
            
            Format:
            {
                "paris city": [
                    {
                        "entity": "Paris",        # Entity trong KB
                        "score": 0.95,            # Raw similarity score
                        "norm_score": 1.0         # Normalized score (0-1)
                    },
                    {
                        "entity": "Paris, Texas",
                        "score": 0.45,
                        "norm_score": 0.47
                    }
                ]
            }
        """
        pass