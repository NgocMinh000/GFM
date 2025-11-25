"""
utils.py - Các hàm tiện ích chung

Chức năng:
- processing_phrases: Clean và normalize text
- directory_exists: Tạo directory an toàn
- extract_json_dict: Parse JSON từ text
"""

import json
import os
import re

# Delimiter cho Knowledge Graph triples
KG_DELIMITER = ","


def processing_phrases(phrase: str) -> str:
    """
    Làm sạch và chuẩn hóa phrase.
    
    Xử lý:
    - Convert int → string
    - Lowercase
    - Loại bỏ ký tự đặc biệt (chỉ giữ A-Z, a-z, 0-9, space)
    - Strip whitespace
    
    Args:
        phrase: Text cần xử lý (str hoặc int)
    
    Returns:
        Text đã được clean và normalize
    
    Ví dụ:
        processing_phrases("Bill Gates (CEO)") → "bill gates ceo"
        processing_phrases("Microsoft Corp.") → "microsoft corp"
        processing_phrases(2024) → "2024"
    """
    if isinstance(phrase, int):
        return str(phrase)
    
    # Regex: [^A-Za-z0-9 ] = match ký tự KHÔNG phải alphanumeric hoặc space
    # Thay thế bằng space, lowercase, và strip
    return re.sub("[^A-Za-z0-9 ]", " ", phrase.lower()).strip()


def directory_exists(path: str) -> None:
    """
    Tạo parent directory của file path nếu chưa tồn tại.
    
    Hữu ích trước khi ghi file để tránh "No such file or directory" error.
    
    Args:
        path: Đường dẫn đầy đủ đến file (ví dụ: "/data/output/file.txt")
    
    Returns:
        None (chỉ tạo directories)
    
    Ví dụ:
        directory_exists("/data/results/output.json")
        # Tạo thư mục /data/results/ nếu chưa có
        
        with open("/data/results/output.json", 'w') as f:
            json.dump(data, f)
    """
    dir = os.path.dirname(path)  # Extract parent directory
    if not os.path.exists(dir):
        os.makedirs(dir)  # Tạo directory + tất cả parents


def extract_json_dict(text: str) -> str | dict:
    """
    Trích xuất JSON dictionary đầu tiên từ text.
    
    Tìm và parse JSON object trong text (hữu ích cho LLM responses).
    
    Args:
        text: Text có thể chứa JSON object
    
    Returns:
        dict: Nếu tìm thấy và parse thành công
        str: Empty string "" nếu không tìm thấy hoặc lỗi
    
    Ví dụ:
        text = "Result: {'name': 'Alice', 'age': 25}"
        extract_json_dict(text) → {'name': 'Alice', 'age': 25}
        
        text = "No JSON here"
        extract_json_dict(text) → ""
    
    Lưu ý:
        - Chỉ trả về JSON object đầu tiên (bỏ qua các object sau)
        - Không hỗ trợ arrays [...]
        - Regex pattern handle nested objects
    """
    # Regex pattern match JSON object (bao gồm nested braces)
    pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\})*)*\}"
    match = re.search(pattern, text)

    if match:
        json_string = match.group()
        try:
            json_dict = json.loads(json_string)
            return json_dict
        except json.JSONDecodeError:
            return ""  # JSON không valid
    else:
        return ""  # Không tìm thấy JSON