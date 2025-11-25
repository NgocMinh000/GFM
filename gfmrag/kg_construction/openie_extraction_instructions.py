"""
================================================================================
FILE: openie_extraction_instructions.py - Hướng dẫn trích xuất OpenIE
================================================================================

MÔ TẢ TỔNG QUAN:
File này chứa các prompts (lời nhắc) và templates (mẫu) để hướng dẫn các mô hình
ngôn ngữ lớn (LLM) thực hiện hai nhiệm vụ chính:
1. Named Entity Recognition (NER) - Nhận dạng thực thể có tên
2. Open Information Extraction (OpenIE) - Trích xuất thông tin mở

CHỨC NĂNG CHÍNH:
1. NER (Named Entity Recognition):
   - Nhận dạng các thực thể có tên trong đoạn văn (người, địa điểm, tổ chức, thời gian, v.v.)
   - Sử dụng few-shot learning với 1 ví dụ mẫu
   - Trả về kết quả dưới dạng JSON

2. OpenIE (Open Information Extraction):
   - Trích xuất các bộ ba quan hệ từ văn bản (subject-predicate-object)
   - Xây dựng RDF graph từ đoạn văn và danh sách entities đã nhận dạng
   - Đảm bảo các triple chứa ít nhất 1, tốt nhất là 2 entities đã nhận dạng
   - Giải quyết các đại từ về tên cụ thể

CẤU TRÚC PROMPTS:
- System Message: Hướng dẫn nhiệm vụ
- Few-shot Examples: Ví dụ mẫu (input + output)
- User Input Template: Template cho input của người dùng

VÍ DỤ MẪU:
Đoạn văn: "Radio City is India's first private FM radio station..."
NER Output: ["Radio City", "India", "3 July 2001", ...]
OpenIE Output: [["Radio City", "located in", "India"], ...]

SỬ DỤNG:
Các prompts này được sử dụng bởi các mô hình OpenIE để:
1. Gọi LLM với ner_prompts để nhận dạng entities
2. Gọi LLM với openie_post_ner_prompts để trích xuất relations

LƯU Ý:
- Sử dụng LangChain's ChatPromptTemplate để tạo cấu trúc hội thoại
- Định dạng output là JSON để dễ parse và xử lý
- Few-shot learning giúp mô hình hiểu rõ format mong muốn
================================================================================
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# ============================================================================
# VÍ DỤ MẪU DÙNG CHUNG (One-shot Example)
# ============================================================================
# Đoạn văn mẫu về Radio City dùng để minh họa cho cả NER và OpenIE

one_shot_passage = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

# Output mẫu cho NER: Danh sách các thực thể có tên
# Bao gồm: tên tổ chức, địa điểm, thời gian, ngôn ngữ, website
one_shot_passage_entities = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""

# ============================================================================
# NER PROMPTS - Hướng dẫn cho Named Entity Recognition
# ============================================================================

ner_instruction = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.
Strictly follow the required JSON format.
"""
# Hướng dẫn: Nhiệm vụ là trích xuất các thực thể có tên từ đoạn văn
# Yêu cầu: Trả về dạng JSON list, tuân thủ nghiêm ngặt định dạng JSON

# Input mẫu cho NER: Đoạn văn được bọc trong ``` để dễ nhận dạng
ner_input_one_shot = f"""Paragraph:
```
{one_shot_passage}
```
"""

# Output mẫu cho NER: Danh sách entities ở dạng JSON
ner_output_one_shot = one_shot_passage_entities

# Template cho input của người dùng
# Sẽ được fill với user_input khi gọi
ner_user_input = "Paragraph:```\n{user_input}\n```"

# Tạo ChatPromptTemplate hoàn chỉnh cho NER
# Cấu trúc: System -> Human (example) -> AI (example) -> Human (user input)
ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(ner_instruction),           # Hướng dẫn nhiệm vụ
        HumanMessage(ner_input_one_shot),        # Ví dụ input
        AIMessage(ner_output_one_shot),          # Ví dụ output
        HumanMessagePromptTemplate.from_template(ner_user_input),  # User input
    ]
)

# ============================================================================
# OpenIE PROMPTS - Hướng dẫn cho Open Information Extraction
# ============================================================================

# Output mẫu cho OpenIE: Danh sách các bộ ba (triples)
# Mỗi triple có dạng [subject, predicate, object]
# LƯU Ý: Có lỗi cú pháp JSON ở dòng 5 (thiếu dấu phẩy) - đây có thể là cố ý
# để test khả năng xử lý lỗi của mô hình
one_shot_passage_triples = """{"triples": [
            ["Radio City", "located in", "India"],
            ["Radio City", "is", "private FM radio station"],
            ["Radio City", "started on", "3 July 2001"],
            ["Radio City", "plays songs in", "Hindi"],
            ["Radio City", "plays songs in", "English"]
            ["Radio City", "forayed into", "New Media"],
            ["Radio City", "launched", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "launched in", "May 2008"],
            ["PlanetRadiocity.com", "is", "music portal"],
            ["PlanetRadiocity.com", "offers", "news"],
            ["PlanetRadiocity.com", "offers", "videos"],
            ["PlanetRadiocity.com", "offers", "songs"]
    ]
}
"""

# Hướng dẫn cho OpenIE Post-NER
openie_post_ner_instruction = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

"""
# Hướng dẫn: 
# - Xây dựng RDF graph từ đoạn văn và danh sách entities
# - Mỗi triple nên chứa ít nhất 1, tốt nhất là 2 named entities
# - Giải quyết đại từ (pronouns) về tên cụ thể để rõ ràng
#   Ví dụ: "It plays" -> "Radio City plays"

# Frame (template) cho OpenIE input
# Kết hợp cả đoạn văn và JSON entities đã trích xuất từ NER
openie_post_ner_frame = """Convert the paragraph into a JSON dict, it has a named entity list and a triple list.
Paragraph:
```
{passage}
```

{named_entity_json}
"""
# Template này yêu cầu:
# - Input: Đoạn văn + danh sách named entities (từ NER)
# - Output: JSON dict chứa cả entities và triples

# Tạo input mẫu cho OpenIE bằng cách thay thế placeholders
openie_post_ner_input_one_shot = openie_post_ner_frame.replace(
    "{passage}", one_shot_passage
).replace("{named_entity_json}", one_shot_passage_entities)
# Kết quả: Đoạn văn Radio City + danh sách entities đã nhận dạng

# Output mẫu cho OpenIE: Danh sách các triples
openie_post_ner_output_one_shot = one_shot_passage_triples

# Tạo ChatPromptTemplate hoàn chỉnh cho OpenIE
# Cấu trúc tương tự NER: System -> Example -> User input
openie_post_ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(openie_post_ner_instruction),     # Hướng dẫn nhiệm vụ
        HumanMessage(openie_post_ner_input_one_shot),   # Ví dụ input
        AIMessage(openie_post_ner_output_one_shot),     # Ví dụ output
        HumanMessagePromptTemplate.from_template(openie_post_ner_frame),  # User input
    ]
)

# ============================================================================
# CÁCH SỬ DỤNG TRONG CODE
# ============================================================================
"""
# 1. NER - Nhận dạng entities:
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
input_text = "Barack Obama was born in Hawaii..."

# Format prompt với input của user
messages = ner_prompts.format_messages(user_input=input_text)

# Gọi LLM
response = llm.invoke(messages)

# Parse JSON response
entities = json.loads(response.content)
# Output: {"named_entities": ["Barack Obama", "Hawaii", ...]}

# 2. OpenIE - Trích xuất relations:
# Format prompt với passage và entities
messages = openie_post_ner_prompts.format_messages(
    passage=input_text,
    named_entity_json=json.dumps(entities)
)

# Gọi LLM
response = llm.invoke(messages)

# Parse JSON response
triples = json.loads(response.content)
# Output: {"triples": [["Barack Obama", "born in", "Hawaii"], ...]}
"""