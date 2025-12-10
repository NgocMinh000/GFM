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
# VÍ DỤ MẪU DÙNG CHUNG (One-shot Example) - MEDICAL DOMAIN
# ============================================================================
# Đoạn văn mẫu về medical domain để hướng dẫn model focus vào entities/relations y tế

one_shot_passage = """Metformin in Type 2 Diabetes
Metformin is a first-line medication for the treatment of type 2 diabetes, particularly in people who are overweight.
It works by decreasing glucose production in the liver and improving insulin sensitivity in peripheral tissues.
Common side effects include nausea, diarrhea, and abdominal pain.
Metformin has been shown to reduce the risk of cardiovascular disease and may help prevent the progression from prediabetes to type 2 diabetes.
The medication is contraindicated in patients with severe renal impairment or acute kidney injury."""

# Output mẫu cho NER: Danh sách các thực thể y tế
# Bao gồm: thuốc, bệnh, triệu chứng, bộ phận cơ thể, protein/hormone
one_shot_passage_entities = """{"named_entities":
    ["Metformin", "type 2 diabetes", "glucose", "liver", "insulin", "peripheral tissues", "nausea", "diarrhea", "abdominal pain", "cardiovascular disease", "prediabetes", "renal impairment", "acute kidney injury"]
}
"""

# ============================================================================
# NER PROMPTS - Hướng dẫn cho Named Entity Recognition
# ============================================================================

ner_instruction = """Your task is to extract MEDICAL named entities from the given paragraph.

IMPORTANT: Only extract entities relevant to the MEDICAL/HEALTHCARE domain. Focus on:
- Drugs/Medications (e.g., Metformin, Aspirin, antibiotics)
- Diseases/Conditions (e.g., diabetes, hypertension, cancer)
- Symptoms/Signs (e.g., fever, pain, nausea, fatigue)
- Medical Procedures/Treatments (e.g., surgery, chemotherapy, dialysis)
- Genes/Proteins/Hormones (e.g., insulin, TP53, hemoglobin)
- Anatomical Structures (e.g., liver, heart, blood vessels, cells)
- Medical Devices/Equipment (e.g., stethoscope, MRI scanner)
- Biological Molecules (e.g., glucose, cholesterol, antibodies)

DO NOT extract:
- General organizations, locations, or dates (unless directly medical)
- People's names (unless they are patients/doctors in medical context)
- Non-medical products or technologies

Respond with a JSON list of medical entities only.
Strictly follow the required JSON format.
"""
# Hướng dẫn: Nhiệm vụ là trích xuất CÁC THỰC THỂ Y TẾ từ đoạn văn
# Focus: Chỉ extract medical/healthcare entities
# Exclude: Non-medical general entities

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

# Output mẫu cho OpenIE: Danh sách các bộ ba (triples) - MEDICAL DOMAIN
# Mỗi triple có dạng [subject, predicate, object]
# Focus: Chỉ trích xuất relations liên quan đến y tế
one_shot_passage_triples = """{"triples": [
            ["Metformin", "is", "first-line medication"],
            ["Metformin", "treats", "type 2 diabetes"],
            ["Metformin", "decreases", "glucose production"],
            ["glucose", "produced in", "liver"],
            ["Metformin", "improves", "insulin sensitivity"],
            ["insulin", "acts on", "peripheral tissues"],
            ["Metformin", "causes", "nausea"],
            ["Metformin", "causes", "diarrhea"],
            ["Metformin", "causes", "abdominal pain"],
            ["Metformin", "reduces risk of", "cardiovascular disease"],
            ["Metformin", "prevents progression from", "prediabetes"],
            ["prediabetes", "progresses to", "type 2 diabetes"],
            ["Metformin", "contraindicated in", "renal impairment"],
            ["Metformin", "contraindicated in", "acute kidney injury"]
    ]
}
"""

# Hướng dẫn cho OpenIE Post-NER
openie_post_ner_instruction = """Your task is to construct a MEDICAL RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a MEDICAL relationship in the RDF graph.

IMPORTANT: Focus on MEDICAL/HEALTHCARE relationships only. Extract relationships such as:
- Drug-Disease: treats, prevents, cures, manages, prescribed for
- Drug-Symptom: causes (side effects), relieves, alleviates
- Disease-Symptom: characterized by, presents with, associated with
- Disease-Anatomical: affects, damages, located in
- Drug-Mechanism: inhibits, blocks, activates, decreases, increases
- Gene-Disease: mutation causes, associated with, risk factor for
- Procedure-Disease: used to treat, diagnoses, monitors

DO NOT extract:
- General non-medical relationships (e.g., organizational, geographical)
- Temporal relationships unless medically relevant

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the medical named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
- Use medical domain relationships whenever possible (e.g., "treats", "causes", "associated with" instead of generic "is related to").

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