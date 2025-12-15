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

ner_instruction = """Your task is to extract named entities from the given paragraph.

Extract entities that are medically relevant OR scientifically/biologically related. Be inclusive but focus on substantive nouns.

=== CATEGORIES TO EXTRACT ===

Extract nouns/noun phrases from these categories:

1. **Drugs & Chemicals**: Medications, compounds, elements, substances
   • Examples: Metformin, Prednisone, Oxygen, beclomethasone dipropionate, cortisol, glucose

2. **Diseases & Conditions**: Medical conditions, syndromes, disorders
   • Examples: diabetes, hypertension, ichthyosis, aldosteronism, seizures

3. **Symptoms & Signs**: Observable manifestations
   • Examples: fever, pain, inflammation, alopecia, ectropion

4. **Anatomy**: Body parts, organs, tissues, cells
   • Examples: liver, adrenal cortex, skin, blood vessels, cornea

5. **Biological Entities**: Proteins, hormones, genes, enzymes
   • Examples: insulin, hemoglobin, TP53, cytochrome c oxidase, glucocorticoid

6. **Medical Procedures**: Treatments, therapies, diagnostic methods
   • Examples: chemotherapy, dialysis, therapy, transplant

=== WHAT TO AVOID ===

Do NOT extract:
- Pure numbers or measurements alone: "94%", "8 hours", "100mg"
- Generic properties alone: "half-life", "protein binding", "clearance"
- Routes alone: "oral", "intravenous" (unless part of entity name)
- Adjectives without nouns: "high", "severe", "chronic"

=== GUIDELINES ===

✓ Extract the entity name as it appears in text (keep proper formatting)
✓ Include chemical names even if complex: "beclomethasone 17-monopropionate"
✓ Include disease syndrome names even if long
✓ Be generous - if it's a concrete noun with medical/biological relevance, extract it
✓ When in doubt, EXTRACT rather than skip

=== EXAMPLES ===

Text: "Hydrocortisone is a glucocorticoid produced by the adrenal cortex with a half-life of 8 hours."
Extract: ["Hydrocortisone", "glucocorticoid", "adrenal cortex"]
Skip: "half-life", "8 hours"

Text: "Oxygen therapy treats hypoxia in patients with respiratory failure."
Extract: ["Oxygen", "therapy", "hypoxia", "patients", "respiratory failure"]

Respond with a JSON list of named entities.
Format: {"named_entities": ["entity1", "entity2", ...]}
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
# Focus: Chỉ trích xuất relations liên quan đến y tế, normalized to canonical forms
one_shot_passage_triples = """{"triples": [
            ["Metformin", "is_a", "first-line medication"],
            ["Metformin", "treats", "type 2 diabetes"],
            ["Metformin", "decreases", "glucose production"],
            ["glucose", "produced_in", "liver"],
            ["Metformin", "improves", "insulin sensitivity"],
            ["insulin", "acts_on", "peripheral tissues"],
            ["Metformin", "causes", "nausea"],
            ["Metformin", "causes", "diarrhea"],
            ["Metformin", "causes", "abdominal pain"],
            ["Metformin", "prevents", "cardiovascular disease"],
            ["Metformin", "prevents", "prediabetes"],
            ["prediabetes", "progresses_to", "type 2 diabetes"],
            ["Metformin", "contraindicated_in", "renal impairment"],
            ["Metformin", "contraindicated_in", "acute kidney injury"]
    ]
}
"""

# Hướng dẫn cho OpenIE Post-NER
openie_post_ner_instruction = """Your task is to construct a MEDICAL RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a MEDICAL relationship in the RDF graph.

CRITICAL: After extracting relationships, you MUST NORMALIZE them to canonical forms to ensure consistency.

=== RELATIONSHIP EXTRACTION & NORMALIZATION PROCESS ===

Step 1: Extract the medical relationship from the text naturally
Step 2: NORMALIZE it to the most common canonical form following the guidelines below

=== CANONICAL MEDICAL RELATIONSHIP FORMS ===

Use these CANONICAL forms (standardized, concise, medical terminology):

1. THERAPEUTIC RELATIONSHIPS:
   • treats (NOT "is used to treat", "helps treat", "used for")
   • prevents (NOT "helps prevent", "can prevent")
   • manages (NOT "helps manage", "used to manage")
   • indicated_for (NOT "indicated in", "prescribed for")
   • contraindicated_in (NOT "should not be used in")

2. CAUSATION RELATIONSHIPS:
   • causes (NOT "can cause", "may cause", "leads to")
   • triggers (for acute events)
   • induces (for physiological changes)

3. AMELIORATION RELATIONSHIPS:
   • alleviates (NOT "helps with", "relieves")
   • reduces (NOT "decreases", "lowers" - use "reduces" for symptoms)
   • improves (NOT "helps improve")

4. PRESENTATION RELATIONSHIPS:
   • presents_with (NOT "shows", "has symptoms of")
   • characterized_by (NOT "characterized as", "marked by")
   • associated_with (NOT "linked to", "related to", "connected to")

5. ANATOMICAL RELATIONSHIPS:
   • affects (NOT "impacts", "influences")
   • damages (NOT "harms", "injures")
   • located_in (NOT "found in", "occurs in")

6. MECHANISM RELATIONSHIPS:
   • inhibits (NOT "blocks", "suppresses" - use "inhibits" for pharmacology)
   • activates (NOT "stimulates", "triggers")
   • increases (for upregulation)
   • decreases (for downregulation)
   • binds_to (for molecular binding)
   • targets (for drug targets)

7. METABOLIC RELATIONSHIPS:
   • metabolized_by (NOT "metabolized in", "broken down by")
   • secreted_by (NOT "secreted from", "produced by" - use for hormones/enzymes)
   • synthesized_in (for biosynthesis)
   • excreted_by (for elimination)

8. CLASSIFICATION RELATIONSHIPS:
   • is_a (NOT "is", "is type of", "classified as")
   • subtype_of (for disease subtypes)

9. PROGRESSION RELATIONSHIPS:
   • progresses_to (NOT "develops into", "leads to")
   • complication_of (NOT "complicates")
   • risk_factor_for (NOT "increases risk of", "risk for")

10. DIAGNOSTIC/PROCEDURAL:
    • diagnoses (NOT "used to diagnose")
    • monitors (NOT "used to monitor")
    • detects (for diagnostic tests)

11. GENETIC RELATIONSHIPS:
    • mutation_causes (for genetic mutations → disease)
    • encodes (gene → protein)
    • regulates (for gene regulation)

=== NORMALIZATION EXAMPLES ===

Text says → Normalize to:
• "is a medication" → "is_a"
• "Metformin is used to treat diabetes" → "treats"
• "can cause nausea" → "causes"
• "helps reduce pain" → "reduces"
• "is broken down by the liver" → "metabolized_by"
• "secreted from the pancreas" → "secreted_by"
• "blocks the enzyme" → "inhibits"
• "is linked to heart disease" → "associated_with"
• "leads to kidney failure" → "progresses_to"
• "shows symptoms of fever" → "presents_with"

=== IMPORTANT RULES ===

1. **ALWAYS normalize** to the canonical form - don't keep verbose forms
2. **Use underscore notation** (e.g., "is_a", "binds_to", "secreted_by")
3. **Avoid generic terms**: Never use "relates_to", "linked_to", "connected_to"
4. **Keep it concise**: "treats" not "is_used_to_treat"
5. **Be consistent**: Same meaning → same relationship name
6. **Medical terminology**: Use proper medical verbs when available

=== EXAMPLES OF CORRECT NORMALIZATION ===

✅ CORRECT:
  ["Metformin", "is_a", "medication"]                    # NOT "is"
  ["Metformin", "treats", "type 2 diabetes"]             # NOT "is used to treat"
  ["Metformin", "causes", "nausea"]                      # NOT "can cause"
  ["insulin", "secreted_by", "pancreas"]                 # NOT "secreted from"
  ["warfarin", "metabolized_by", "liver"]                # NOT "broken down by"
  ["aspirin", "inhibits", "COX enzyme"]                  # NOT "blocks"
  ["diabetes", "associated_with", "obesity"]             # NOT "linked to"

❌ INCORRECT - Not normalized:
  ["Metformin", "is", "medication"]                      # Should be "is_a"
  ["Metformin", "is_used_to_treat", "diabetes"]         # Should be "treats"
  ["Metformin", "can_cause", "nausea"]                  # Should be "causes"
  ["insulin", "secreted_from", "pancreas"]              # Should be "secreted_by"
  ["diabetes", "linked_to", "obesity"]                  # Should be "associated_with"

=== SUMMARY ===

1. Extract relationships from text naturally
2. NORMALIZE to canonical forms using the guidelines above
3. Keep relationship vocabulary controlled but not overly restrictive
4. Expected result: ~40-60 unique relationship types (not 193+)

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