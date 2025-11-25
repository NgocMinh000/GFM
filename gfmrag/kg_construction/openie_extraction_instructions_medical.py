"""
================================================================================
FILE: openie_extraction_instructions_medical.py - Medical Domain Prompts
================================================================================

MÔ TẢ:
File này chứa prompts được customize cho domain Y TẾ.
Tập trung vào trích xuất:
- Thực thể y khoa: Bệnh, thuốc, triệu chứng, cơ quan, xét nghiệm
- Quan hệ y khoa: Chẩn đoán, điều trị, chỉ định, ảnh hưởng

SỬ DỤNG:
Thay thế imports trong các file sử dụng:
  from gfmrag.kg_construction.openie_extraction_instructions_medical import (
      ner_prompts,
      openie_post_ner_prompts,
  )
================================================================================
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

# ============================================================================
# VÍ DỤ MẪU - MEDICAL CASE REPORT (One-shot Example)
# ============================================================================

one_shot_passage = """Patient Case Report: Acute Myocardial Infarction
A 58-year-old male patient with a history of type 2 diabetes mellitus and hypertension presented to the emergency department with sudden onset of severe chest pain radiating to the left arm and jaw, accompanied by shortness of breath and diaphoresis. The pain started 2 hours prior to arrival.

Past Medical History:
- Type 2 diabetes mellitus (diagnosed 10 years ago), managed with Metformin 1000mg twice daily
- Essential hypertension (diagnosed 5 years ago), controlled with Lisinopril 10mg once daily and Amlodipine 5mg once daily
- Hyperlipidemia, treated with Atorvastatin 20mg at bedtime

Physical Examination:
- Blood pressure: 160/95 mmHg
- Heart rate: 110 beats per minute
- Oxygen saturation: 92% on room air

Diagnostic Tests:
- 12-lead ECG showed ST-segment elevation in leads II, III, and aVF, consistent with inferior wall myocardial infarction
- Troponin I: 5.8 ng/mL (normal <0.04 ng/mL) - markedly elevated
- CK-MB: 85 U/L (normal <25 U/L) - elevated
- Blood glucose: 185 mg/dL

Emergency coronary angiography revealed 95% stenosis of the proximal right coronary artery with evidence of thrombus. Percutaneous coronary intervention (PCI) was performed with successful placement of a drug-eluting stent.

Post-procedure medications include dual antiplatelet therapy with Aspirin 81mg daily and Clopidogrel 75mg daily, continuation of Atorvastatin at increased dose of 80mg daily, and addition of Metoprolol 25mg twice daily."""

# Output mẫu cho NER: Danh sách các thực thể y khoa
one_shot_passage_entities = """{"named_entities": [
    "58-year-old male patient",
    "type 2 diabetes mellitus",
    "hypertension",
    "emergency department",
    "chest pain",
    "left arm",
    "jaw",
    "shortness of breath",
    "diaphoresis",
    "Metformin",
    "1000mg twice daily",
    "essential hypertension",
    "Lisinopril",
    "10mg once daily",
    "Amlodipine",
    "5mg once daily",
    "hyperlipidemia",
    "Atorvastatin",
    "20mg at bedtime",
    "blood pressure",
    "160/95 mmHg",
    "heart rate",
    "110 beats per minute",
    "oxygen saturation",
    "92%",
    "12-lead ECG",
    "ST-segment elevation",
    "leads II, III, and aVF",
    "inferior wall myocardial infarction",
    "Troponin I",
    "5.8 ng/mL",
    "CK-MB",
    "85 U/L",
    "blood glucose",
    "185 mg/dL",
    "coronary angiography",
    "95% stenosis",
    "proximal right coronary artery",
    "thrombus",
    "percutaneous coronary intervention",
    "PCI",
    "drug-eluting stent",
    "dual antiplatelet therapy",
    "Aspirin",
    "81mg daily",
    "Clopidogrel",
    "75mg daily",
    "80mg daily",
    "Metoprolol",
    "25mg twice daily"
]}
"""

# ============================================================================
# NER PROMPTS - Hướng dẫn cho Named Entity Recognition (Medical Domain)
# ============================================================================

ner_instruction = """Your task is to extract MEDICAL named entities from the given clinical text or medical document.

Focus on extracting the following types of medical entities:

**Clinical Entities:**
- Diseases, disorders, and medical conditions (e.g., "type 2 diabetes mellitus", "myocardial infarction")
- Signs and symptoms (e.g., "chest pain", "shortness of breath", "fever")
- Anatomical structures and body parts (e.g., "right coronary artery", "left arm", "inferior wall")

**Medications and Treatments:**
- Drug names (generic and brand names) (e.g., "Metformin", "Lisinopril", "Aspirin")
- Dosages and frequencies (e.g., "1000mg twice daily", "10mg once daily")
- Treatment procedures (e.g., "percutaneous coronary intervention", "coronary angiography")

**Diagnostic Information:**
- Laboratory tests and values (e.g., "Troponin I", "5.8 ng/mL", "blood glucose")
- Diagnostic procedures (e.g., "12-lead ECG", "coronary angiography")
- Imaging findings (e.g., "ST-segment elevation", "95% stenosis")
- Vital signs (e.g., "blood pressure", "160/95 mmHg", "heart rate")

**Patient Information:**
- Age and demographics (e.g., "58-year-old male patient")
- Temporal information (e.g., "10 years ago", "2 hours prior")

**Medical Devices and Equipment:**
- Medical devices (e.g., "drug-eluting stent")
- Equipment (e.g., "emergency department")

Respond with a JSON list of entities.
Strictly follow the required JSON format: {"named_entities": [...]}
"""

# Input mẫu cho NER: Đoạn văn được bọc trong ``` để dễ nhận dạng
ner_input_one_shot = f"""Clinical Text:
```
{one_shot_passage}
```
"""

# Output mẫu cho NER: Danh sách entities ở dạng JSON
ner_output_one_shot = one_shot_passage_entities

# Template cho input của người dùng
ner_user_input = "Clinical Text:```\n{user_input}\n```"

# Tạo ChatPromptTemplate hoàn chỉnh cho NER
ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(ner_instruction),           # Hướng dẫn nhiệm vụ
        HumanMessage(ner_input_one_shot),        # Ví dụ input
        AIMessage(ner_output_one_shot),          # Ví dụ output
        HumanMessagePromptTemplate.from_template(ner_user_input),  # User input
    ]
)

# ============================================================================
# OpenIE PROMPTS - Hướng dẫn cho Open Information Extraction (Medical Domain)
# ============================================================================

# Output mẫu cho OpenIE: Danh sách các bộ ba (triples) về quan hệ y khoa
one_shot_passage_triples = """{"triples": [
    ["58-year-old male patient", "has medical history of", "type 2 diabetes mellitus"],
    ["58-year-old male patient", "has medical history of", "hypertension"],
    ["58-year-old male patient", "presented with", "chest pain"],
    ["chest pain", "radiates to", "left arm"],
    ["chest pain", "radiates to", "jaw"],
    ["58-year-old male patient", "experienced", "shortness of breath"],
    ["58-year-old male patient", "experienced", "diaphoresis"],
    ["type 2 diabetes mellitus", "managed with", "Metformin"],
    ["Metformin", "prescribed at", "1000mg twice daily"],
    ["essential hypertension", "controlled with", "Lisinopril"],
    ["Lisinopril", "prescribed at", "10mg once daily"],
    ["essential hypertension", "controlled with", "Amlodipine"],
    ["Amlodipine", "prescribed at", "5mg once daily"],
    ["hyperlipidemia", "treated with", "Atorvastatin"],
    ["Atorvastatin", "prescribed at", "20mg at bedtime"],
    ["58-year-old male patient", "had vital sign", "blood pressure"],
    ["blood pressure", "measured at", "160/95 mmHg"],
    ["58-year-old male patient", "had vital sign", "heart rate"],
    ["heart rate", "measured at", "110 beats per minute"],
    ["58-year-old male patient", "had vital sign", "oxygen saturation"],
    ["oxygen saturation", "measured at", "92%"],
    ["58-year-old male patient", "underwent", "12-lead ECG"],
    ["12-lead ECG", "showed", "ST-segment elevation"],
    ["ST-segment elevation", "observed in", "leads II, III, and aVF"],
    ["ST-segment elevation", "indicates", "inferior wall myocardial infarction"],
    ["Troponin I", "elevated at", "5.8 ng/mL"],
    ["Troponin I", "indicates", "myocardial infarction"],
    ["CK-MB", "elevated at", "85 U/L"],
    ["blood glucose", "measured at", "185 mg/dL"],
    ["58-year-old male patient", "underwent", "coronary angiography"],
    ["coronary angiography", "revealed", "95% stenosis"],
    ["stenosis", "located in", "proximal right coronary artery"],
    ["proximal right coronary artery", "contained", "thrombus"],
    ["58-year-old male patient", "treated with", "percutaneous coronary intervention"],
    ["percutaneous coronary intervention", "involved", "drug-eluting stent"],
    ["drug-eluting stent", "placed in", "proximal right coronary artery"],
    ["58-year-old male patient", "prescribed", "dual antiplatelet therapy"],
    ["dual antiplatelet therapy", "includes", "Aspirin"],
    ["Aspirin", "prescribed at", "81mg daily"],
    ["dual antiplatelet therapy", "includes", "Clopidogrel"],
    ["Clopidogrel", "prescribed at", "75mg daily"],
    ["Atorvastatin", "dose increased to", "80mg daily"],
    ["58-year-old male patient", "prescribed", "Metoprolol"],
    ["Metoprolol", "prescribed at", "25mg twice daily"]
]}
"""

# Hướng dẫn cho OpenIE Post-NER (Medical Domain)
openie_post_ner_instruction = """Your task is to construct a MEDICAL KNOWLEDGE GRAPH from the given clinical text and medical named entity lists.

Extract relationships that represent CLINICAL FACTS and MEDICAL RELATIONSHIPS, such as:

**Patient-Condition Relationships:**
- ["patient", "has medical history of", "disease/condition"]
- ["patient", "diagnosed with", "disease/condition"]
- ["patient", "presented with", "symptom"]
- ["patient", "experienced", "symptom"]

**Treatment and Medication Relationships:**
- ["patient", "treated with", "medication/procedure"]
- ["patient", "prescribed", "medication"]
- ["patient", "underwent", "procedure/test"]
- ["condition", "managed with", "medication"]
- ["condition", "treated with", "procedure"]
- ["medication", "prescribed at", "dosage"]
- ["medication", "dose increased to", "dosage"]

**Diagnostic Relationships:**
- ["test/procedure", "showed", "finding"]
- ["test/procedure", "revealed", "finding"]
- ["finding", "indicates", "diagnosis"]
- ["finding", "consistent with", "diagnosis"]
- ["biomarker", "elevated at", "value"]
- ["biomarker", "measured at", "value"]

**Anatomical and Localization Relationships:**
- ["disease/finding", "affects", "anatomical structure"]
- ["disease/finding", "located in", "anatomical structure"]
- ["finding", "observed in", "anatomical location"]
- ["device/stent", "placed in", "anatomical structure"]
- ["symptom", "radiates to", "body part"]

**Temporal Relationships:**
- ["condition", "diagnosed", "time period"]
- ["symptom", "started", "time point"]
- ["medication", "taken for", "duration"]

**Causality and Association:**
- ["condition", "caused by", "factor"]
- ["medication", "controls", "condition"]
- ["finding", "suggestive of", "diagnosis"]

Respond with a JSON list of triples, with each triple representing a medical relationship in the format:
[subject, predicate, object]

**IMPORTANT REQUIREMENTS:**
- Each triple MUST contain at least one, but preferably TWO, of the medical named entities from the provided list
- Use clinically accurate predicates that reflect true medical relationships
- Clearly resolve pronouns to specific patient identifiers or medical terms (e.g., "He" → "patient")
- Maintain temporal accuracy and causality when extracting time-related relationships
- Focus on extracting factual medical information, not speculative or uncertain statements
- For medications, always include dosage information when available
- For diagnostic findings, link them to the diagnostic procedure and clinical significance

Strictly follow the required JSON format: {"triples": [...]}
"""

# Frame (template) cho OpenIE input
openie_post_ner_frame = """Convert the clinical text into a JSON dict containing the named entity list and a medical knowledge graph triple list.

Clinical Text:
```
{passage}
```

{named_entity_json}
"""

# Tạo input mẫu cho OpenIE bằng cách thay thế placeholders
openie_post_ner_input_one_shot = openie_post_ner_frame.replace(
    "{passage}", one_shot_passage
).replace("{named_entity_json}", one_shot_passage_entities)

# Output mẫu cho OpenIE: Danh sách các triples
openie_post_ner_output_one_shot = one_shot_passage_triples

# Tạo ChatPromptTemplate hoàn chỉnh cho OpenIE
openie_post_ner_prompts = ChatPromptTemplate.from_messages(
    [
        SystemMessage(openie_post_ner_instruction),     # Hướng dẫn nhiệm vụ
        HumanMessage(openie_post_ner_input_one_shot),   # Ví dụ input
        AIMessage(openie_post_ner_output_one_shot),     # Ví dụ output
        HumanMessagePromptTemplate.from_template(openie_post_ner_frame),  # User input
    ]
)

# ============================================================================
# MEDICAL RELATIONSHIP TYPES (Reference)
# ============================================================================
"""
Common medical predicates for knowledge graph construction:

DIAGNOSIS & CONDITIONS:
- diagnosed_with, has_medical_history_of, presented_with, experienced

TREATMENTS:
- treated_with, managed_with, controlled_with, prescribed, underwent

MEDICATIONS:
- prescribed_at, dose_increased_to, taking, administered

DIAGNOSTICS:
- showed, revealed, indicated, consistent_with, suggestive_of

MEASUREMENTS:
- measured_at, elevated_at, level_of

ANATOMY:
- affects, located_in, observed_in, placed_in, radiates_to, contained

TEMPORAL:
- started, occurred, diagnosed_ago, for_duration

CAUSALITY:
- caused_by, due_to, results_in, leads_to
"""
