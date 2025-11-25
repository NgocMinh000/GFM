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
- Diseases, disorders, and medical conditions (e.g., "type 2 diabetes mellitus", "myocardial infarction", "essential hypertension")
  * Use full medical terminology when available (e.g., "type 2 diabetes mellitus" not "T2DM" or "diabetes")
  * Include severity/stage if mentioned (e.g., "stage 3 chronic kidney disease", "severe asthma")
- Signs and symptoms (e.g., "chest pain", "shortness of breath", "fever", "diaphoresis")
  * Include clinically relevant qualifiers (e.g., "severe chest pain", "sudden onset chest pain")
- Anatomical structures and body parts (e.g., "right coronary artery", "left arm", "inferior wall")
  * Be specific with laterality and location (e.g., "proximal right coronary artery", "distal left anterior descending artery")

**Medications and Treatments:**
- Drug names - extract generic names (e.g., "Metformin", "Lisinopril", "Aspirin", "Clopidogrel")
  * Prefer generic names over brand names when both are mentioned
  * Extract each drug separately if multiple drugs mentioned together
- Dosages and frequencies - extract as complete units (e.g., "1000mg twice daily", "10mg once daily", "81mg daily")
  * ALWAYS include both dose amount AND frequency together as one entity
  * Include route if mentioned (e.g., "5mg sublingual")
- Treatment procedures (e.g., "percutaneous coronary intervention", "coronary angiography", "coronary artery bypass grafting")
  * Include procedure modifiers (e.g., "emergency coronary angiography", "elective PCI")

**Diagnostic Information:**
- Laboratory tests and biomarkers (e.g., "Troponin I", "CK-MB", "blood glucose", "HbA1c", "creatinine")
  * Extract test name separate from value
- Lab values with units (e.g., "5.8 ng/mL", "185 mg/dL", "160/95 mmHg", "110 beats per minute")
  * ALWAYS include units with numerical values as single entity
  * Extract percentage values with % symbol (e.g., "92%", "95% stenosis")
- Diagnostic procedures (e.g., "12-lead ECG", "coronary angiography", "chest X-ray", "echocardiogram")
- Imaging/test findings (e.g., "ST-segment elevation", "95% stenosis", "thrombus", "ejection fraction")
  * Extract specific findings, not vague terms like "abnormal" alone
- Vital signs (e.g., "blood pressure", "heart rate", "oxygen saturation", "respiratory rate")
  * Extract vital sign name separate from its measured value

**Patient Information:**
- Patient demographics (e.g., "58-year-old male patient", "65-year-old female")
  * Include age, gender together when possible
- Temporal information (e.g., "10 years ago", "2 hours prior", "at bedtime", "for 5 days")
  * Extract specific time references related to medical events or treatments

**Medical Devices and Equipment:**
- Medical devices (e.g., "drug-eluting stent", "bare-metal stent", "pacemaker")
- Clinical locations (e.g., "emergency department", "intensive care unit", "catheterization lab")

**EXTRACTION GUIDELINES:**

1. **Entity Boundaries:**
   - Include modifiers that change medical meaning: "acute myocardial infarction" (keep "acute" with condition)
   - For compound terms, extract the complete medical phrase: "inferior wall myocardial infarction" not just "infarction"
   - Extract medication name and dosage as SEPARATE entities: "Metformin" AND "1000mg twice daily"

2. **Normalization:**
   - Expand common medical abbreviations: "MI" → "myocardial infarction", "HTN" → "hypertension", "DM" → "diabetes mellitus"
   - Use standard medical terminology over lay terms: "heart attack" → "myocardial infarction"
   - Keep measurement units with values: "5.8 ng/mL" not just "5.8"

3. **What NOT to extract:**
   - Generic verbs: "showed", "revealed", "indicated", "diagnosed" (these become relations, not entities)
   - Negation words alone: "no", "without", "denies", "absent" (but DO extract the negated entity)
   - Pronouns: "he", "she", "it", "his", "her" (resolve to actual patient identifier if needed)
   - Vague qualifiers alone: "normal", "abnormal", "positive", "negative" without specific entity

4. **Handling Special Cases:**
   - **Negated findings**: Still extract the entity (e.g., "no fever" → extract "fever")
   - **Uncertain findings**: Extract the entity (e.g., "possible pneumonia" → extract "pneumonia", "rule out MI" → extract "myocardial infarction")
   - **Value ranges**: Extract as single entity (e.g., "120-140 mg/dL", "2-3 weeks")
   - **Multiple items**: Extract each separately (e.g., "Aspirin and Clopidogrel" → "Aspirin", "Clopidogrel")
   - **Acronyms with expansion**: Use full form (e.g., "STEMI (ST-elevation myocardial infarction)" → "ST-elevation myocardial infarction")

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

# Hướng dẫn cho OpenIE Post-NER (Medical Domain) - IMPROVED VERSION
openie_post_ner_instruction = """Your task is to construct a MEDICAL KNOWLEDGE GRAPH from the given clinical text and medical named entity lists.

Extract relationships that represent CLINICAL FACTS and MEDICAL RELATIONSHIPS. Use the following relationship types:

**Patient-Condition Relationships:**
- ["patient", "has medical history of", "disease/condition"] - For chronic/pre-existing conditions
- ["patient", "diagnosed with", "disease/condition"] - For current/new diagnoses
- ["patient", "presented with", "symptom"] - For chief complaints at presentation
- ["patient", "experienced", "symptom"] - For symptoms reported during history
- ["patient", "had vital sign", "vital sign name"] - Link patient to vital sign measurement

**Treatment and Medication Relationships:**
- ["patient", "treated with", "procedure/intervention"] - For therapeutic procedures performed
- ["patient", "prescribed", "medication"] - For medications ordered
- ["patient", "underwent", "diagnostic procedure"] - For diagnostic tests/procedures performed
- ["condition", "managed with", "medication"] - Link condition to its treatment
- ["condition", "treated with", "procedure"] - Link condition to therapeutic procedure
- ["condition", "controlled with", "medication"] - For chronic disease management
- ["medication", "prescribed at", "dosage"] - Link medication to its dose/frequency
- ["medication", "dose increased to", "dosage"] - For dose adjustments
- ["medication", "dose decreased to", "dosage"] - For dose reductions
- ["procedure", "involved", "device/technique"] - Components of procedure

**Diagnostic Relationships:**
- ["diagnostic procedure", "showed", "finding"] - Direct test results
- ["diagnostic procedure", "revealed", "finding"] - Significant findings from tests
- ["finding", "indicates", "diagnosis"] - Findings that establish diagnosis
- ["finding", "consistent with", "diagnosis"] - Findings that support diagnosis
- ["finding", "suggestive of", "diagnosis"] - Findings that hint at diagnosis
- ["biomarker/test", "elevated at", "value"] - For abnormally high values
- ["biomarker/test", "measured at", "value"] - For normal or specific values
- ["vital sign", "measured at", "value"] - Link vital sign to measured value
- ["finding", "observed in", "anatomical location"] - Localize findings to anatomy

**Anatomical and Localization Relationships:**
- ["disease/finding", "affects", "anatomical structure"] - Disease impact on organ/structure
- ["disease/finding", "located in", "anatomical structure"] - Precise anatomical location
- ["device/stent", "placed in", "anatomical structure"] - Device placement location
- ["symptom", "radiates to", "body part"] - Pain/symptom radiation pattern
- ["anatomical structure", "contained", "pathology"] - Pathology within structure

**Temporal Relationships:**
- ["condition", "diagnosed", "time period"] - When condition was diagnosed
- ["symptom", "started", "time point"] - Symptom onset time
- ["medication", "taken for", "duration"] - Duration of medication use
- ["event", "occurred", "time point"] - When medical event happened

**Causality and Association:**
- ["factor", "caused", "condition"] - Direct causation
- ["medication", "controls", "condition"] - Medication effectiveness
- ["condition", "resulted in", "complication"] - Condition leading to complication

**EXTRACTION GUIDELINES:**

1. **Relationship Selection:**
   - Choose predicates that accurately reflect the clinical relationship
   - Use specific predicates over generic ones: prefer "controlled with" over "treated with" for chronic disease management
   - Directionality matters: ["test", "showed", "finding"] NOT ["finding", "showed by", "test"]

2. **Entity Usage:**
   - Each triple MUST contain at least ONE entity from the provided named entity list
   - Strongly prefer triples with BOTH subject and object from the entity list
   - Use "patient" or patient identifier (e.g., "58-year-old male patient") consistently
   - Resolve all pronouns to actual entities

3. **Medical Accuracy:**
   - Only extract relationships that are explicitly stated or strongly implied in the text
   - Do NOT infer relationships based on general medical knowledge alone
   - For diagnostic findings, always link: [procedure] → [finding] → [diagnosis] when available
   - For medications, always link: [patient] → [medication] AND [medication] → [dosage]

4. **Handling Special Cases:**
   - **Negated relationships**: Extract with negative predicate if clinically significant (e.g., ["patient", "denies", "chest pain"])
   - **Uncertain relationships**: You may extract with uncertainty predicates (e.g., ["finding", "suggestive of", "diagnosis"])
   - **Multiple objects**: Create separate triples (e.g., "pain in arm and jaw" → ["pain", "radiates to", "arm"], ["pain", "radiates to", "jaw"])
   - **Intermediate nodes**: Create chain if needed (e.g., [test] → [finding] → [diagnosis] rather than skipping [finding])

5. **Completeness:**
   - For medication mentions, extract BOTH:
     * ["patient", "prescribed", "medication"]
     * ["medication", "prescribed at", "dosage"]
   - For vital signs, extract BOTH:
     * ["patient", "had vital sign", "vital sign name"]
     * ["vital sign name", "measured at", "value"]
   - For diagnostic procedures, extract BOTH:
     * ["patient", "underwent", "procedure"]
     * ["procedure", "showed/revealed", "finding"]

6. **What NOT to extract:**
   - Do NOT create triples with only generic terms not in entity list
   - Do NOT extract relationships between temporal modifiers alone (e.g., ["10 years ago", "before", "5 years ago"])
   - Do NOT create circular or redundant relationships
   - Do NOT extract relationships from example/hypothetical scenarios in the text

Respond with a JSON list of triples, with each triple representing a medical relationship in the format:
[subject, predicate, object]

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
