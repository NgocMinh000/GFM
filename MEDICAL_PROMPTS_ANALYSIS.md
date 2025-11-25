# BÁO CÁO PHÂN TÍCH PROMPTS - HỆ THỐNG GFM-RAG

## 📋 MỤC LỤC
1. [Tổng quan](#tổng-quan)
2. [Prompts cho NER (Named Entity Recognition)](#1-prompts-cho-ner)
3. [Prompts cho OpenIE (Open Information Extraction)](#2-prompts-cho-openie)
4. [Prompts cho QA (Question Answering)](#3-prompts-cho-qa)
5. [Đề xuất customize cho domain Y tế](#4-đề-xuất-customize-cho-domain-y-tế)

---

## TỔNG QUAN

Hệ thống GFM-RAG sử dụng LLMs (YEScale API) cho 3 tác vụ chính:
1. **NER**: Nhận dạng các thực thể có tên trong text
2. **OpenIE**: Trích xuất quan hệ giữa các thực thể (triples: subject-predicate-object)
3. **QA**: Trả lời câu hỏi dựa trên documents được retrieve

Tất cả các prompts đều sử dụng **Few-shot learning** (cung cấp examples) để hướng dẫn LLM.

---

## 1. PROMPTS CHO NER (Named Entity Recognition)

### 📍 VỊ TRÍ FILES:

#### File 1: `gfmrag/kg_construction/openie_extraction_instructions.py`
**Dòng 70-100**

```python
ner_instruction = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.
Strictly follow the required JSON format.
"""

# Ví dụ mẫu
one_shot_passage = """Radio City
Radio City is India's first private FM radio station and was started on 3 July 2001.
It plays Hindi, English and regional songs.
Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features."""

one_shot_passage_entities = """{"named_entities":
    ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]
}
"""
```

**Đặc điểm:**
- ✅ Domain-agnostic (không specific cho domain nào)
- ✅ Few-shot với 1 example
- ✅ Output: JSON format
- ⚠️ Example về radio station (không liên quan y tế)

---

#### File 2: `gfmrag/kg_construction/ner_model/llm_ner_model.py`
**Dòng 33-47**

```python
query_prompt_one_shot_input = """Please extract all named entities that are important for solving the questions below.
Place the named entities in json format.

Question: Which magazine was started first Arthur's Magazine or First for Women?
"""

query_prompt_one_shot_output = """
{"named_entities": ["First for Women", "Arthur's Magazine"]}
"""

query_prompt_template = """
Question: {}
"""
```

**Đặc điểm:**
- ✅ Tập trung vào entities quan trọng cho câu hỏi
- ✅ Few-shot với 1 example
- ⚠️ Example về magazines (không liên quan y tế)
- ℹ️ Được dùng cho QA-based NER

---

## 2. PROMPTS CHO OPENIE (Open Information Extraction)

### 📍 VỊ TRÍ FILE: `gfmrag/kg_construction/openie_extraction_instructions.py`
**Dòng 105-174**

```python
openie_post_ner_instruction = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
"""

# Ví dụ output
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
```

**Đặc điểm:**
- ✅ Yêu cầu rõ ràng về RDF graph construction
- ✅ Hướng dẫn giải quyết đại từ (pronoun resolution)
- ✅ Yêu cầu triple chứa ít nhất 1-2 named entities
- ⚠️ Example về radio station (không liên quan y tế)
- ⚠️ Relations trong example: "located in", "is", "started on", "plays songs in", "launched", "offers"

---

## 3. PROMPTS CHO QA (Question Answering)

### 📍 VỊ TRÍ FILES:

#### File 1: `gfmrag/workflow/config/qa_prompt/zero_shot.yaml`

```yaml
system_prompt: 'As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'

doc_prompt: "Wikipedia Title: {title}\n{content}\n"

question_prompt: "Question: {question}\nThought: "

examples: []
```

**Đặc điểm:**
- ✅ Yêu cầu reasoning process (Thought:)
- ✅ Concise answer
- ⚠️ Giả định documents từ Wikipedia
- ⚠️ Không có examples (zero-shot)

---

#### File 2: `gfmrag/workflow/config/qa_prompt/hotpotqa.yaml`

```yaml
system_prompt: 'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Conclude with "Answer: " to present a concise, definitive response, devoid of additional elaborations.'

doc_prompt: "Wikipedia Title: {title}\n{content}\n"

question_prompt: "Question: {question}\nThought: "

examples:
  - input: |-
      Wikipedia Title: Milk and Honey (album)
      [... John Lennon album information ...]

      Wikipedia Title: Walls and Bridges
      [... John Lennon album information ...]

      Question: Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records...?

    response: |-
      The album issued by Apple Records... is Walls and Bridges.
      Answer: Walls and Bridges.
```

**Đặc điểm:**
- ✅ Few-shot với examples
- ✅ Multi-hop reasoning (cần kết hợp nhiều documents)
- ⚠️ Examples về John Lennon/music (không liên quan y tế)
- ⚠️ Giả định Wikipedia format

---

## 4. ĐỀ XUẤT CUSTOMIZE CHO DOMAIN Y TẾ

### 🏥 A. PROMPTS NER CHO Y TẾ

#### Đề xuất thay đổi `openie_extraction_instructions.py`:

```python
# CURRENT (Generic)
ner_instruction = """Your task is to extract named entities from the given paragraph.
Respond with a JSON list of entities.
Strictly follow the required JSON format.
"""

# PROPOSED (Medical Domain)
ner_instruction = """Your task is to extract medical named entities from the given clinical text or medical document.
Focus on extracting:
- Diseases, conditions, and symptoms
- Medications and treatments
- Medical procedures and tests
- Anatomical structures
- Patient demographics (age, gender)
- Medical devices and equipment
- Laboratory values and measurements

Respond with a JSON list of entities.
Strictly follow the required JSON format.
"""

# Example passage (Medical)
one_shot_passage = """Patient Case Report
A 45-year-old male patient presented to the emergency department with acute chest pain radiating to the left arm.
The patient has a history of hypertension and type 2 diabetes mellitus, currently managed with Metformin 1000mg twice daily and Lisinopril 10mg once daily.
ECG showed ST-segment elevation in leads II, III, and aVF, suggesting inferior wall myocardial infarction.
Troponin I levels were elevated at 5.2 ng/mL (normal <0.04 ng/mL).
Emergency coronary angiography was performed, revealing 90% stenosis of the right coronary artery."""

one_shot_passage_entities = """{"named_entities": [
    "45-year-old male",
    "acute chest pain",
    "left arm",
    "hypertension",
    "type 2 diabetes mellitus",
    "Metformin",
    "1000mg",
    "Lisinopril",
    "10mg",
    "ECG",
    "ST-segment elevation",
    "inferior wall myocardial infarction",
    "Troponin I",
    "5.2 ng/mL",
    "coronary angiography",
    "right coronary artery",
    "90% stenosis"
]}
"""
```

---

### 🏥 B. PROMPTS OPENIE CHO Y TẾ

#### Đề xuất thay đổi cho `openie_post_ner_instruction`:

```python
# CURRENT (Generic)
openie_post_ner_instruction = """Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.
"""

# PROPOSED (Medical Domain)
openie_post_ner_instruction = """Your task is to construct a medical knowledge graph from the given clinical text and named entity lists.
Extract relationships representing medical facts, such as:
- Patient characteristics: ["patient", "has", "disease/condition"]
- Treatments: ["patient", "treated with", "medication/procedure"]
- Diagnostic findings: ["test", "shows", "finding"]
- Dosage information: ["medication", "prescribed at", "dose"]
- Anatomical relationships: ["disease", "affects", "anatomical structure"]
- Temporal relationships: ["symptom", "occurred before", "diagnosis"]
- Causal relationships: ["condition", "caused by", "factor"]

Respond with a JSON list of triples, with each triple representing a medical relationship.

Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the medical entities in the list.
- Use clinically accurate predicates (e.g., "diagnosed with", "treated with", "indicates")
- Clearly resolve pronouns to specific patient or medical terms.
- Maintain temporal accuracy when extracting time-related relationships.
"""

# Example triples (Medical)
one_shot_passage_triples = """{"triples": [
    ["45-year-old male patient", "presented with", "acute chest pain"],
    ["acute chest pain", "radiates to", "left arm"],
    ["patient", "has medical history of", "hypertension"],
    ["patient", "has medical history of", "type 2 diabetes mellitus"],
    ["patient", "treated with", "Metformin"],
    ["Metformin", "prescribed at", "1000mg twice daily"],
    ["patient", "treated with", "Lisinopril"],
    ["Lisinopril", "prescribed at", "10mg once daily"],
    ["ECG", "showed", "ST-segment elevation"],
    ["ST-segment elevation", "indicates", "inferior wall myocardial infarction"],
    ["Troponin I", "elevated at", "5.2 ng/mL"],
    ["patient", "underwent", "coronary angiography"],
    ["coronary angiography", "revealed", "90% stenosis"],
    ["stenosis", "located in", "right coronary artery"]
]}
"""
```

---

### 🏥 C. PROMPTS QA CHO Y TẾ

#### Đề xuất thay đổi `qa_prompt/zero_shot.yaml`:

```yaml
# PROPOSED (Medical Domain)
system_prompt: 'As an advanced medical knowledge assistant, your task is to analyze clinical documents, research papers, and medical literature to answer medical questions accurately. Your response should start after "Thought: ", where you will:
1. Identify relevant medical entities and concepts
2. Analyze relationships between symptoms, diagnoses, treatments
3. Consider clinical context and medical evidence
4. Provide reasoning based on medical knowledge

Conclude with "Answer: " to present a concise, evidence-based medical response. Always cite specific information from the provided documents.'

doc_prompt: "Medical Document: {title}\n{content}\n"

question_prompt: "Medical Question: {question}\nThought: "

examples: []
```

#### Đề xuất example cho `qa_prompt/medical.yaml`:

```yaml
system_prompt: 'As an advanced medical knowledge assistant...'

doc_prompt: "Medical Document: {title}\n{content}\n"

question_prompt: "Medical Question: {question}\nThought: "

examples:
  - input: |-
      Medical Document: Diabetes Mellitus Management Guidelines
      Type 2 diabetes mellitus is a chronic metabolic disorder characterized by hyperglycemia resulting from insulin resistance and relative insulin deficiency. First-line pharmacological treatment typically includes Metformin, which reduces hepatic glucose production and improves insulin sensitivity. The standard initial dose is 500-850mg once or twice daily, titrated up to a maximum of 2000-2550mg per day. Common side effects include gastrointestinal disturbances such as nausea and diarrhea.

      Medical Document: Cardiovascular Risk in Diabetes
      Patients with type 2 diabetes have a 2-4 fold increased risk of cardiovascular disease compared to the general population. Hypertension is present in approximately 70% of diabetic patients. ACE inhibitors like Lisinopril are recommended as first-line antihypertensive agents in diabetic patients due to their nephroprotective effects. Typical starting dose is 10mg once daily, which can be increased to 40mg daily if needed.

      Medical Question: What are the first-line medications for managing a patient with type 2 diabetes and hypertension, and what are their typical starting doses?
      Thought:

    response: |-
      Based on the provided documents:
      1. For type 2 diabetes: Metformin is the first-line treatment, with starting dose of 500-850mg once or twice daily
      2. For hypertension in diabetic patients: ACE inhibitors like Lisinopril are first-line, with starting dose of 10mg once daily
      3. These medications address both conditions while providing additional benefits (Metformin improves insulin sensitivity, Lisinopril offers nephroprotection)

      Answer: First-line medications are Metformin (500-850mg once or twice daily) for diabetes and Lisinopril (10mg once daily) for hypertension. These are recommended specifically for diabetic patients with hypertension.
```

---

## 5. TẬP TRUNG CÁC FILE CẦN SỬA

### ⚠️ PRIORITY 1 - CORE PROMPTS (BẮT BUỘC SỬA):

1. **`gfmrag/kg_construction/openie_extraction_instructions.py`**
   - Dòng 70-74: `ner_instruction`
   - Dòng 54-64: `one_shot_passage` và `one_shot_passage_entities` (NER example)
   - Dòng 128-139: `openie_post_ner_instruction`
   - Dòng 110-125: `one_shot_passage_triples` (OpenIE example)

2. **`gfmrag/kg_construction/ner_model/llm_ner_model.py`**
   - Dòng 33-42: `query_prompt_one_shot_input` và `query_prompt_one_shot_output`

### ⚠️ PRIORITY 2 - QA PROMPTS (NÊN SỬA):

3. **`gfmrag/workflow/config/qa_prompt/zero_shot.yaml`**
   - Dòng 1: `system_prompt`
   - Dòng 2: `doc_prompt`

4. **`gfmrag/workflow/config/qa_prompt/hotpotqa.yaml`**
   - Dòng 1: `system_prompt`
   - Dòng 2: `doc_prompt`
   - Dòng 5-25: `examples` (thay bằng medical examples)

### 📝 OPTIONAL - TẠO FILE MỚI:

5. **`gfmrag/workflow/config/qa_prompt/medical.yaml`** (TẠO MỚI)
   - Copy từ `hotpotqa.yaml`
   - Customize với medical examples

---

## 6. HƯỚNG DẪN CUSTOMIZE

### Bước 1: Backup files gốc
```bash
cd /home/user/GFM
cp gfmrag/kg_construction/openie_extraction_instructions.py gfmrag/kg_construction/openie_extraction_instructions.py.bak
cp gfmrag/kg_construction/ner_model/llm_ner_model.py gfmrag/kg_construction/ner_model/llm_ner_model.py.bak
```

### Bước 2: Sửa NER prompts
- Mở `openie_extraction_instructions.py`
- Thay thế `ner_instruction` bằng medical version
- Thay thế `one_shot_passage` bằng medical case report
- Thay thế `one_shot_passage_entities` với medical entities

### Bước 3: Sửa OpenIE prompts
- Trong cùng file `openie_extraction_instructions.py`
- Thay thế `openie_post_ner_instruction` bằng medical version
- Thay thế `one_shot_passage_triples` với medical triples

### Bước 4: Sửa QA prompts
- Mở `gfmrag/workflow/config/qa_prompt/zero_shot.yaml`
- Customize `system_prompt` và `doc_prompt`
- Tạo `medical.yaml` với medical examples

### Bước 5: Test
```bash
python test_yescale_connection.py  # Verify API works
python -m gfmrag.workflow.stage1_index_dataset  # Test with medical data
```

---

## 7. LƯU Ý QUAN TRỌNG

### ✅ DO:
- Giữ nguyên JSON format output
- Giữ nguyên structure của prompts (System -> Example -> User)
- Sử dụng medical terminology chính xác
- Examples phải realistic và clinically accurate
- Test với medical dataset thật

### ❌ DON'T:
- Không thay đổi variable names
- Không thay đổi format của one_shot variables
- Không xóa docstrings
- Không break JSON syntax trong examples
- Không sử dụng abbreviations không giải thích

---

## 8. CHECKLIST

- [ ] Backup files gốc
- [ ] Đọc và hiểu current prompts
- [ ] Chuẩn bị medical examples (real case reports)
- [ ] Sửa NER instruction trong `openie_extraction_instructions.py`
- [ ] Sửa NER examples trong `openie_extraction_instructions.py`
- [ ] Sửa OpenIE instruction trong `openie_extraction_instructions.py`
- [ ] Sửa OpenIE examples trong `openie_extraction_instructions.py`
- [ ] Sửa NER query prompts trong `llm_ner_model.py`
- [ ] Sửa QA system prompts trong `zero_shot.yaml`
- [ ] Tạo `medical.yaml` với medical QA examples
- [ ] Test API connection
- [ ] Test stage1 pipeline với medical data
- [ ] Verify output quality
- [ ] Document changes

---

## 9. CONTACT & SUPPORT

Nếu cần hỗ trợ khi customize prompts:
- Test YEScale API: `python test_yescale_connection.py`
- Test stage1 pipeline: `python -m gfmrag.workflow.stage1_index_dataset`
- Check logs: `gfmrag/workflow/outputs/kg_construction/*/stage1_index_dataset.log`

---

**Generated:** 2025-11-25
**System:** GFM-RAG with YEScale LLM Integration
**Purpose:** Medical Domain Customization Guide
