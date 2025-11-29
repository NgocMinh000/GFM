# TỔNG KẾT MEDICAL DOMAIN CUSTOMIZATION

## 📊 OVERVIEW

Hệ thống GFM-RAG đã được customize hoàn chỉnh cho domain Y TẾ với các thành phần sau:

### ✅ Hoàn thành:
1. ✅ **YEScale API Integration** - Kết nối với YEScale LLM API
2. ✅ **Medical NER/OpenIE Prompts** - Trích xuất thực thể và quan hệ y khoa
3. ✅ **Medical QA Prompts** - Trả lời câu hỏi y khoa với clinical reasoning
4. ✅ **Cache Management System** - Script xóa cache để re-run workflow
5. ✅ **Comprehensive Documentation** - Hướng dẫn chi tiết và troubleshooting

---

## 📁 FILES CREATED

### 1. Medical Prompts

#### A. NER và OpenIE Prompts
**File:** `gfmrag/kg_construction/openie_extraction_instructions_medical.py`

**Nội dung:**
- Medical NER instruction với entity types:
  - Diseases, disorders, medical conditions
  - Signs and symptoms
  - Anatomical structures and body parts
  - Drug names and dosages
  - Laboratory tests and values
  - Diagnostic procedures
  - Vital signs

- Medical OpenIE instruction với relationship types:
  - Patient-Condition: `has medical history of`, `diagnosed with`, `presented with`
  - Treatment: `treated with`, `prescribed`, `prescribed at`, `managed with`
  - Diagnostic: `showed`, `revealed`, `indicates`, `elevated at`
  - Anatomical: `affects`, `located in`, `placed in`, `radiates to`

- Complete medical case example:
  - 58-year-old male with acute myocardial infarction
  - Full clinical history, examination, diagnostics
  - 44 medical relationship triples

**Usage:**
```bash
cp gfmrag/kg_construction/openie_extraction_instructions_medical.py \
   gfmrag/kg_construction/openie_extraction_instructions.py
```

---

#### B. Medical QA Prompts

**File 1:** `gfmrag/workflow/config/qa_prompt/hotpotqa_medical.yaml`

**Nội dung:**
- Medical system prompt với clinical reasoning framework
- 2 comprehensive medical examples:
  - Example 1: Antiplatelet therapy in STEMI
    - 5 medical documents (MI management, aspirin, clopidogrel, etc.)
    - Question về loading doses
    - Clinical reasoning process
  - Example 2: ACE inhibitors in diabetic hypertension
    - 4 medical documents (HTN classification, lisinopril, diabetes management)
    - Question về dosing và renoprotective effects
    - Clinical reasoning process

**File 2:** `gfmrag/workflow/config/qa_prompt/zero_shot_medical.yaml`

**Nội dung:**
- Medical system prompt (zero-shot)
- "Clinical Reasoning:" thay vì "Thought:"
- "Medical Document:" thay vì "Wikipedia Title:"
- Focus on evidence-based medicine

**Usage:**
```bash
cd gfmrag/workflow/config/qa_prompt
cp hotpotqa_medical.yaml hotpotqa.yaml
cp zero_shot_medical.yaml zero_shot.yaml
```

---

### 2. Cache Management

**File:** `clear_cache.sh`

**Tính năng:**
- Interactive script với confirmation prompts
- Identifies all cache locations:
  - `gfmrag/workflow/tmp/kg_construction/`
  - `gfmrag/workflow/tmp/qa_construction/`
  - `gfmrag/workflow/outputs/`
  - `__pycache__/` directories
- Shows file counts and sizes
- Selective clearing options

**Usage:**
```bash
./clear_cache.sh
```

---

### 3. Documentation

#### A. Medical Domain Guide
**File:** `MEDICAL_DOMAIN_GUIDE.md` (600+ lines)

**Sections:**
1. Tổng quan hệ thống
2. Xóa cache và chạy lại workflow (3 methods)
3. Sử dụng medical prompts:
   - NER/OpenIE prompts (3 options)
   - QA prompts (2 options)
4. Test và verify:
   - API connection test
   - Medical data testing
   - Entity verification
   - Relationship verification
5. Troubleshooting (5 common issues)
6. Workflow overview diagram
7. Quick reference commands
8. Checklists

---

#### B. Medical Prompts Analysis
**File:** `MEDICAL_PROMPTS_ANALYSIS.md` (462 lines)

**Sections:**
1. Analysis of existing prompts
2. Medical domain proposals for each prompt type
3. Priority list of files to modify
4. Step-by-step customization guide
5. Implementation checklist

---

## 🔧 HOW TO USE

### Quick Start (3 steps):

**Step 1: Install Medical NER/OpenIE Prompts**
```bash
cd /home/user/GFM

# Backup original
cp gfmrag/kg_construction/openie_extraction_instructions.py \
   gfmrag/kg_construction/openie_extraction_instructions_generic.py.bak

# Install medical version
cp gfmrag/kg_construction/openie_extraction_instructions_medical.py \
   gfmrag/kg_construction/openie_extraction_instructions.py
```

**Step 2: Install Medical QA Prompts (Optional - for Stage 2+)**
```bash
cd gfmrag/workflow/config/qa_prompt

# Backup originals
cp hotpotqa.yaml hotpotqa_generic.yaml.bak
cp zero_shot.yaml zero_shot_generic.yaml.bak

# Install medical versions
cp hotpotqa_medical.yaml hotpotqa.yaml
cp zero_shot_medical.yaml zero_shot.yaml
```

**Step 3: Clear Cache và Run Workflow**
```bash
cd /home/user/GFM

# Clear cache
./clear_cache.sh

# Run Stage 1
python -m gfmrag.workflow.stage1_index_dataset
```

---

## 📋 FEATURE COMPARISON

### Generic vs Medical Prompts

| Feature | Generic Prompts | Medical Prompts |
|---------|----------------|-----------------|
| **Entity Types** | Generic named entities | Clinical entities (diseases, medications, symptoms, organs, tests) |
| **Relationship Types** | Generic predicates | Medical predicates (diagnosed_with, treated_with, prescribed_at, etc.) |
| **Examples** | Wikipedia articles (John Lennon, albums) | Medical case reports (MI, diabetes, hypertension) |
| **System Prompt** | "Reading comprehension assistant" | "Medical reading comprehension assistant" |
| **Reasoning Framework** | "Thought:" | "Clinical Reasoning:" |
| **Document Type** | "Wikipedia Title:" | "Medical Document:" |
| **Focus** | General knowledge | Evidence-based medicine, clinical decision-making |

---

## 🎯 MEDICAL ENTITY TYPES

Các loại thực thể y khoa được trích xuất:

### Clinical Entities:
- ✅ Diseases, disorders, medical conditions
  - Examples: "type 2 diabetes mellitus", "myocardial infarction", "hypertension"
- ✅ Signs and symptoms
  - Examples: "chest pain", "shortness of breath", "fever", "diaphoresis"
- ✅ Anatomical structures and body parts
  - Examples: "right coronary artery", "left arm", "inferior wall", "jaw"

### Medications and Treatments:
- ✅ Drug names (generic and brand)
  - Examples: "Metformin", "Lisinopril", "Aspirin", "Clopidogrel"
- ✅ Dosages and frequencies
  - Examples: "1000mg twice daily", "10mg once daily", "162-325mg"
- ✅ Treatment procedures
  - Examples: "percutaneous coronary intervention", "coronary angiography"

### Diagnostic Information:
- ✅ Laboratory tests and values
  - Examples: "Troponin I", "5.8 ng/mL", "blood glucose", "185 mg/dL"
- ✅ Diagnostic procedures
  - Examples: "12-lead ECG", "coronary angiography"
- ✅ Imaging findings
  - Examples: "ST-segment elevation", "95% stenosis"
- ✅ Vital signs
  - Examples: "blood pressure", "160/95 mmHg", "heart rate"

---

## 🔗 MEDICAL RELATIONSHIP TYPES

Các loại quan hệ y khoa được trích xuất:

### Patient-Condition Relationships:
- `has medical history of` - Medical history
- `diagnosed with` - Diagnosis
- `presented with` - Chief complaint/symptoms

### Treatment and Medication Relationships:
- `treated with` - Treatment procedures
- `prescribed` - Medication prescriptions
- `prescribed at` - Dosages
- `managed with` - Disease management

### Diagnostic Relationships:
- `showed` - Test results
- `revealed` - Diagnostic findings
- `indicates` - Clinical interpretation
- `elevated at` - Abnormal lab values
- `measured at` - Vital sign measurements

### Anatomical and Localization Relationships:
- `affects` - Disease impact on organs
- `located in` - Anatomical location
- `placed in` - Device/stent placement
- `radiates to` - Pain radiation

---

## 📊 EXAMPLE MEDICAL TRIPLES

From the medical case report in the prompts:

```json
[
  ["58-year-old male patient", "has medical history of", "type 2 diabetes mellitus"],
  ["58-year-old male patient", "has medical history of", "hypertension"],
  ["58-year-old male patient", "presented with", "chest pain"],
  ["chest pain", "radiates to", "left arm"],
  ["type 2 diabetes mellitus", "managed with", "Metformin"],
  ["Metformin", "prescribed at", "1000mg twice daily"],
  ["essential hypertension", "controlled with", "Lisinopril"],
  ["Lisinopril", "prescribed at", "10mg once daily"],
  ["12-lead ECG", "showed", "ST-segment elevation"],
  ["ST-segment elevation", "indicates", "inferior wall myocardial infarction"],
  ["Troponin I", "elevated at", "5.8 ng/mL"],
  ["coronary angiography", "revealed", "95% stenosis"],
  ["stenosis", "located in", "proximal right coronary artery"],
  ["58-year-old male patient", "treated with", "percutaneous coronary intervention"],
  ["drug-eluting stent", "placed in", "proximal right coronary artery"]
]
```

Total: 44 medical triples in example

---

## ✅ VERIFICATION CHECKLIST

### Before Running Workflow:
- [ ] YEScale API configured trong `.env`
- [ ] Test API: `python test_yescale_connection.py` → All 4 tests pass
- [ ] Medical NER/OpenIE prompts installed
- [ ] Medical QA prompts installed (optional, for Stage 2+)
- [ ] Medical dataset prepared in `data/hotpotqa/raw/`
- [ ] Cache cleared: `./clear_cache.sh`

### After Running Workflow:
- [ ] Check logs: No errors in `gfmrag/workflow/outputs/kg_construction/*/stage1_index_dataset.log`
- [ ] NER results contain medical entities (diseases, medications, symptoms)
- [ ] Triples contain medical predicates (diagnosed_with, treated_with, etc.)
- [ ] Knowledge graph has clinical meaning

---

## 🔍 TESTING

### Test API Connection:
```bash
python test_yescale_connection.py
```

Expected output:
```
✅ PASS - OpenAI SDK
✅ PASS - Raw Requests
✅ PASS - ChatGPT Class
✅ PASS - LangChain Model

Total: 4/4 tests passed
```

### Verify Medical Entities:
```bash
# Check NER results
cat gfmrag/workflow/tmp/kg_construction/*/ner_results.jsonl | head -5

# Should see medical entities like:
# "type 2 diabetes mellitus"
# "chest pain"
# "Metformin"
# "blood pressure"
```

### Verify Medical Relationships:
```bash
# Check triples
cat gfmrag/workflow/tmp/kg_construction/*/triples.jsonl | head -5

# Should see medical predicates like:
# "diagnosed_with"
# "treated_with"
# "prescribed_at"
# "has_medical_history_of"
```

---

## 📚 DOCUMENTATION FILES

1. **MEDICAL_DOMAIN_GUIDE.md** (600+ lines)
   - Complete usage guide
   - Cache management
   - Prompt installation
   - Testing and verification
   - Troubleshooting

2. **MEDICAL_PROMPTS_ANALYSIS.md** (462 lines)
   - Detailed prompt analysis
   - Medical domain proposals
   - Customization guide

3. **MEDICAL_CUSTOMIZATION_SUMMARY.md** (this file)
   - Overview of all customizations
   - Quick reference

4. **YESCALE_INTEGRATION.md**
   - YEScale API integration details
   - Technical documentation

---

## 🚀 QUICK REFERENCE

### One-liner: Clear cache and run workflow
```bash
./clear_cache.sh && python -m gfmrag.workflow.stage1_index_dataset
```

### Install all medical prompts
```bash
# NER/OpenIE prompts
cp gfmrag/kg_construction/openie_extraction_instructions_medical.py \
   gfmrag/kg_construction/openie_extraction_instructions.py

# QA prompts
cd gfmrag/workflow/config/qa_prompt
cp hotpotqa_medical.yaml hotpotqa.yaml
cp zero_shot_medical.yaml zero_shot.yaml
```

### Force recompute (không xóa cache)
```bash
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.force=True \
  qa_constructor.force=True
```

### Check results
```bash
# Logs
tail -f gfmrag/workflow/outputs/kg_construction/*/stage1_index_dataset.log

# NER results
ls -la gfmrag/workflow/tmp/kg_construction/

# Triples
ls -la gfmrag/workflow/tmp/kg_construction/
```

---

## 🎓 MEDICAL DOMAIN FOCUS

### Primary Goals:
1. ✅ **Extract medical entities** - Diseases, medications, symptoms, organs, tests
2. ✅ **Identify medical relationships** - Diagnosis, treatment, prescription, indication

### Medical Knowledge Graph Structure:
```
[Patient] --has_medical_history_of--> [Disease]
[Patient] --diagnosed_with--> [Disease]
[Patient] --presented_with--> [Symptom]
[Patient] --treated_with--> [Procedure]
[Patient] --prescribed--> [Medication]
[Medication] --prescribed_at--> [Dosage]
[Disease] --managed_with--> [Medication]
[Test] --showed--> [Finding]
[Finding] --indicates--> [Diagnosis]
[Biomarker] --elevated_at--> [Value]
[Disease] --affects--> [Organ]
[Stenosis] --located_in--> [Artery]
[Stent] --placed_in--> [Artery]
[Pain] --radiates_to--> [Body Part]
```

---

## 📞 SUPPORT

### For Cache Issues:
See: `MEDICAL_DOMAIN_GUIDE.md` → Section 5.A (Cache không bị xóa)

### For API Issues:
See: `MEDICAL_DOMAIN_GUIDE.md` → Section 5.B (YEScale API errors)

### For Extraction Issues:
See: `MEDICAL_DOMAIN_GUIDE.md` → Section 5.C-D (Medical entities/relationships)

### For Workflow Issues:
See: `MEDICAL_DOMAIN_GUIDE.md` → Section 5.E (Workflow chạy quá lâu)

---

## 🎉 SUMMARY

**Medical domain customization is COMPLETE!**

✅ Medical NER/OpenIE prompts created with clinical examples
✅ Medical QA prompts created with reasoning framework
✅ Cache management script created
✅ Comprehensive documentation written
✅ Quick start guide provided
✅ Troubleshooting guide included

**Next Steps:**
1. Install medical prompts
2. Prepare medical dataset
3. Clear cache
4. Run workflow
5. Verify results

**Refer to:** `MEDICAL_DOMAIN_GUIDE.md` for complete instructions.

---

**Last updated:** 2025-11-25
**System:** GFM-RAG with YEScale + Medical Domain
**Status:** ✅ Ready for medical data processing
