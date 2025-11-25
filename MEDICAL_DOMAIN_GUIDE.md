# HƯỚNG DẪN SỬ DỤNG DOMAIN Y TẾ - GFM-RAG

## 📋 MỤC LỤC
1. [Tổng quan](#tổng-quan)
2. [Xóa Cache và Chạy lại Workflow](#xóa-cache)
3. [Sử dụng Medical Prompts](#sử-dụng-medical-prompts)
4. [Test và Verify](#test-và-verify)
5. [Troubleshooting](#troubleshooting)

---

## 1. TỔNG QUAN

Hệ thống GFM-RAG đã được tích hợp YEScale API và customize cho domain Y TẾ.

### Mục đích chính:
- ✅ **Trích xuất thực thể y khoa**: Bệnh, thuốc, triệu chứng, cơ quan, xét nghiệm
- ✅ **Xác định quan hệ y khoa**: Chẩn đoán, điều trị, chỉ định, ảnh hưởng

### Files quan trọng:
```
GFM/
├── gfmrag/
│   ├── kg_construction/
│   │   ├── openie_extraction_instructions.py          # Generic prompts (gốc)
│   │   └── openie_extraction_instructions_medical.py  # Medical NER/OpenIE prompts (MỚI)
│   └── workflow/config/qa_prompt/
│       ├── hotpotqa.yaml                              # Generic QA prompt (gốc)
│       ├── hotpotqa_medical.yaml                      # Medical QA prompt (MỚI)
│       ├── zero_shot.yaml                             # Generic zero-shot (gốc)
│       └── zero_shot_medical.yaml                     # Medical zero-shot (MỚI)
├── clear_cache.sh                                      # Script xóa cache
├── MEDICAL_DOMAIN_GUIDE.md                            # Guide này
└── MEDICAL_PROMPTS_ANALYSIS.md                        # Phân tích chi tiết
```

---

## 2. XÓA CACHE VÀ CHẠY LẠI WORKFLOW

### A. Hiểu về Cache System

Hệ thống lưu cache ở các vị trí sau:
1. **KG Construction cache**: `gfmrag/workflow/tmp/kg_construction/`
2. **QA Construction cache**: `gfmrag/workflow/tmp/qa_construction/`
3. **Workflow outputs**: `gfmrag/workflow/outputs/`
4. **Python cache**: `__pycache__/` directories

**Lý do có cache:**
- Giảm thời gian xử lý khi chạy lại
- Tránh gọi API nhiều lần cho cùng data
- Cache được tạo dựa trên hash của config

**Khi nào cần xóa cache:**
- ✅ Thay đổi prompts
- ✅ Thay đổi config
- ✅ Thay đổi dataset
- ✅ Muốn force re-run từ đầu

---

### B. Phương pháp 1: Sử dụng Script (RECOMMENDED)

**Bước 1: Chạy script xóa cache**
```bash
cd /home/user/GFM
./clear_cache.sh
```

**Script sẽ hỏi từng bước:**
```
================================================
GFM-RAG Cache Clearing Script
================================================

Found: KG Construction Cache
  Location: gfmrag/workflow/tmp/kg_construction
  Files: 25
  Size: 1.2M
  Remove this cache? (y/N): y
✓ Removed

Found: QA Construction Cache
  Location: gfmrag/workflow/tmp/qa_construction
  Files: 10
  Size: 512K
  Remove this cache? (y/N): y
✓ Removed

... (các cache khác)
```

**Bước 2: Chạy lại workflow**
```bash
python -m gfmrag.workflow.stage1_index_dataset
```

---

### C. Phương pháp 2: Xóa thủ công

```bash
# Xóa KG cache
rm -rf gfmrag/workflow/tmp/kg_construction/*

# Xóa QA cache
rm -rf gfmrag/workflow/tmp/qa_construction/*

# Xóa workflow outputs
rm -rf gfmrag/workflow/outputs/*

# Xóa Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
```

---

### D. Phương pháp 3: Override Config (Không xóa cache)

Chạy với `force=True` để bỏ qua cache:

```bash
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.force=True \
  qa_constructor.force=True
```

**Lưu ý:** Cách này vẫn giữ cache cũ nhưng tính toán lại.

---

## 3. SỬ DỤNG MEDICAL PROMPTS

### A. Option 1: Replace toàn bộ (RECOMMENDED cho Y tế)

**Bước 1: Backup file gốc**
```bash
cd /home/user/GFM
cp gfmrag/kg_construction/openie_extraction_instructions.py \
   gfmrag/kg_construction/openie_extraction_instructions_generic.py.bak
```

**Bước 2: Replace bằng medical version**
```bash
cp gfmrag/kg_construction/openie_extraction_instructions_medical.py \
   gfmrag/kg_construction/openie_extraction_instructions.py
```

**Bước 3: Xóa cache và chạy lại**
```bash
./clear_cache.sh
python -m gfmrag.workflow.stage1_index_dataset
```

---

### B. Option 2: Symbolic link (Linh hoạt switch)

**Setup:**
```bash
cd gfmrag/kg_construction

# Backup gốc
mv openie_extraction_instructions.py openie_extraction_instructions_generic.py

# Tạo symlink đến medical
ln -s openie_extraction_instructions_medical.py openie_extraction_instructions.py
```

**Switch về generic:**
```bash
rm openie_extraction_instructions.py
ln -s openie_extraction_instructions_generic.py openie_extraction_instructions.py
```

**Switch về medical:**
```bash
rm openie_extraction_instructions.py
ln -s openie_extraction_instructions_medical.py openie_extraction_instructions.py
```

---

### C. Option 3: Edit trực tiếp (Manual)

**Nếu muốn customize thêm:**
```bash
vim gfmrag/kg_construction/openie_extraction_instructions.py
```

**Sửa các phần:**
1. **Line 70-74**: `ner_instruction` - Hướng dẫn NER
2. **Line 54-64**: `one_shot_passage` - Example passage
3. **Line 61-64**: `one_shot_passage_entities` - Example entities
4. **Line 128-139**: `openie_post_ner_instruction` - Hướng dẫn OpenIE
5. **Line 110-125**: `one_shot_passage_triples` - Example triples

**Refer to:** `MEDICAL_PROMPTS_ANALYSIS.md` để xem full proposals.

---

### D. Sử dụng Medical QA Prompts (Optional)

Ngoài NER và OpenIE prompts, bạn cũng có thể customize QA prompts cho domain y tế.

**Files đã được tạo:**
- `gfmrag/workflow/config/qa_prompt/hotpotqa_medical.yaml` - Với medical examples
- `gfmrag/workflow/config/qa_prompt/zero_shot_medical.yaml` - Zero-shot medical prompt

**Cách sử dụng:**

**Option 1: Backup và replace (Recommended)**
```bash
cd /home/user/GFM/gfmrag/workflow/config/qa_prompt

# Backup originals
cp hotpotqa.yaml hotpotqa_generic.yaml.bak
cp zero_shot.yaml zero_shot_generic.yaml.bak

# Replace với medical versions
cp hotpotqa_medical.yaml hotpotqa.yaml
cp zero_shot_medical.yaml zero_shot.yaml
```

**Option 2: Chỉ định trực tiếp trong config**
```bash
# Chỉnh sửa config để dùng medical prompts
python -m gfmrag.workflow.stage1_index_dataset \
  qa_constructor.prompt_name=hotpotqa_medical
```

**Medical QA Prompts Features:**
- ✅ Clinical reasoning framework ("Clinical Reasoning:" thay vì "Thought:")
- ✅ Medical examples về antiplatelet therapy, diabetes management
- ✅ Terminology: "Medical Document" thay vì "Wikipedia Title"
- ✅ Focus on evidence-based medicine và clinical decision-making

**Example medical prompt:**
```yaml
system_prompt: 'As an advanced medical reading comprehension assistant,
  your task is to analyze clinical documents and answer medical questions
  with accuracy and clinical reasoning...'
```

**Khi nào cần dùng Medical QA prompts:**
- ✅ Khi chạy Stage 2+ (QA construction và inference)
- ✅ Khi test với medical Q&A pairs
- ✅ Khi cần câu trả lời có clinical reasoning
- ❌ Không cần cho Stage 1 (chỉ KG construction)

---

## 4. TEST VÀ VERIFY

### A. Test API Connection

**Trước khi chạy workflow, test YEScale API:**
```bash
python test_yescale_connection.py
```

**Expected output:**
```
✅ PASS - OpenAI SDK
✅ PASS - Raw Requests
✅ PASS - ChatGPT Class
✅ PASS - LangChain Model

Total: 4/4 tests passed
```

---

### B. Test với Medical Data

**Bước 1: Chuẩn bị medical dataset**

Đặt medical documents trong:
```
data/hotpotqa/raw/
├── dataset_corpus.json    # Medical documents
├── train.json             # Medical Q&A pairs
└── test.json             # Medical Q&A pairs
```

**Format document:**
```json
{
  "id": "doc_001",
  "title": "Type 2 Diabetes Mellitus Management",
  "text": "Type 2 diabetes mellitus is characterized by...",
  "url": "medical_source"
}
```

**Bước 2: Chạy workflow**
```bash
# Xóa cache trước
./clear_cache.sh

# Chạy stage 1
python -m gfmrag.workflow.stage1_index_dataset
```

**Bước 3: Check outputs**
```bash
# Xem logs
tail -f gfmrag/workflow/outputs/kg_construction/*/stage1_index_dataset.log

# Xem KG results
ls -la gfmrag/workflow/tmp/kg_construction/

# Check NER results
cat gfmrag/workflow/tmp/kg_construction/*/ner_results.jsonl | head -5

# Check triples
cat gfmrag/workflow/tmp/kg_construction/*/triples.jsonl | head -5
```

---

### C. Verify Medical Entities

**Script để check quality:**
```python
import json

# Load NER results
with open('gfmrag/workflow/tmp/kg_construction/.../ner_results.jsonl') as f:
    for line in f:
        data = json.loads(line)
        entities = data['entities']

        # Count entity types
        medical_keywords = ['disease', 'medication', 'symptom', 'mg', 'patient']
        medical_entities = [e for e in entities if any(kw in e.lower() for kw in medical_keywords)]

        print(f"Document: {data['id']}")
        print(f"  Total entities: {len(entities)}")
        print(f"  Medical entities: {len(medical_entities)}")
        print(f"  Sample: {medical_entities[:5]}")
        print()
```

---

### D. Verify Medical Relationships

```python
import json
from collections import Counter

# Load triples
triples = []
with open('gfmrag/workflow/tmp/kg_construction/.../triples.jsonl') as f:
    for line in f:
        data = json.loads(line)
        triples.extend(data['triples'])

# Count predicates
predicates = [t[1] for t in triples]
predicate_counts = Counter(predicates)

print("Top 20 Predicates:")
for pred, count in predicate_counts.most_common(20):
    print(f"  {pred}: {count}")

# Medical predicates to look for
medical_preds = [
    'diagnosed_with', 'treated_with', 'prescribed_at',
    'has_medical_history_of', 'managed_with', 'showed',
    'elevated_at', 'indicates'
]

print("\nMedical Predicates Found:")
for pred in medical_preds:
    count = predicate_counts.get(pred, 0)
    status = "✅" if count > 0 else "❌"
    print(f"  {status} {pred}: {count}")
```

---

## 5. TROUBLESHOOTING

### A. Cache không bị xóa

**Vấn đề:** Sau khi xóa cache, workflow vẫn không chạy lại

**Giải pháp:**
```bash
# Option 1: Xóa toàn bộ tmp directory
rm -rf gfmrag/workflow/tmp/

# Option 2: Force recompute
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.force=True \
  qa_constructor.force=True

# Option 3: Thay đổi dataset path trong config
python -m gfmrag.workflow.stage1_index_dataset \
  dataset.data_name=hotpotqa_medical
```

---

### B. YEScale API errors

**Vấn đề:** API returns 400/404 errors

**Check:**
```bash
# 1. Verify .env
cat .env | grep YESCALE

# Expected:
# YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completions
# YESCALE_API_KEY=sk-xxx

# 2. Test API
python test_yescale_connection.py

# 3. Check logs
tail -f gfmrag/workflow/outputs/kg_construction/*/stage1_index_dataset.log | grep -i error
```

---

### C. Không extract được medical entities

**Vấn đề:** NER results không chứa medical entities

**Check prompts:**
```bash
# 1. Verify đang dùng medical prompts
python -c "
from gfmrag.kg_construction.openie_extraction_instructions import ner_instruction
print(ner_instruction[:200])
"

# Should contain: "MEDICAL named entities"

# 2. If not, replace với medical version
cp gfmrag/kg_construction/openie_extraction_instructions_medical.py \
   gfmrag/kg_construction/openie_extraction_instructions.py

# 3. Clear cache và chạy lại
./clear_cache.sh
python -m gfmrag.workflow.stage1_index_dataset
```

---

### D. Medical relationships không đúng

**Vấn đề:** Triples có predicates generic (như "is", "has") thay vì medical

**Fix:**
1. Check `openie_post_ner_instruction` có mention medical predicates không
2. Verify example triples có medical relationships
3. Xem `MEDICAL_PROMPTS_ANALYSIS.md` section 2 để xem expected predicates

**Expected medical predicates:**
- `diagnosed_with`, `treated_with`, `prescribed_at`
- `has_medical_history_of`, `managed_with`
- `showed`, `revealed`, `indicates`
- `elevated_at`, `measured_at`
- `located_in`, `affects`, `radiates_to`

---

### E. Workflow chạy quá lâu

**Vấn đề:** Stage 1 chạy hàng giờ

**Nguyên nhân:**
- Dataset quá lớn
- Gọi YEScale API cho mỗi document
- Retry logic khi API fail

**Giảm thời gian:**
```bash
# 1. Giảm số documents
head -100 data/hotpotqa/raw/dataset_corpus.json > data/hotpotqa/raw/dataset_corpus_small.json

# 2. Tăng num_processes
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.num_processes=20 \
  qa_constructor.num_processes=20

# 3. Cache đúng cách (đừng clear cache khi không cần)
# Chỉ clear cache khi:
# - Thay đổi prompts
# - Thay đổi config
# - Thay đổi data
```

---

## 6. WORKFLOW OVERVIEW

### Luồng chạy hoàn chỉnh:

```
1. SETUP
   ├─► .env configured (YESCALE_API_BASE_URL, YESCALE_API_KEY)
   ├─► Medical prompts installed (openie_extraction_instructions.py)
   └─► Medical dataset prepared (data/hotpotqa/)

2. CLEAR CACHE (if needed)
   └─► ./clear_cache.sh

3. RUN STAGE 1
   └─► python -m gfmrag.workflow.stage1_index_dataset
       │
       ├─► NER: Extract medical entities
       │   └─► Output: ner_results.jsonl
       │
       ├─► OpenIE: Extract medical relationships
       │   └─► Output: triples.jsonl
       │
       └─► Entity Linking: Link to KB
           └─► Output: Knowledge Graph

4. VERIFY RESULTS
   ├─► Check logs
   ├─► Inspect entities
   └─► Analyze relationships

5. ITERATE (if needed)
   ├─► Adjust prompts
   ├─► Clear cache
   └─► Re-run
```

---

## 7. QUICK REFERENCE

### Xóa cache và chạy lại (One-liner):
```bash
./clear_cache.sh && python -m gfmrag.workflow.stage1_index_dataset
```

### Test API:
```bash
python test_yescale_connection.py
```

### Switch to medical NER/OpenIE prompts:
```bash
cp gfmrag/kg_construction/openie_extraction_instructions_medical.py \
   gfmrag/kg_construction/openie_extraction_instructions.py
```

### Switch to medical QA prompts (optional):
```bash
cd gfmrag/workflow/config/qa_prompt
cp hotpotqa_medical.yaml hotpotqa.yaml
cp zero_shot_medical.yaml zero_shot.yaml
```

### Force recompute:
```bash
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.force=True \
  qa_constructor.force=True
```

### Check results:
```bash
ls -la gfmrag/workflow/tmp/kg_construction/
ls -la gfmrag/workflow/tmp/qa_construction/
tail -f gfmrag/workflow/outputs/kg_construction/*/stage1_index_dataset.log
```

---

## 8. CHECKLISTS

### ✅ Trước khi chạy workflow:
- [ ] YEScale API configured trong `.env`
- [ ] Test API connection pass (`python test_yescale_connection.py`)
- [ ] Medical NER/OpenIE prompts đã được install
- [ ] Medical QA prompts đã được install (optional, for Stage 2+)
- [ ] Medical dataset đã prepared
- [ ] Cache đã được clear (nếu cần)

### ✅ Sau khi chạy workflow:
- [ ] Check logs không có errors
- [ ] NER results chứa medical entities
- [ ] Triples chứa medical predicates
- [ ] Knowledge graph có ý nghĩa y khoa

---

## 9. SUPPORT

**Files tham khảo:**
- `MEDICAL_PROMPTS_ANALYSIS.md`: Phân tích chi tiết prompts
- `YESCALE_INTEGRATION.md`: Hướng dẫn YEScale API
- `test_yescale_connection.py`: Test script

**Common commands:**
```bash
# Test API
python test_yescale_connection.py

# Clear cache
./clear_cache.sh

# Run workflow
python -m gfmrag.workflow.stage1_index_dataset

# Force recompute
python -m gfmrag.workflow.stage1_index_dataset \
  kg_constructor.force=True qa_constructor.force=True

# Check logs
tail -f gfmrag/workflow/outputs/kg_construction/*/stage1_index_dataset.log
```

---

**Last updated:** 2025-11-25
**System:** GFM-RAG with YEScale + Medical Domain
**Purpose:** Complete guide for medical domain usage
