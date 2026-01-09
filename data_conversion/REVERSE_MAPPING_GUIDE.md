# Reverse MONDO → UMLS Mapping Guide

## 🎯 Câu hỏi: Có thể chuyển ngược từ MONDO sang UMLS không?

### ✅ Câu trả lời: CÓ THỂ - Đã có sẵn trong code!

---

## 📊 Hiểu về Mapping Direction

### File gốc: `umls_mondo.csv` (từ PrimeKG)

```csv
umls_id,mondo_id
C0011849,MONDO:0005148
C0004096,MONDO:0004975
C0007097,MONDO:0004992
```

**Direction:** UMLS → MONDO (forward)
- Column 1: UMLS CUI (source)
- Column 2: MONDO ID (target)

---

### Reverse Direction: MONDO → UMLS

**Đây là điều bạn cần!**

```csv
mondo_id,umls_id
MONDO:0005148,C0011849
MONDO:0004975,C0004096
MONDO:0004992,C0007097
```

**Direction:** MONDO → UMLS (reverse)
- Column 1: MONDO ID (source)
- Column 2: UMLS CUI (target)

---

## 🔧 3 CÁCH THỰC HIỆN

### CÁCH 1: Dùng `primekg_to_umls_triples.py` (KHUYẾN NGHỊ) ⭐

**Code này ĐÃ TỰ ĐỘNG reverse mapping!**

```bash
python primekg_to_umls_triples.py kg.csv output.txt \
    --mapping umls_mondo.csv \
    --strategy map
```

**Logic bên trong:**
```python
# Code tự động reverse!
df = pd.read_csv('umls_mondo.csv')

# Create MONDO→UMLS mapping (reverse!)
mondo_to_umls = dict(zip(df['mondo_id'], df['umls_id']))
#                         ^^^^^^^^^^^^^^  ^^^^^^^^^^
#                         MONDO (key)     UMLS (value)

# Apply to PrimeKG
for row in kg_df:
    if row['x_id'].startswith('MONDO:'):
        row['x_cui'] = mondo_to_umls[row['x_id']]
```

**Output:**
```
C0011849,treats,C0004096
C0020538,associated_with,C0007097
```

**→ Không cần làm gì thêm, code đã handle reverse!**

---

### CÁCH 2: Tạo file `mondo_to_umls.csv` riêng

Nếu bạn muốn file reverse riêng biệt:

```bash
python reverse_umls_mondo_mapping.py \
    umls_mondo.csv \
    mondo_to_umls.csv
```

**Output:** `mondo_to_umls.csv`
```csv
mondo_id,umls_id
MONDO:0000001,C0001234
MONDO:0000002,C0005678
MONDO:0005148,C0011849
```

**Sử dụng:**
```python
import pandas as pd

# Load reverse mapping
mondo_to_umls = pd.read_csv('mondo_to_umls.csv')
mapping_dict = dict(zip(mondo_to_umls['mondo_id'], mondo_to_umls['umls_id']))

# Convert MONDO ID to UMLS CUI
mondo_id = "MONDO:0005148"
umls_cui = mapping_dict[mondo_id]  # → "C0011849"
```

---

### CÁCH 3: Manual reverse với pandas (1 dòng lệnh)

```bash
# Quick one-liner
python -c "import pandas as pd; df=pd.read_csv('umls_mondo.csv'); df[['mondo_id','umls_id']].drop_duplicates('mondo_id').to_csv('mondo_to_umls.csv', index=False)"
```

---

## ⚠️ Lưu ý quan trọng: Many-to-One Problem

### Vấn đề:

**UMLS → MONDO:** Many-to-one (nhiều-đến-một)
- 192 UMLS CUIs về autism → 37 MONDO disease concepts
- Tỷ lệ: ~5 UMLS : 1 MONDO

**Ví dụ:**
```
C0001234 → MONDO:0005148
C0005678 → MONDO:0005148  # Same MONDO!
C0009012 → MONDO:0005148  # Same MONDO!
```

### Khi reverse (MONDO → UMLS):

**Chọn CUI nào?**
- Option 1: Lấy first match ✅ (default trong code của tôi)
- Option 2: Lấy "preferred" CUI (nếu có metadata)
- Option 3: Tạo list of all CUIs

**Code xử lý:**
```python
# Default: Keep first
df_reversed.drop_duplicates(subset=['mondo_id'], keep='first')

# Output:
# MONDO:0005148 → C0001234 (first in list)
```

**Kết quả:**
- Input: 15,000-20,000 UMLS→MONDO mappings
- Output: ~5,000-7,000 unique MONDO→UMLS mappings (sau dedup)

---

## 📋 Example Workflow

### Scenario: Convert PrimeKG diseases to UMLS

**Step 1: Get mapping file**
```bash
./download_primekg_mapping.sh primekg_analysis
# → Downloads umls_mondo.csv
```

**Step 2: (Optional) Create reverse file**
```bash
python reverse_umls_mondo_mapping.py \
    primekg_analysis/umls_mondo.csv \
    primekg_analysis/mondo_to_umls.csv
```

**Step 3: Convert PrimeKG**
```bash
# Auto reverse (recommended)
python primekg_to_umls_triples.py \
    primekg_analysis/kg.csv \
    primekg_analysis/output/umls_triples.txt \
    --mapping primekg_analysis/umls_mondo.csv \
    --strategy map
```

**Step 4: Verify**
```bash
# Check output
head -20 primekg_analysis/output/umls_triples.txt

# Should see UMLS CUIs (C#######)
C0011849,treats,C0004096
C0020538,associated_with,C0007097
```

---

## 🔍 Verify Mapping Quality

### Test reverse mapping:

```python
import pandas as pd

# Load original
umls_mondo = pd.read_csv('umls_mondo.csv')
print(f"Original UMLS→MONDO: {len(umls_mondo)} mappings")

# Load reversed
mondo_umls = pd.read_csv('mondo_to_umls.csv')
print(f"Reversed MONDO→UMLS: {len(mondo_umls)} mappings")

# Check sample
sample_mondo = "MONDO:0005148"
umls_cui = mondo_umls[mondo_umls['mondo_id'] == sample_mondo]['umls_id'].values[0]
print(f"{sample_mondo} → {umls_cui}")

# Verify reverse works
original_pairs = umls_mondo[umls_mondo['mondo_id'] == sample_mondo]
print(f"Original UMLS CUIs for {sample_mondo}:")
print(original_pairs['umls_id'].tolist())
```

**Expected output:**
```
Original UMLS→MONDO: 15423 mappings
Reversed MONDO→UMLS: 5892 mappings
MONDO:0005148 → C0011849
Original UMLS CUIs for MONDO:0005148:
['C0011849', 'C0011860', 'C0011854', 'C0011853', 'C0011862']
                      ^^^^^^^^
                      Picked first
```

---

## 📊 Comparison: Direct vs Reverse

| Aspect | umls_mondo.csv | mondo_to_umls.csv |
|--------|----------------|-------------------|
| Direction | UMLS → MONDO | MONDO → UMLS |
| Rows | 15,000-20,000 | 5,000-7,000 |
| Use case | Map UMLS to diseases | Map diseases to UMLS |
| Duplicates | None | Removed (many UMLS→1 MONDO) |
| Source | PrimeKG repo | Generated from umls_mondo.csv |

---

## 💡 Which Approach to Use?

### ✅ Use CÁCH 1 (Auto reverse in primekg_to_umls_triples.py) if:
- Bạn muốn convert PrimeKG → UMLS triples
- Bạn muốn tự động hóa hoàn toàn
- Bạn OK với first-match strategy

### ✅ Use CÁCH 2 (Create mondo_to_umls.csv) if:
- Bạn cần file mapping riêng cho reference
- Bạn muốn inspect mapping trước khi dùng
- Bạn cần reuse mapping cho nhiều tasks

### ✅ Use CÁCH 3 (Manual pandas) if:
- Bạn muốn quick test
- Bạn biết pandas
- One-time conversion

---

## 🚀 Quick Start

### Fastest way (1 command):

```bash
# Download + Convert in one go
python primekg_to_umls_triples.py \
    primekg_analysis/kg.csv \
    primekg_analysis/output/umls_triples.txt \
    --mapping primekg_analysis/umls_mondo.csv \
    --strategy map

# Code tự động:
# 1. Load umls_mondo.csv
# 2. Reverse mapping MONDO→UMLS
# 3. Replace all MONDO IDs
# 4. Output UMLS CUI triples
```

**Done!** ✅

---

## ❓ FAQ

### Q: File umls_mondo.csv ở đâu?

**A:** Download từ PrimeKG repo:
```bash
./download_primekg_mapping.sh primekg_analysis
```

Hoặc:
```bash
git clone https://github.com/mims-harvard/PrimeKG.git
cp PrimeKG/datasets/data/umls/umls_mondo.csv .
```

---

### Q: Tại sao reverse có ít rows hơn?

**A:** Vì many-to-one:
- 5 UMLS CUIs → 1 MONDO ID
- Khi reverse: 1 MONDO ID → 1 UMLS CUI (picked first)
- Giảm từ 15K xuống 6K rows

---

### Q: Làm sao biết chọn đúng CUI?

**A:** Có 3 strategies:
1. **First match** (default) - Nhanh, đơn giản
2. **Preferred CUI** - Cần UMLS metadata (chính xác hơn)
3. **Most common** - Dựa trên frequency trong corpus

Code hiện tại dùng strategy 1 (first match).

---

### Q: Có mất data không?

**A:** Có, nhưng chấp nhận được:
- Input: 192 autism-related UMLS CUIs
- Output: 37 autism MONDO IDs
- Reverse: 37 MONDO → 37 UMLS CUIs (chọn 1 trong 192)
- **Data loss:** Semantic information preserved, just one representative CUI per MONDO

---

### Q: Drugs và genes thì sao?

**A:**
- **Diseases:** MONDO → UMLS ✅ (có mapping)
- **Drugs:** DrugBank IDs → Không có UMLS mapping ❌
- **Genes:** NCBI Gene IDs → Một số có UMLS mapping ⚠️

**Solution:** Use Strategy 2B (--keep-unmapped) để giữ drugs/genes.

---

## 📚 References

- PrimeKG map_umls_mondo.py source code
- UMLS Metathesaurus documentation
- MONDO Disease Ontology

---

**Created:** 2026-01-09
**Version:** 1.0.0
**Author:** GFM-RAG Team

---

## TÓM TẮT

**Câu hỏi:** Có thể reverse MONDO→UMLS không?

**Trả lời:** ✅ **CÓ** - Code đã implement sẵn!

**Cách dùng:**
```bash
python primekg_to_umls_triples.py kg.csv output.txt \
    --mapping umls_mondo.csv --strategy map
```

**Lưu ý:** Many-to-one mapping → chọn first match

**Kết quả:** MONDO IDs thành UMLS CUIs hoàn toàn tự động!
