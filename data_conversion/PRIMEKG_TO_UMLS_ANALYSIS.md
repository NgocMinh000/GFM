# PrimeKG → UMLS CUI-based Triples - Phân tích khả thi

## 📊 Phân tích cấu trúc PrimeKG

### 1. Format hiện tại của PrimeKG kg.csv

```csv
x_index,x_id,x_type,x_name,x_source,relation,display_relation,y_index,y_id,y_type,y_name,y_source
```

**Ví dụ dữ liệu:**
```csv
1,MONDO:0000001,disease,Diabetes mellitus,MONDO,treats,drug_disease,500,DB00001,drug,Insulin,DrugBank
2,NCBIGene:1234,gene/protein,TP53,NCBI,interacts_with,ppi,501,NCBIGene:5678,gene/protein,BRCA1,NCBI
```

### 2. Các loại IDs trong PrimeKG

| Node Type | ID Format | Source | Example |
|-----------|-----------|--------|---------|
| Disease | `MONDO:xxxxxxx` | MONDO | MONDO:0005148 |
| Drug | `DB#####` | DrugBank | DB00001 |
| Gene/Protein | `NCBIGene:####` | NCBI | NCBIGene:7157 |
| Anatomy | `UBERON:xxxxxxx` | UBERON | UBERON:0001062 |
| Phenotype | `HP:xxxxxxx` | HPO | HP:0000118 |

**❌ VẤN ĐỀ:** `x_id` và `y_id` KHÔNG phải UMLS CUI trực tiếp!

- Diseases dùng **MONDO IDs** (format: `MONDO:0000001`)
- UMLS CUIs có format: `C` + 7 chữ số (ví dụ: `C0001234`)

---

## 🔍 Mapping UMLS CUI trong PrimeKG

### File mapping có sẵn:

1. **umls_mondo.csv** - Map giữa UMLS CUI và MONDO ID
   ```csv
   umls_id,mondo_id
   C0011849,MONDO:0005148
   C0001234,MONDO:0000001
   ```

2. **disease_features.csv** - Chứa umls_description cho diseases
   ```csv
   node_index,mondo_id,mondo_name,umls_description
   1,MONDO:0005148,type 2 diabetes mellitus,"UMLS descriptions..."
   ```

### Thống kê mapping:

- **192 UMLS concepts** về autism → **37 MONDO disease concepts**
- Tỷ lệ: ~5 UMLS CUIs map vào 1 MONDO ID (nhiều-đến-một)

---

## ✅ KHẢ THI - Có 3 chiến lược

---

## CHIẾN LƯỢC 1: Filter chỉ lấy UMLS entities (ĐƠN GIẢN)

### Cách thực hiện:

Chỉ lấy các rows có `x_source="UMLS"` hoặc `y_source="UMLS"`

### Ưu điểm:
- ✅ Đơn giản, không cần mapping
- ✅ x_id/y_id đã là UMLS CUI sẵn
- ✅ Chạy nhanh

### Nhược điểm:
- ❌ Mất rất nhiều data (diseases dùng MONDO, drugs dùng DrugBank)
- ❌ Có thể chỉ còn 5-10% triples

### Code mẫu:

```python
import pandas as pd

# Load PrimeKG
kg = pd.read_csv('kg.csv', low_memory=False)

# Filter: chỉ giữ entities có source=UMLS
umls_kg = kg[
    (kg['x_source'] == 'UMLS') | (kg['y_source'] == 'UMLS')
]

# Export triples (x_id, relation, y_id)
with open('umls_triples.txt', 'w') as f:
    for _, row in umls_kg.iterrows():
        f.write(f"{row['x_id']},{row['relation']},{row['y_id']}\n")
```

### Khi nào dùng:
- Bạn chỉ quan tâm UMLS concepts thuần túy
- Không cần diseases (vì diseases dùng MONDO)

---

## CHIẾN LƯỢC 2: Map MONDO → UMLS CUI (RECOMMENDED)

### Cách thực hiện:

1. Download PrimeKG kg.csv
2. Download umls_mondo.csv (hoặc clone repo để get file)
3. Map tất cả MONDO IDs → UMLS CUIs
4. Replace x_id/y_id với CUIs
5. Export triples

### Ưu điểm:
- ✅ Giữ được nhiều data nhất (diseases mapped)
- ✅ Output toàn bộ là UMLS CUIs
- ✅ Tích hợp tốt với Stage 3 UMLS Mapping

### Nhược điểm:
- ⚠️ Phức tạp hơn, cần mapping logic
- ⚠️ Một số entities không map được (drugs, genes)
- ⚠️ Nhiều UMLS CUIs map vào 1 MONDO (mất uniqueness)

### Quy trình chi tiết:

```
1. Load kg.csv
2. Load umls_mondo.csv
3. For each row:
   a. If x_id is MONDO:xxxxx → lookup umls_mondo.csv → get UMLS CUI
   b. If y_id is MONDO:xxxxx → lookup umls_mondo.csv → get UMLS CUI
   c. If không map được → skip hoặc giữ nguyên ID
4. Export: CUI,relation,CUI
```

### Code outline:

```python
import pandas as pd

# Load data
kg = pd.read_csv('kg.csv', low_memory=False)
umls_mondo = pd.read_csv('umls_mondo.csv')

# Create mapping dict: MONDO ID → UMLS CUI
mondo_to_umls = dict(zip(umls_mondo['mondo_id'], umls_mondo['umls_id']))

# Map function
def map_to_cui(id_value, source):
    if source == 'MONDO' and id_value in mondo_to_umls:
        return mondo_to_umls[id_value]
    elif source == 'UMLS':
        return id_value
    else:
        return None  # or keep original

# Apply mapping
kg['x_cui'] = kg.apply(lambda r: map_to_cui(r['x_id'], r['x_source']), axis=1)
kg['y_cui'] = kg.apply(lambda r: map_to_cui(r['y_id'], r['y_source']), axis=1)

# Filter: chỉ giữ rows có cả 2 CUIs
kg_cui = kg.dropna(subset=['x_cui', 'y_cui'])

# Export
with open('umls_cui_triples.txt', 'w') as f:
    for _, row in kg_cui.iterrows():
        f.write(f"{row['x_cui']},{row['relation']},{row['y_cui']}\n")
```

### Khi nào dùng:
- Bạn muốn giữ nhiều data nhất có thể
- Bạn cần diseases mapped sang UMLS
- Bạn có thể accept một số entities không có CUI

---

## CHIẾN LƯỢC 3: Hybrid - Map tất cả sang UMLS (TOÀN DIỆN)

### Cách thực hiện:

Map tất cả ontology IDs sang UMLS bằng nhiều mapping files:
- MONDO → UMLS (umls_mondo.csv)
- DrugBank → UMLS (từ UMLS hoặc mapping files khác)
- NCBI Gene → UMLS (từ UMLS hoặc BioThings API)
- HPO → UMLS
- ...

### Ưu điểm:
- ✅ Toàn diện nhất
- ✅ Output 100% UMLS CUIs
- ✅ Không mất data

### Nhược điểm:
- ❌ Rất phức tạp, cần nhiều mapping files
- ❌ Tốn thời gian implement
- ❌ Một số IDs không tìm được mapping

### Khi nào dùng:
- Bạn cần output toàn bộ UMLS
- Bạn có thời gian tạo/tìm mapping files
- Research project yêu cầu standardization hoàn toàn

---

## 🎯 KHUYẾN NGHỊ - Chọn chiến lược nào?

### 👉 Nếu bạn muốn NHANH và ĐƠN GIẢN:
**→ Dùng CHIẾN LƯỢC 1** (Filter UMLS only)

Nhưng lưu ý: Sẽ mất nhiều data

### 👉 Nếu bạn muốn CÂN BẰNG (RECOMMENDED):
**→ Dùng CHIẾN LƯỢC 2** (Map MONDO → UMLS)

- Giữ được diseases (quan trọng nhất)
- Code không quá phức tạp
- Phù hợp với Stage 3 UMLS Mapping pipeline

### 👉 Nếu bạn cần TOÀN DIỆN:
**→ Dùng CHIẾN LƯỢC 3** (Map tất cả)

Nhưng cần nhiều thời gian và resources

---

## 📋 CÁC BƯỚC THỰC HIỆN (Chiến lược 2 - RECOMMENDED)

### Bước 1: Chuẩn bị data

```bash
cd /home/user/GFM/data_conversion/primekg_analysis

# Download PrimeKG (nếu có network)
wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620

# Hoặc clone repo để get mapping files
git clone https://github.com/mims-harvard/PrimeKG.git
cd PrimeKG/datasets/data/umls
# Copy umls_mondo.csv
```

### Bước 2: Tạo converter script

```bash
cd /home/user/GFM/data_conversion
# Tôi sẽ tạo file primekg_to_umls_triples.py
```

### Bước 3: Run conversion

```bash
python primekg_to_umls_triples.py kg.csv umls_mondo.csv output/primekg_umls_triples.txt
```

### Bước 4: Validate output

```bash
python validate_triples.py output/primekg_umls_triples.txt
```

### Bước 5: Use in GFM pipeline

```bash
cp output/primekg_umls_triples.txt /home/user/GFM/data/kg.txt
cd /home/user/GFM
python -m gfmrag.workflow.stage1_index_dataset
```

---

## 📊 Ước tính kết quả (Chiến lược 2)

| Metric | Before | After |
|--------|--------|-------|
| Total triples | 4,050,249 | ~500,000 - 1,000,000 |
| With UMLS CUI | ~10-20% | 100% |
| Diseases mapped | 17,080 MONDO | ~15,000 UMLS CUI |
| Coverage | Full | Diseases + some genes |

**Lưu ý:** Drugs, anatomy, pathways sẽ bị loại bỏ nếu không có UMLS mapping.

---

## ⚠️ THÁCH THỨC

### 1. Nhiều-đến-một mapping
- 192 UMLS CUIs → 37 MONDO IDs
- Khi map ngược, chọn CUI nào?
- **Giải pháp:** Chọn preferred CUI hoặc lấy first match

### 2. Không phải tất cả entities có UMLS CUI
- Genes: Một số có, một số không
- Drugs: DrugBank IDs không phải UMLS
- Pathways: Reactome IDs không có UMLS
- **Giải pháp:** Accept data loss hoặc dùng Chiến lược 3

### 3. Data quality
- Mapping có thể không chính xác 100%
- Cần validate output
- **Giải pháp:** Dùng validate_triples.py và manual check sample

---

## 🚀 NEXT STEP

Bạn muốn tôi:

**Option A:** Implement Chiến lược 1 (Filter UMLS only) - NHANH
**Option B:** Implement Chiến lược 2 (Map MONDO→UMLS) - RECOMMENDED
**Option C:** Phân tích thêm để quyết định?

Hãy cho tôi biết bạn muốn đi theo hướng nào, tôi sẽ viết code cụ thể!

---

**Created:** 2026-01-09
**Author:** GFM-RAG Team
