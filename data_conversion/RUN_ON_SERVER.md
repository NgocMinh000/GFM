# Hướng dẫn chạy trên Server

## 📋 Prerequisites

Bạn cần có 2 files trên server:

1. **mondo.obo** (~130-150 MB) - Upload vào `primekg_data/mondo.obo`
2. **kg.csv** (~936 MB) - Tải từ Harvard Dataverse

## 🚀 Các bước thực hiện

### Bước 1: Clone repo (nếu chưa có)

```bash
git clone -b claude/analyze-stage3-umls-mapping-0cGgL https://github.com/NgocMinh000/GFM.git
cd GFM/data_conversion
```

### Bước 2: Upload mondo.obo

```bash
# Từ máy local, SCP lên server:
scp mondo.obo <user>@<server>:/home/user/GFM/data_conversion/primekg_data/mondo.obo

# Verify:
ls -lh primekg_data/mondo.obo
# Kết quả mong đợi: ~130-150 MB
```

### Bước 3: Test mondo.obo (Optional)

```bash
python test_mondo_obo.py primekg_data/mondo.obo
```

**Kết quả mong đợi:**
```
✓ Found mondo.obo: 138.5 MB

Testing first 1000 lines...

Results from first 1000 lines:
  [Term] blocks: 45
  xref: UMLS: 12
  skos:exactMatch UMLS: 23
✅ File looks good! UMLS cross-references detected.
```

### Bước 4: Parse mondo.obo → tạo umls_mondo.csv

```bash
python create_umls_mondo_mapping.py
```

**Output:**
- File: `primekg_data/umls_mondo.csv`
- Size: ~1-2 MB
- Format: `mondo_id,umls_id`

**Kết quả mong đợi:**
```
Parsing MONDO .obo file...
  Processed 100,000 lines...
  Processed 200,000 lines...
  ...

Extracted 15,234 MONDO→UMLS references
Final mapping: 14,567 unique MONDO→UMLS mappings
  - Unique MONDO IDs: 12,345
  - Unique UMLS CUIs: 13,890

Sample mappings:
  MONDO:0000001 → C0012634
  MONDO:0000005 → C0004096
  ...

✅ UMLS-MONDO Mapping Created Successfully!
```

### Bước 5: Download kg.csv (nếu chưa có)

```bash
cd primekg_data
wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620

# Verify:
ls -lh kg.csv
# Kết quả mong đợi: ~936 MB, 8,100,498 rows
```

### Bước 6: Convert kg.csv → UMLS CUI triples

```bash
cd ..  # Quay về data_conversion/

# Chạy pipeline với mapping
python primekg_pipeline.py --skip-download --strategy map
```

**Output:**
- File: `primekg_output/primekg_umls_triples.txt`
- Format: `head,relation,tail` (với UMLS CUIs)

**Kết quả mong đợi:**
```
Loading PrimeKG...
  8,100,498 rows

Using MAP strategy (MONDO→UMLS)
  Loaded 14,567 MONDO→UMLS mappings

Converting...
  Mapped: 450,000 triples
  Unmapped: 7,650,498 rows

Writing to: primekg_output/primekg_umls_triples.txt
✅ Wrote 450,000 triples
```

### Bước 7: Verify output

```bash
# Kiểm tra file output
head -20 primekg_output/primekg_umls_triples.txt

# Đếm số triples
wc -l primekg_output/primekg_umls_triples.txt
```

## 📊 Kết quả mong đợi

**Files tạo ra:**

1. `primekg_data/umls_mondo.csv` (~1-2 MB)
   - 14,000-15,000 mappings
   - Format: `MONDO:0000001,C0012634`

2. `primekg_output/primekg_umls_triples.txt` (~20-30 MB)
   - 400,000-500,000 triples
   - Format: `C0011849,treats,C0004096`

## ⚠️ Troubleshooting

### Error: mondo.obo not found

```bash
# Verify file exists:
ls -lh primekg_data/mondo.obo

# If 0 bytes, re-upload:
scp mondo.obo <user>@<server>:/home/user/GFM/data_conversion/primekg_data/mondo.obo
```

### Error: No UMLS references found

Nếu parse mondo.obo trả về 0 mappings:
1. Kiểm tra file mondo.obo có đúng format không
2. Chạy test: `python test_mondo_obo.py primekg_data/mondo.obo`
3. Verify file có chứa `xref: UMLS:` hoặc `property_value: skos:exactMatch UMLS:`

### Error: kg.csv not found

```bash
cd primekg_data
wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620
```

## 📝 Next Steps

Sau khi có `primekg_umls_triples.txt`:

```bash
# Copy to GFM data directory
cp primekg_output/primekg_umls_triples.txt /home/user/GFM/data/kg.txt

# Run GFM pipeline
cd /home/user/GFM
python -m gfmrag.workflow.stage1_index_dataset
python -m gfmrag.workflow.stage2_entity_resolution
python -m gfmrag.workflow.stage3_umls_mapping
```

## 🔍 Debug Mode

Để debug chi tiết:

```bash
# Test chỉ 1000 rows đầu tiên
python -c "
import pandas as pd
from primekg_to_umls_triples import PrimeKGToUMLSConverter

df = pd.read_csv('primekg_data/kg.csv', nrows=1000)
print(f'Sources: {df.x_source.value_counts().to_dict()}')
print(f'Types: {df.x_type.value_counts().to_dict()}')
"
```

## 💡 Tips

1. **Tốc độ parse mondo.obo:** ~3-5 phút cho 150MB file
2. **Tốc độ convert kg.csv:** ~5-10 phút cho 8M rows
3. **Memory:** Cần ít nhất 4GB RAM
4. **Disk space:** Cần ít nhất 2GB free space

## 📞 Support

Nếu gặp vấn đề, check log output và báo cáo error message đầy đủ.
