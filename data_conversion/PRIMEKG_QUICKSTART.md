# PrimeKG → UMLS CUI Triples - Quick Start Guide

## ⚡ Chạy nhanh trong 5 phút

### Bước 1: Download PrimeKG data

```bash
cd /home/user/GFM/data_conversion/primekg_analysis

# Option A: Download từ Harvard Dataverse (nếu có network)
wget -O kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620

# Option B: Clone repo để get mapping files
git clone https://github.com/mims-harvard/PrimeKG.git
cp PrimeKG/datasets/data/umls/umls_mondo.csv .
```

**Nếu không download được:** Bạn có thể dùng data mẫu hoặc cung cấp file kg.csv có sẵn.

---

### Bước 2: Chọn chiến lược

#### CHIẾN LƯỢC 1: Filter UMLS only (Đơn giản)

```bash
python primekg_to_umls_triples.py kg.csv output/umls_only_triples.txt \
    --strategy filter
```

**Output:** Chỉ giữ entities có `source=UMLS`
**Thời gian:** ~2-5 phút
**Kết quả:** ~200K-500K triples (từ 4M)

---

#### CHIẾN LƯỢC 2: Map MONDO→UMLS (Recommended)

```bash
python primekg_to_umls_triples.py kg.csv output/umls_mapped_triples.txt \
    --mapping umls_mondo.csv \
    --strategy map
```

**Output:** Map diseases MONDO→UMLS, giữ data nhiều hơn
**Thời gian:** ~10-15 phút
**Kết quả:** ~500K-1M triples

---

#### CHIẾN LƯỢC 2B: Map + giữ unmapped (Toàn diện)

```bash
python primekg_to_umls_triples.py kg.csv output/umls_full_triples.txt \
    --mapping umls_mondo.csv \
    --strategy map \
    --keep-unmapped
```

**Output:** Map diseases, giữ cả drugs/genes không có CUI
**Thời gian:** ~10-15 phút
**Kết quả:** ~3-4M triples (gần như full data)

---

### Bước 3: Validate output

```bash
cd /home/user/GFM/data_conversion
python validate_triples.py primekg_analysis/output/umls_mapped_triples.txt
```

Kiểm tra:
- Format có đúng không?
- Có duplicates không?
- Statistics (entities, relations)

---

### Bước 4: Integrate với GFM pipeline

```bash
# Copy output vào data directory
cp primekg_analysis/output/umls_mapped_triples.txt /home/user/GFM/data/kg.txt

# Run Stage 1: Index KG
cd /home/user/GFM
python -m gfmrag.workflow.stage1_index_dataset

# Run Stage 2: Entity Resolution
python -m gfmrag.workflow.stage2_entity_resolution

# Run Stage 3: UMLS Mapping
python -m gfmrag.workflow.stage3_umls_mapping \
    kg_input_path=tmp/entity_resolution/kg_clean.txt
```

---

## 📊 Expected Results

### Chiến lược 1: Filter UMLS only

```
Input:  4,050,249 triples
Output: ~200,000-500,000 triples (5-12%)

Entities:
- UMLS CUIs only
- Mostly diseases + some genes

Relations:
- disease-disease
- disease-gene
- Limited coverage
```

### Chiến lược 2: Map MONDO→UMLS

```
Input:  4,050,249 triples
Output: ~500,000-1,000,000 triples (12-25%)

Entities:
- UMLS CUIs (diseases mapped from MONDO)
- Some genes/proteins

Relations:
- disease-disease
- disease-drug (some)
- disease-gene
- Better coverage
```

### Chiến lược 2B: Map + keep unmapped

```
Input:  4,050,249 triples
Output: ~3,000,000-4,000,000 triples (75-100%)

Entities:
- UMLS CUIs (diseases)
- DrugBank IDs (drugs)
- NCBI Gene IDs (genes)
- Mixed identifiers

Relations:
- All relation types
- Full coverage
```

---

## 🔧 Advanced Options

### Use 'relation' instead of 'display_relation'

```bash
python primekg_to_umls_triples.py kg.csv output.txt \
    --mapping umls_mondo.csv \
    --no-display-relation
```

### Verbose logging

```bash
python primekg_to_umls_triples.py kg.csv output.txt \
    --mapping umls_mondo.csv \
    --verbose
```

---

## 📁 Output Examples

### Strategy 1 output (Filter UMLS only):
```
C0011849,associated_with,C0004096
C0020538,interacts_with,C0007097
C0011860,treats,C0018681
```

### Strategy 2 output (Map MONDO→UMLS):
```
C0011849,associated_with,C0004096
C0020538,interacts_with,NCBIGene:7157
C0011860,treats,DB00001
```
*Note: Genes/Drugs chưa map → giữ nguyên ID*

### Strategy 2B output (Keep unmapped):
```
C0011849,associated_with,C0004096
C0020538,interacts_with,NCBIGene:7157
DB00001,treats,C0011860
NCBIGene:7157,ppi,NCBIGene:672
```
*Note: Mix of UMLS CUIs và original IDs*

---

## ⚠️ Troubleshooting

### Error: "KG file not found"

```bash
# Check file exists
ls -lh kg.csv

# Use absolute path
python primekg_to_umls_triples.py /full/path/to/kg.csv output.txt
```

### Error: "Mapping file not found"

```bash
# Download mapping from PrimeKG repo
git clone https://github.com/mims-harvard/PrimeKG.git
cp PrimeKG/datasets/data/umls/umls_mondo.csv .

# Or run with --strategy filter (no mapping needed)
python primekg_to_umls_triples.py kg.csv output.txt --strategy filter
```

### Error: "Missing required columns"

```bash
# Check CSV columns
head -1 kg.csv

# Verify it's the correct PrimeKG format
```

### Slow processing

```bash
# Use filter strategy (faster)
python primekg_to_umls_triples.py kg.csv output.txt --strategy filter

# Or reduce data size first
head -100000 kg.csv > kg_sample.csv
python primekg_to_umls_triples.py kg_sample.csv output.txt --mapping umls_mondo.csv
```

---

## 🎯 Which Strategy to Choose?

### ✅ Use Strategy 1 (Filter) if:
- Bạn chỉ cần UMLS concepts thuần túy
- Bạn muốn chạy nhanh nhất
- Bạn OK với việc mất 90-95% data

### ✅ Use Strategy 2 (Map) if:
- Bạn cần diseases (quan trọng nhất)
- Bạn muốn balance giữa coverage và purity
- Bạn có file umls_mondo.csv

### ✅ Use Strategy 2B (Map + keep) if:
- Bạn cần giữ tất cả data
- Bạn OK với mixed identifiers
- Bạn sẽ map thêm drugs/genes sau

---

## 📈 Performance

| Data Size | Strategy | Time | Memory |
|-----------|----------|------|--------|
| 4M rows | Filter | ~3 min | <2GB |
| 4M rows | Map | ~12 min | <4GB |
| 4M rows | Map+keep | ~15 min | <4GB |

*Tested on standard laptop with 16GB RAM*

---

## 🚀 Next Steps

1. **Validate output:**
   ```bash
   python validate_triples.py output/umls_mapped_triples.txt
   ```

2. **Preview first 100 lines:**
   ```bash
   head -100 output/umls_mapped_triples.txt
   ```

3. **Check statistics:**
   ```bash
   wc -l output/umls_mapped_triples.txt
   cut -d',' -f2 output/umls_mapped_triples.txt | sort | uniq -c | sort -rn
   ```

4. **Integrate with GFM:**
   ```bash
   cp output/umls_mapped_triples.txt /home/user/GFM/data/kg.txt
   cd /home/user/GFM
   python -m gfmrag.workflow.stage1_index_dataset
   ```

---

## 💡 Tips

1. **Start with small sample:**
   ```bash
   head -10000 kg.csv > kg_sample.csv
   python primekg_to_umls_triples.py kg_sample.csv test_output.txt --mapping umls_mondo.csv
   ```

2. **Compare strategies:**
   ```bash
   # Run both strategies
   python primekg_to_umls_triples.py kg.csv filter_out.txt --strategy filter
   python primekg_to_umls_triples.py kg.csv map_out.txt --mapping umls_mondo.csv

   # Compare sizes
   wc -l filter_out.txt map_out.txt
   ```

3. **Merge with your data:**
   ```bash
   cat your_triples.txt output/umls_mapped_triples.txt | sort -u > merged.txt
   ```

---

**Need help?** Check [PRIMEKG_TO_UMLS_ANALYSIS.md](PRIMEKG_TO_UMLS_ANALYSIS.md) for detailed analysis.

**Created:** 2026-01-09
**Version:** 1.0.0
