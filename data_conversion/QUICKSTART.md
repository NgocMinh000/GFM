# Quick Start Guide - CSV to Triples Conversion

Hướng dẫn nhanh để chuyển đổi dữ liệu CSV của bạn sang format triples cho GFM-RAG pipeline.

## ⚡ Chạy nhanh trong 30 giây

```bash
cd /home/user/GFM/data_conversion

# Bước 1: Đặt file CSV của bạn vào thư mục input/
cp /path/to/your_data.csv input/

# Bước 2: Chạy conversion (chọn 1 trong 2 cách)

# Cách 1: Dùng quick script (đơn giản nhất)
./quick_convert.sh input/your_data.csv output/triples.txt

# Cách 2: Dùng Python script trực tiếp
python csv_to_triples.py input/your_data.csv output/triples.txt
```

Done! ✅

## 📋 Các ví dụ thực tế

### Example 1: Basic protein-protein interaction data

**File của bạn:** `protein_interactions.csv`
```csv
relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source
protein_protein,interacts_with,0,123,protein,BRCA1,UniProt,1,456,protein,TP53,UniProt
```

**Command:**
```bash
./quick_convert.sh input/protein_interactions.csv output/ppi_triples.txt
```

**Output:** `output/ppi_triples.txt`
```
BRCA1,interacts_with,TP53
```

### Example 2: Drug-disease relationships with metadata

**File của bạn:** `drug_disease.csv`
```csv
relation,display_relation,x_id,x_type,x_name,x_source,y_id,y_type,y_name,y_source
compound_disease,treats,DB00001,drug,Aspirin,DrugBank,C0018681,disease,Headache,MESH
```

**Command:**
```bash
python csv_to_triples.py input/drug_disease.csv output/drug_triples.txt --add-metadata
```

**Output includes metadata:**
```
Aspirin,treats,Headache
Aspirin,has_type,drug
Aspirin,has_id,DB00001
Aspirin,has_source,DrugBank
Headache,has_type,disease
Headache,has_id,C0018681
Headache,has_source,MESH
```

### Example 3: Custom column names

**File của bạn có format khác:**
```csv
source,edge_type,target
GeneA,regulates,GeneB
```

**Command:**
```bash
python csv_to_triples.py input/custom.csv output/triples.txt \
    --head-column source \
    --relation-column edge_type \
    --tail-column target
```

## 🔍 Validate output

Luôn validate output trước khi dùng:

```bash
# Check if file is valid
python validate_triples.py output/triples.txt

# Quick check with head
head -20 output/triples.txt

# Count triples
wc -l output/triples.txt
```

## 🚀 Tích hợp với GFM-RAG

Sau khi convert xong, chạy full pipeline:

```bash
# Copy output vào data directory
cp output/triples.txt /home/user/GFM/data/kg.txt

# Run Stage 1: Index KG
cd /home/user/GFM
python -m gfmrag.workflow.stage1_index_dataset

# Run Stage 2: Entity Resolution
python -m gfmrag.workflow.stage2_entity_resolution

# Run Stage 3: UMLS Mapping
python -m gfmrag.workflow.stage3_umls_mapping \
    kg_input_path=tmp/entity_resolution/kg_clean.txt
```

## 📊 Expected output sizes

| Input CSV Rows | Output Triples | Time |
|----------------|----------------|------|
| 100 | 100 | <1s |
| 1,000 | 1,000 | ~1s |
| 10,000 | 10,000 | ~5s |
| 100,000 | 100,000 | ~30s |
| 1,000,000 | 1,000,000 | ~5min |

*Note: With `--add-metadata`, output triples = input rows × 7 (main triple + 6 metadata triples)*

## ❓ Common Questions

### Q: Tôi có nhiều file CSV, làm sao merge?

**A:** Chạy conversion từng file, sau đó merge:

```bash
# Convert each file
python csv_to_triples.py input/file1.csv output/triples1.txt
python csv_to_triples.py input/file2.csv output/triples2.txt

# Merge and deduplicate
cat output/triples1.txt output/triples2.txt | sort -u > output/all_triples.txt
```

### Q: File CSV có quá nhiều cột, cái nào quan trọng?

**A:** Chỉ cần 3 cột:
- Head entity (default: `x_name`)
- Relation (default: `display_relation`)
- Tail entity (default: `y_name`)

Các cột khác sẽ được ignore (trừ khi dùng `--add-metadata`)

### Q: Làm sao biết conversion thành công?

**A:** Check 3 điều:
1. Script báo "✅ Conversion completed successfully!"
2. Output file tồn tại: `ls -lh output/triples.txt`
3. Validate pass: `python validate_triples.py output/triples.txt`

### Q: Output có duplicates, làm sao xóa?

**A:** Script tự động xóa duplicates (default). Nếu muốn giữ:

```bash
python csv_to_triples.py input/data.csv output/triples.txt --no-deduplicate
```

### Q: Entity names có uppercase/lowercase khác nhau?

**A:** Script normalize text (strip whitespace). Nếu muốn tắt:

```bash
python csv_to_triples.py input/data.csv output/triples.txt --no-normalize
```

## 🎯 Best Practices

1. **Luôn test với sample data trước:**
   ```bash
   head -100 input/big_file.csv > input/sample.csv
   ./quick_convert.sh input/sample.csv output/test.txt
   ```

2. **Validate output:**
   ```bash
   python validate_triples.py output/triples.txt
   ```

3. **Backup original data:**
   ```bash
   cp input/your_data.csv input/your_data.csv.backup
   ```

4. **Check statistics:**
   - Số entities có hợp lý không?
   - Số relations có hợp lý không?
   - Có duplicates không?

5. **Version control:**
   ```bash
   git add data_conversion/output/triples.txt
   git commit -m "Add converted triples from protein data"
   ```

## 📁 File Structure

```
data_conversion/
├── README.md                       # Full documentation
├── QUICKSTART.md                   # This file
├── csv_to_triples.py              # Main conversion script
├── validate_triples.py            # Validation script
├── quick_convert.sh               # Quick conversion helper
├── input/                         # Put your CSV files here
│   └── sample_data.csv           # Example data
└── output/                        # Conversion output
    ├── triples.txt               # Basic output
    └── triples_with_metadata.txt # Output with metadata
```

## 🔗 Related Docs

- [README.md](README.md) - Detailed documentation
- [STAGE1_ARCHITECTURE.md](../STAGE1_ARCHITECTURE.md) - KG indexing
- [STAGE2_ARCHITECTURE.md](../STAGE2_ARCHITECTURE.md) - Entity resolution
- [STAGE3_ARCHITECTURE.md](../STAGE3_ARCHITECTURE.md) - UMLS mapping

## 💡 Tips

1. **Large files**: Sử dụng `--verbose` để track progress
2. **Custom format**: Dùng `--head-column`, `--relation-column`, `--tail-column`
3. **Rich metadata**: Dùng `--add-metadata` để giữ types, IDs, sources
4. **Quality check**: Luôn chạy `validate_triples.py` trước khi dùng

---

**Need help?** Check [README.md](README.md) for full documentation.
