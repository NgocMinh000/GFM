# CSV to Triples Converter

Công cụ chuyển đổi dữ liệu CSV phức tạp (với nhiều cột metadata) sang format triples đơn giản (entity-relation-entity) để sử dụng cho GFM-RAG pipeline.

## 📋 Tổng quan

Tool này giúp bạn chuyển đổi từ:

**Input format (CSV phức tạp):**
```csv
relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source
protein_protein,ppi,0,9796,gene/protein,PHYHIP,NCBI,8889,56992,gene/protein,KIF15,NCBI
```

**Output format (triples đơn giản):**
```
PHYHIP,ppi,KIF15
```

## 🚀 Cài đặt

```bash
# Cài đặt dependencies (nếu chưa có)
pip install pandas tqdm
```

## 📖 Cách sử dụng

### 1. Basic Usage - Chuyển đổi cơ bản

```bash
cd /home/user/GFM/data_conversion

# Chuyển đổi CSV thành triples
python csv_to_triples.py input/your_data.csv output/triples.txt
```

**Kết quả:**
- File output chỉ chứa 3 cột: `head,relation,tail`
- Tự động loại bỏ duplicates
- Normalize text (xóa whitespace thừa)

### 2. Advanced Options

#### 2.1. Thêm metadata (types, IDs, sources)

```bash
python csv_to_triples.py input/your_data.csv output/triples_full.txt --add-metadata
```

**Output sẽ bao gồm:**
- Triples chính: `PHYHIP,ppi,KIF15`
- Entity types: `PHYHIP,has_type,gene/protein`
- Entity IDs: `PHYHIP,has_id,9796`
- Entity sources: `PHYHIP,has_source,NCBI`

#### 2.2. Sử dụng cột relation khác

```bash
# Dùng cột 'relation' thay vì 'display_relation'
python csv_to_triples.py input/your_data.csv output/triples.txt \
    --relation-column relation
```

#### 2.3. Custom column names

```bash
# Nếu CSV của bạn có tên cột khác
python csv_to_triples.py input/your_data.csv output/triples.txt \
    --head-column source_entity \
    --relation-column edge_type \
    --tail-column target_entity
```

#### 2.4. Giữ duplicates và không normalize

```bash
python csv_to_triples.py input/your_data.csv output/triples.txt \
    --no-deduplicate \
    --no-normalize
```

#### 2.5. Verbose logging để debug

```bash
python csv_to_triples.py input/your_data.csv output/triples.txt --verbose
```

## 📁 Cấu trúc thư mục

```
data_conversion/
├── README.md                    # Hướng dẫn này
├── csv_to_triples.py           # Script chính
├── input/                       # Thư mục chứa CSV input
│   └── sample_data.csv         # Dữ liệu mẫu
└── output/                      # Thư mục chứa triples output
    ├── triples.txt             # Output cơ bản
    └── triples_with_metadata.txt  # Output với metadata
```

## 🎯 Examples

### Example 1: Basic conversion

**Input:** `input/sample_data.csv`
```csv
relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source
protein_protein,ppi,0,9796,gene/protein,PHYHIP,NCBI,8889,56992,gene/protein,KIF15,NCBI
protein_protein,ppi,1,7918,gene/protein,GPANK1,NCBI,2798,9240,gene/protein,PNMA1,NCBI
```

**Command:**
```bash
python csv_to_triples.py input/sample_data.csv output/triples.txt
```

**Output:** `output/triples.txt`
```
PHYHIP,ppi,KIF15
GPANK1,ppi,PNMA1
```

**Statistics:**
```
Total rows processed:    2
Valid triples:           2
Invalid rows:            0
Duplicate triples:       0
Unique entities:         4
Unique relations:        1
```

### Example 2: With metadata

**Command:**
```bash
python csv_to_triples.py input/sample_data.csv output/triples_full.txt --add-metadata
```

**Output:** `output/triples_full.txt`
```
PHYHIP,ppi,KIF15
PHYHIP,has_type,gene/protein
PHYHIP,has_id,9796
PHYHIP,has_source,NCBI
KIF15,has_type,gene/protein
KIF15,has_id,56992
KIF15,has_source,NCBI
GPANK1,ppi,PNMA1
GPANK1,has_type,gene/protein
GPANK1,has_id,7918
GPANK1,has_source,NCBI
PNMA1,has_type,gene/protein
PNMA1,has_id,9240
PNMA1,has_source,NCBI
```

## 🔧 Parameters

| Parameter | Mặc định | Mô tả |
|-----------|----------|--------|
| `--head-column` | `x_name` | Tên cột cho head entity |
| `--relation-column` | `display_relation` | Tên cột cho relation |
| `--tail-column` | `y_name` | Tên cột cho tail entity |
| `--fallback-relation` | `relation` | Cột relation dự phòng |
| `--add-metadata` | `False` | Thêm metadata triples |
| `--no-deduplicate` | `False` | Giữ duplicate triples |
| `--no-normalize` | `False` | Tắt text normalization |
| `--verbose` | `False` | Hiển thị debug logs |

## 📊 Features

### ✅ Xử lý dữ liệu thông minh

- **Deduplication**: Tự động loại bỏ triples trùng lặp
- **Normalization**: Chuẩn hóa text (strip whitespace, xử lý punctuation)
- **Validation**: Kiểm tra dữ liệu hợp lệ (không thiếu head/relation/tail)
- **Fallback**: Tự động dùng cột dự phòng nếu cột chính trống

### 📈 Statistics chi tiết

Script cung cấp summary đầy đủ:
- Tổng số rows xử lý
- Số triples hợp lệ/không hợp lệ
- Số duplicates
- Số entities/relations duy nhất
- Top relations phổ biến
- Entity types (nếu dùng --add-metadata)

### 🎨 Progress tracking

- Progress bar cho batch processing lớn
- Real-time logging
- Error handling với messages rõ ràng

## 🔄 Tích hợp với GFM-RAG Pipeline

### Bước 1: Chuyển đổi dữ liệu

```bash
# Chuyển CSV của bạn thành triples
python csv_to_triples.py input/your_protein_data.csv output/kg.txt
```

### Bước 2: Chạy Stage 1 (Index KG)

```bash
# Copy output vào đúng thư mục
cp output/kg.txt /home/user/GFM/data/kg.txt

# Chạy Stage 1
cd /home/user/GFM
python -m gfmrag.workflow.stage1_index_dataset
```

### Bước 3: Chạy Stage 2 (Entity Resolution)

```bash
python -m gfmrag.workflow.stage2_entity_resolution
```

Output sẽ tạo `tmp/entity_resolution/kg_clean.txt` với SYNONYM_OF edges.

### Bước 4: Chạy Stage 3 (UMLS Mapping)

```bash
python -m gfmrag.workflow.stage3_umls_mapping \
    kg_input_path=tmp/entity_resolution/kg_clean.txt
```

## 🧪 Testing với dữ liệu mẫu

```bash
cd /home/user/GFM/data_conversion

# Test basic conversion
python csv_to_triples.py input/sample_data.csv output/test_basic.txt

# Test with metadata
python csv_to_triples.py input/sample_data.csv output/test_metadata.txt --add-metadata

# Xem kết quả
head -20 output/test_basic.txt
head -40 output/test_metadata.txt
```

## 📝 Notes

### Supported CSV formats

Script hỗ trợ nhiều format CSV:
- **Standard format**: `relation,display_relation,x_index,x_id,x_type,x_name,x_source,y_index,y_id,y_type,y_name,y_source`
- **Minimal format**: Chỉ cần `x_name`, `relation`, `y_name`
- **Custom format**: Chỉ định tên cột với parameters

### Handling missing data

- Nếu relation column không tồn tại → dùng fallback column
- Nếu head/tail rỗng → skip row đó (log warning)
- Nếu có NaN/None → normalize thành empty string

### Performance

- **Small files** (<10K rows): ~0.5 giây
- **Medium files** (10K-100K rows): ~2-5 giây
- **Large files** (100K-1M rows): ~20-60 giây
- **Very large files** (>1M rows): Có thể cần 5-10 phút

## ❓ Troubleshooting

### Lỗi: "Missing required columns"

```bash
# Kiểm tra tên cột trong CSV
head -1 input/your_data.csv

# Chỉ định đúng tên cột
python csv_to_triples.py input/your_data.csv output/triples.txt \
    --head-column YOUR_HEAD_COL \
    --tail-column YOUR_TAIL_COL
```

### Lỗi: "Module not found: pandas"

```bash
pip install pandas tqdm
```

### Output file bị trống

```bash
# Bật verbose mode để debug
python csv_to_triples.py input/your_data.csv output/triples.txt --verbose
```

## 🔗 Liên quan

- [STAGE1_ARCHITECTURE.md](../STAGE1_ARCHITECTURE.md) - KG indexing pipeline
- [STAGE2_ARCHITECTURE.md](../STAGE2_ARCHITECTURE.md) - Entity resolution pipeline
- [STAGE3_ARCHITECTURE.md](../STAGE3_ARCHITECTURE.md) - UMLS mapping pipeline
- [QUICKSTART.md](../QUICKSTART.md) - Hướng dẫn chạy toàn bộ pipeline

## 📧 Support

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra lại format CSV input
2. Chạy với `--verbose` để xem logs chi tiết
3. Kiểm tra example data trong `input/sample_data.csv`

---

**Created**: 2026-01-09
**Version**: 1.0.0
**Author**: GFM-RAG Team
