# Hướng dẫn sử dụng đơn giản - PrimeKG to UMLS Triples

## 📌 Tình huống của bạn

Bạn có file **kg.csv** từ PrimeKG nhưng **KHÔNG có** file `umls_mondo.csv` (mapping file).

**Giải pháp:** Dùng strategy "filter" - chỉ lọc entities có `source=UMLS`.

## ✅ Cách chạy

### Option 1: Dùng pipeline (Khuyến nghị)

```bash
cd /home/user/GFM/data_conversion

# Đảm bảo kg.csv ở đúng vị trí
mkdir -p primekg_data
cp /path/to/your/kg.csv primekg_data/kg.csv

# Chạy với strategy filter (không cần mapping file)
python primekg_pipeline.py --skip-download --strategy filter
```

### Option 2: Chạy trực tiếp converter

```bash
cd /home/user/GFM/data_conversion

python primekg_to_umls_triples.py \
    /path/to/kg.csv \
    ./output_triples.txt \
    --strategy filter
```

### Option 3: Test script

```bash
cd /home/user/GFM/data_conversion
chmod +x test_filter_strategy.py
python test_filter_strategy.py /path/to/kg.csv
```

## 📊 Kết quả mong đợi

**Strategy "filter":**
- **Input:** 4M triples (toàn bộ PrimeKG)
- **Output:** 200K-500K triples (chỉ UMLS entities)
- **Thời gian:** ~3-5 phút
- **Format:** `head,relation,tail` với UMLS CUI

**Ví dụ output:**
```
C0011849,treats,C0004096
C0020538,associated_with,C0007097
C0028754,overexpressed_in,C0006826
```

## 🔍 File structure yêu cầu

```
data_conversion/
├── primekg_data/
│   └── kg.csv              # File bạn đã tải (REQUIRED)
│   └── umls_mondo.csv      # Không cần thiết cho strategy filter
├── primekg_output/
│   └── primekg_umls_triples.txt  # Output sẽ được tạo ở đây
└── primekg_to_umls_triples.py
```

## ❓ FAQs

### Q: File umls_mondo.csv của tôi có cấu trúc như kg.csv, đúng không?

**A:** Đúng! File đó thực ra là `kg.csv`, không phải `umls_mondo.csv`.

File `umls_mondo.csv` thật sự nên có cấu trúc:
```csv
umls_id,mondo_id
C0011849,MONDO:0005148
C0020538,MONDO:0005015
```

Nhưng file này **không có sẵn** công khai trong PrimeKG.

### Q: Làm sao để có nhiều triples hơn?

**A:** Có 2 cách:

1. **Dùng `--keep-unmapped`**: Giữ lại cả entities không phải UMLS
   ```bash
   python primekg_pipeline.py --skip-download --strategy filter --keep-unmapped
   ```
   Output: ~3-4M triples (gần như toàn bộ)

2. **Tìm file mapping thật**: Nếu bạn có access vào UMLS source data, có thể tạo file `umls_mondo.csv` và dùng strategy "map"

### Q: Output có đúng format cho GFM không?

**A:** Có! Output format là:
```
head,relation,tail
```

Đúng format mà GFM Stage 1 cần.

### Q: Làm sao verify output?

**A:** Chạy validator:
```bash
python validate_triples.py ./primekg_output/primekg_umls_triples.txt
```

## 🚀 Next Steps sau khi có triples

```bash
# 1. Copy to GFM data directory
cp primekg_output/primekg_umls_triples.txt /home/user/GFM/data/kg.txt

# 2. Run GFM pipeline
cd /home/user/GFM
python -m gfmrag.workflow.stage1_index_dataset
python -m gfmrag.workflow.stage2_entity_resolution
python -m gfmrag.workflow.stage3_umls_mapping
```

## 📝 Tóm tắt

**TL;DR:**
```bash
# Tạo thư mục và copy file
mkdir -p primekg_data
cp /path/to/kg.csv primekg_data/kg.csv

# Chạy
python primekg_pipeline.py --skip-download --strategy filter

# Kết quả
ls -lh primekg_output/primekg_umls_triples.txt
```

Done! ✅
