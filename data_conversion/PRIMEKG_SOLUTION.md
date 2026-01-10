# Giải pháp hoàn chỉnh cho PrimeKG → Triples

## ❗ Vấn đề phát hiện

### 1. kg.csv KHÔNG CÓ source=UMLS
Sau khi phân tích kg.csv (8,100,498 rows), phát hiện:

**Sources có sẵn:**
- DrugBank: 5,611,392
- NCBI: 5,262,458
- UBERON: 3,132,308
- GO: 884,054
- **MONDO: 536,698** ✅ (Diseases)
- HPO: 514,192
- MONDO_grouped: 145,790

**KHÔNG CÓ: UMLS** ❌

### 2. Filter strategy trả về 0 triples
```
Filtered: 8100498 → 0 rows (0.0%)
```
Lý do: Không có entity nào có `x_source=UMLS` hoặc `y_source=UMLS`.

### 3. Mapping file không tồn tại công khai
File `umls_mondo.csv` **không có sẵn** trong PrimeKG public release.

File này được tạo từ MONDO ontology (.obo file) nhưng:
- Download MONDO .obo bị chặn (network restrictions)
- File không có trong GitHub repo
- Harvard Dataverse không cung cấp

## ✅ Giải pháp

### Phương án 1: Sử dụng MONDO IDs (Đơn giản nhất)

PrimeKG có 536,698 MONDO disease IDs. Tạo triples với MONDO thay vì UMLS.

**Cách chạy:**
```bash
cd /home/user/GFM/data_conversion

# Đảm bảo kg.csv ở đúng vị trí
mkdir -p primekg_data
# Tải kg.csv từ Dataverse nếu chưa có:
# wget -O primekg_data/kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620

# Chuyển đổi (chỉ MONDO diseases)
python primekg_to_triples_mondo.py \
    primekg_data/kg.csv \
    primekg_output/primekg_mondo_triples.txt \
    --sources MONDO

# Kết quả: ~270K triples với MONDO disease IDs
```

**Output format:**
```
MONDO:0005148,associated_with,9796
MONDO:0008807,linked_to,DB00001
```

**Ưu điểm:**
- ✅ Không cần mapping file
- ✅ Không cần download MONDO .obo
- ✅ Dữ liệu có sẵn trong kg.csv
- ✅ Chạy ngay được

**Nhược điểm:**
- ❌ Dùng MONDO IDs thay vì UMLS CUIs
- ❌ Cần update GFM pipeline để hỗ trợ MONDO

### Phương án 2: MONDO + DrugBank + NCBI (Cân bằng)

Bao gồm cả diseases (MONDO) và proteins/drugs:

```bash
python primekg_to_triples_mondo.py \
    primekg_data/kg.csv \
    primekg_output/primekg_mixed_triples.txt \
    --sources "MONDO,DrugBank,NCBI"

# Kết quả: ~5-6M triples
```

**Loại entity bao gồm:**
- MONDO: Diseases
- DrugBank: Drugs
- NCBI: Genes/Proteins

### Phương án 3: Tất cả sources

Giữ toàn bộ 8M triples:

```bash
python primekg_to_triples_mondo.py \
    primekg_data/kg.csv \
    primekg_output/primekg_all_triples.txt \
    --sources all

# Kết quả: ~8M triples (toàn bộ knowledge graph)
```

### Phương án 4: Tạo UMLS mapping thủ công (Phức tạp)

Nếu bạn cần UMLS CUIs thực sự:

1. **Download MONDO .obo thủ công:**
```bash
# Thử từ máy khác không bị chặn network
wget https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.obo
# Hoặc
curl -o mondo.obo http://purl.obolibrary.org/obo/mondo.obo

# Copy vào server
scp mondo.obo user@server:/home/user/GFM/data_conversion/primekg_data/
```

2. **Parse và extract UMLS mapping:**
```bash
cd /home/user/GFM/data_conversion
python create_umls_mondo_mapping.py

# Output: primekg_data/umls_mondo.csv
```

3. **Chuyển đổi với mapping:**
```bash
python primekg_pipeline.py --skip-download --strategy map

# Kết quả: Triples với UMLS CUIs
```

## 📊 So sánh output

| Phương án | Triples | UMLS? | Complexity | Khuyến nghị |
|-----------|---------|-------|------------|-------------|
| MONDO only | ~270K | ❌ | ⭐ | Đơn giản nhất |
| MONDO+Drug+Gene | ~5-6M | ❌ | ⭐⭐ | **Cân bằng** ✅ |
| All sources | ~8M | ❌ | ⭐⭐ | Đầy đủ nhất |
| With UMLS mapping | ~500K | ✅ | ⭐⭐⭐⭐⭐ | Khó thực hiện |

## 🚀 Khuyến nghị

### Cho GFM-RAG Pipeline:

**Option A: Dùng MONDO (nhanh nhất)**
```bash
# 1. Tải kg.csv
wget -O primekg_data/kg.csv https://dataverse.harvard.edu/api/access/datafile/6180620

# 2. Convert
python primekg_to_triples_mondo.py \
    primekg_data/kg.csv \
    primekg_output/primekg_triples.txt \
    --sources "MONDO,DrugBank,NCBI"

# 3. Copy to GFM
cp primekg_output/primekg_triples.txt /home/user/GFM/data/kg.txt

# 4. Run GFM (cần update để hỗ trợ MONDO IDs)
cd /home/user/GFM
python -m gfmrag.workflow.stage1_index_dataset
```

**Option B: Dùng pipeline cũ nhưng accept 0 triples**
Nếu GFM pipeline PHẢI có UMLS CUIs, thì filter strategy sẽ trả về 0 triples vì kg.csv không có UMLS source.

## 📝 Các file script

### 1. `primekg_to_triples_mondo.py` ✅ (MỚI)
- Convert PrimeKG → triples với MONDO/DrugBank/NCBI IDs
- Không cần mapping file
- Chạy được ngay

**Usage:**
```bash
python primekg_to_triples_mondo.py kg.csv output.txt --sources "MONDO,DrugBank"
```

### 2. `create_umls_mondo_mapping.py` ⚠️ (Network bị chặn)
- Download MONDO .obo và extract UMLS mapping
- Bị lỗi do network restrictions
- Cần download thủ công

### 3. `primekg_to_umls_triples.py` ❌ (Không hoạt động)
- Cần umls_mondo.csv mapping file
- Filter strategy trả về 0 triples vì kg.csv không có UMLS

### 4. `primekg_pipeline.py` ⚠️ (Đã update)
- Đã fix để không yêu cầu umls_mondo.csv khi dùng filter strategy
- Nhưng vẫn trả về 0 triples

## 🎯 Kết luận

**PrimeKG kg.csv KHÔNG CÓ UMLS CUIs.**

Các option:
1. ✅ **Dùng MONDO IDs** (khuyến nghị) - Script `primekg_to_triples_mondo.py`
2. ⚠️ Tạo mapping thủ công từ MONDO .obo (phức tạp)
3. ❌ Tìm source data khác có UMLS

**Next step:** Quyết định xem GFM-RAG có thể dùng MONDO IDs thay vì UMLS CUIs không.

Nếu GFM PHẢI dùng UMLS, cần:
- Download MONDO .obo thủ công từ máy khác
- Hoặc tìm alternative knowledge graph có sẵn UMLS CUIs
