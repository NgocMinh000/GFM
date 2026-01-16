# Stage 3 UMLS Mapping - Quick Start Guide

## 🚀 Chạy ngay (TL;DR)

```bash
# Kiểm tra setup
python test_stage3_setup.py

# Chạy pipeline
python -m gfmrag.workflow.stage3_umls_mapping
```

## 📋 Yêu cầu (Requirements)

### 1. UMLS Data
```bash
# Cần có files sau tại data/umls/META/:
data/umls/META/
├── MRCONSO.RRF  (~5GB)  - REQUIRED
├── MRSTY.RRF    (~100MB) - REQUIRED
└── MRDEF.RRF    (~500MB) - OPTIONAL

# Download từ: https://www.nlm.nih.gov/research/umls/
```

### 2. Input từ Stage 2
```bash
# Cần có file:
tmp/entity_resolution/kg_clean.txt

# Format: comma-separated (dấu phẩy)
copper,is a,transition metal
copper,is a,trace element
copper,essential for,cytochrome c oxidase
```

### 3. Dependencies
```bash
pip install sentence-transformers scikit-learn faiss-gpu tqdm hydra-core

# Lưu ý: Dùng faiss-gpu (không phải faiss-cpu) để tối ưu tốc độ!
```

### 4. GPU (QUAN TRỌNG!)
```bash
# Pipeline ưu tiên sử dụng GPU để tăng tốc 5-10x:
# - SapBERT embeddings: GPU
# - FAISS index: GPU
# - Cross-encoder: GPU

# Kiểm tra GPU:
nvidia-smi

# Nếu không có GPU, pipeline sẽ tự động fallback về CPU (chậm hơn)
```

## ⚙️ Config mặc định (đã tối ưu cho GPU)

```yaml
# File: gfmrag/workflow/config/stage3_umls_mapping.yaml

# Input
input:
  kg_clean_path: tmp/entity_resolution/kg_clean.txt  # ✓ Đúng path
  umls_data_dir: data/umls/META                       # ✓ Đúng path

# GPU Optimization (đã set sẵn)
candidate_generation:
  sapbert:
    device: cuda        # ✓ GPU
    batch_size: 256     # ✓ Lớn để tận dụng GPU

cross_encoder:
  device: cuda          # ✓ GPU
  inference:
    batch_size: 32      # ✓ Tối ưu cho GPU

general:
  device: cuda          # ✓ GPU priority
```

## 🏃 Chạy Pipeline

### Cách 1: Script tự động (khuyến nghị)
```bash
bash run_stage3_umls_mapping.sh
```

### Cách 2: Chạy trực tiếp
```bash
python -m gfmrag.workflow.stage3_umls_mapping
```

### Cách 3: Override parameters
```bash
# Nếu không có GPU, dùng CPU:
python -m gfmrag.workflow.stage3_umls_mapping general.device=cpu

# Custom input path:
python -m gfmrag.workflow.stage3_umls_mapping \
  input.kg_clean_path=path/to/your/kg_clean.txt

# Giảm batch size nếu GPU memory không đủ:
python -m gfmrag.workflow.stage3_umls_mapping \
  candidate_generation.sapbert.batch_size=64 \
  cross_encoder.inference.batch_size=16
```

## 📊 Thời gian chạy

### Lần đầu (với GPU):
```
Stage 3.0: Parse UMLS          ~30-60 min   (one-time)
         + Precompute SapBERT   ~1-2 hours  (one-time, GPU)
         + Build indices        ~30 min     (one-time)
Total first run:                ~2-3 hours
```

### Các lần sau (với cache + GPU):
```
Stage 3.0: Load UMLS            ~1 min
Stage 3.1: Preprocessing        ~0.5 min
Stage 3.2: Candidate Gen (GPU)  ~2 min      (500 entities)
Stage 3.3: Cluster Agg          ~0.5 min
Stage 3.4: Hard Neg Filter      ~1 min
Stage 3.5: Cross-Encoder (GPU)  ~2 min      (500 entities)
Stage 3.6: Confidence           ~0.5 min
Total:                          ~7-8 min    (với GPU)
                                ~30-40 min  (với CPU)
```

## 📤 Output Files

```
tmp/umls_mapping/
├── final_umls_mappings.json      # Mappings đầy đủ với confidence
├── umls_mapping_triples.txt       # Thêm vào KG
├── mapping_statistics.json        # Thống kê
├── manual_review_queue.json       # Cases cần review
└── pipeline_metrics.json          # Metrics chi tiết
```

### Sử dụng output:
```bash
# 1. Xem statistics
cat tmp/umls_mapping/mapping_statistics.json

# 2. Thêm vào KG
cat tmp/umls_mapping/umls_mapping_triples.txt >> tmp/entity_resolution/kg_with_umls.txt

# 3. Review uncertain cases
cat tmp/umls_mapping/manual_review_queue.json | jq .
```

## 🐛 Troubleshooting

### Error: "kg_clean.txt not found"
```bash
# Kiểm tra path:
ls tmp/entity_resolution/kg_clean.txt

# Nếu file ở chỗ khác:
python -m gfmrag.workflow.stage3_umls_mapping \
  input.kg_clean_path=path/to/your/kg_clean.txt
```

### Error: "CUDA out of memory"
```bash
# Giảm batch size:
python -m gfmrag.workflow.stage3_umls_mapping \
  candidate_generation.sapbert.batch_size=64 \
  cross_encoder.inference.batch_size=16

# Hoặc dùng CPU:
python -m gfmrag.workflow.stage3_umls_mapping general.device=cpu
```

### Error: "MRCONSO.RRF not found"
```bash
# Kiểm tra UMLS files:
ls -lh data/umls/META/*.RRF

# Nếu chưa có, download từ:
# https://www.nlm.nih.gov/research/umls/
```

### Warning: "Skipping malformed line"
```bash
# Kiểm tra format của kg_clean.txt:
head -5 tmp/entity_resolution/kg_clean.txt

# Phải là comma-separated:
# ✓ ĐÚNG:  copper,is a,transition metal
# ✗ SAI:   copper | is a | transition metal
# ✗ SAI:   copper\tis a\ttransition metal
```

## 💡 Tối ưu hiệu suất

### 1. Sử dụng GPU (quan trọng nhất!)
```yaml
# Config đã set sẵn:
general.device: cuda
candidate_generation.sapbert.device: cuda
cross_encoder.device: cuda
```

### 2. Tăng batch size (nếu GPU memory đủ)
```bash
python -m gfmrag.workflow.stage3_umls_mapping \
  candidate_generation.sapbert.batch_size=512 \
  cross_encoder.inference.batch_size=64
```

### 3. Giảm số candidates (trade-off: tốc độ vs accuracy)
```bash
python -m gfmrag.workflow.stage3_umls_mapping \
  candidate_generation.ensemble.final_k=64 \
  cluster_aggregation.output_k=32 \
  hard_negative_filtering.output_k=16
```

### 4. Sử dụng cache (tự động)
```bash
# Lần đầu: Pipeline tạo cache tại data/umls/processed/
# Các lần sau: Load từ cache (~1 min thay vì 2-3 giờ)

# Xóa cache nếu cần rebuild:
rm -rf data/umls/processed/
```

## 📝 Format Input

### kg_clean.txt phải có format:
```
entity1,relation,entity2

Ví dụ:
copper,is a,transition metal
copper,is a,trace element
copper,essential for,cytochrome c oxidase
diabetes,is a,disease
metformin,treats,diabetes
```

### Lưu ý:
- ✓ Dùng dấu phẩy `,` (comma)
- ✗ KHÔNG dùng `|` (pipe)
- ✗ KHÔNG dùng `\t` (tab)
- Entity names có thể chứa spaces: `diabetes mellitus,is a,disease`

## 🎯 Expected Results

### Confidence Distribution (mong đợi):
```
High confidence (≥0.75):     60-80% entities  → Auto-accept
Medium confidence (0.5-0.75): 15-30% entities  → Review recommended
Low confidence (<0.5):        5-10% entities   → Manual review
```

### Accuracy (mong đợi):
```
Overall Top-1 Accuracy:  85-92%
Recall@5:                93-97%
Recall@10:               95-98%
```

## 📚 Documentation đầy đủ

- `STAGE3_UMLS_MAPPING_README.md` - Complete user guide
- `STAGE3_IMPLEMENTATION_SUMMARY.md` - Technical details
- `STAGE3_ARCHITECTURE.txt` - Architecture diagram

---

**Ready to run?**
```bash
python -m gfmrag.workflow.stage3_umls_mapping
```
