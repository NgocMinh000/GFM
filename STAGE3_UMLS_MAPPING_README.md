# Stage 3: UMLS Mapping Pipeline

## Tổng quan (Overview)

Pipeline 6 giai đoạn để ánh xạ các thực thể y sinh học sang UMLS CUIs (Concept Unique Identifiers).

## Kiến trúc (Architecture)

```
Stage 3.0: UMLS Data Loading & Indexing
   └─ Đọc và xử lý MRCONSO.RRF, MRSTY.RRF, MRDEF.RRF
   └─ Tạo cache cho các lần chạy tiếp theo

Stage 3.1: Preprocessing & Entity Extraction
   └─ Chuẩn hóa entities từ Stage 2 output
   └─ Phân cụm synonyms

Stage 3.2: Candidate Generation (SapBERT + TF-IDF)
   ├─ SapBERT: Semantic similarity search (medical BERT)
   ├─ TF-IDF: Character n-gram matching
   └─ Reciprocal Rank Fusion ensemble

Stage 3.3: Synonym Cluster Aggregation
   └─ Gộp kết quả từ các entities trong cùng 1 cluster
   └─ Phát hiện outliers

Stage 3.4: Hard Negative Filtering
   └─ Lọc các candidates có semantic type không khớp
   └─ Phát hiện hard negatives (similar string, different meaning)

Stage 3.5: Cross-Encoder Reranking
   └─ Rerank bằng PubMedBERT cross-encoder
   └─ Kết hợp với scores từ các stages trước

Stage 3.6: Confidence Scoring & Propagation
   └─ Tính confidence dựa trên nhiều factors
   └─ Lan truyền mappings trong synonym clusters
   └─ Phân loại: High (≥0.75), Medium (0.5-0.75), Low (<0.5)
```

## Cài đặt (Setup)

### 1. Chuẩn bị UMLS Data

**Tải UMLS từ UMLS Metathesaurus:**
```bash
# Đăng ký tài khoản miễn phí tại https://www.nlm.nih.gov/research/umls/
# Download "UMLS Metathesaurus Full Release" (file zip ~6GB)
# Giải nén và copy các file RRF vào:
mkdir -p /home/user/GFM/data/umls/META
```

**Cấu trúc thư mục cần thiết:**
```
data/umls/META/
├── MRCONSO.RRF    # Concepts and names (~4.5M concepts, ~15M names)
├── MRSTY.RRF      # Semantic types
└── MRDEF.RRF      # Definitions (optional, recommended)
```

**Kiểm tra files:**
```bash
ls -lh data/umls/META/*.RRF
# MRCONSO.RRF: ~5GB
# MRSTY.RRF:   ~100MB
# MRDEF.RRF:   ~500MB
```

### 2. Cài đặt dependencies

```bash
# Required packages
pip install sentence-transformers scikit-learn faiss-cpu tqdm hydra-core

# For GPU acceleration (recommended):
pip install faiss-gpu

# For visualization (optional):
pip install matplotlib seaborn
```

### 3. Kiểm tra input từ Stage 2

Pipeline cần output từ Stage 2 Entity Resolution:
```bash
# Kiểm tra file kg_clean.txt tồn tại:
ls tmp/kg_construction/*/hotpotqa/kg_clean.txt

# Format: entity1 | relation | entity2
# Ví dụ:
# diabetes mellitus | is_a | disease
# metformin | treats | diabetes
```

## Sử dụng (Usage)

### Chạy Pipeline Hoàn Chỉnh

**Lệnh cơ bản:**
```bash
cd /home/user/GFM
python -m gfmrag.workflow.stage3_umls_mapping
```

**Với custom config:**
```bash
python -m gfmrag.workflow.stage3_umls_mapping \
  input.kg_clean_path=tmp/kg_construction/run1/hotpotqa/kg_clean.txt \
  output.root_dir=tmp/umls_mapping_run1
```

**Override các tham số:**
```bash
# Sử dụng CPU thay vì GPU
python -m gfmrag.workflow.stage3_umls_mapping general.device=cpu

# Tăng số candidates
python -m gfmrag.workflow.stage3_umls_mapping \
  candidate_generation.sapbert.top_k=128 \
  candidate_generation.ensemble.final_k=256

# Force recompute (không dùng cache)
python -m gfmrag.workflow.stage3_umls_mapping general.force_recompute=true
```

### Lần Chạy Đầu Tiên (First-Time Setup)

**Lưu ý:** Lần chạy đầu tiên sẽ mất **2-4 giờ** để:
- Parse UMLS RRF files (~30-60 phút)
- Precompute SapBERT embeddings cho ~4.5M concepts (~2-3 giờ)
- Build TF-IDF index (~30 phút)
- Build FAISS index (~30 phút)

**Các lần chạy sau sẽ nhanh hơn** vì sử dụng cache:
- Load UMLS từ cache: ~1 phút
- Chỉ encode input entities mới: ~1-5 phút cho 500-1000 entities

**Kiểm tra cache:**
```bash
# Cache sẽ được tạo tại:
ls -lh data/umls/processed/
# umls_concepts.pkl        (~500MB)
# umls_aliases.pkl         (~300MB)
# umls_sapbert_embeddings/ (~12GB)
# umls_tfidf_index/        (~2GB)
# umls_faiss_index/        (~8GB)
```

## Output Files

Pipeline tạo các files sau:

### 1. Final Mappings (JSON)
```json
// outputs/final_umls_mappings.json
{
  "diabetes mellitus": {
    "cui": "C0diabetes",
    "name": "Diabetes Mellitus",
    "confidence": 0.92,
    "tier": "high",
    "alternatives": [
      {"cui": "C1234567", "name": "Type 2 Diabetes", "score": 0.85}
    ],
    "cluster_size": 3,
    "is_propagated": false,
    "confidence_factors": {
      "score_margin": 0.15,
      "absolute_score": 0.92,
      "cluster_consensus": 0.95,
      "method_agreement": 1.0
    }
  }
}
```

### 2. KG Triples
```
// outputs/umls_mapping_triples.txt
diabetes mellitus|mapped_to_cui|C0011849
metformin|mapped_to_cui|C0025598
hypertension|mapped_to_cui|C0020538
```

**Sử dụng:** Thêm triples này vào Knowledge Graph:
```bash
cat outputs/umls_mapping_triples.txt >> tmp/kg_construction/*/hotpotqa/kg_final.txt
```

### 3. Statistics
```json
// outputs/mapping_statistics.json
{
  "total_entities": 1250,
  "high_confidence": 875,
  "medium_confidence": 280,
  "low_confidence": 95,
  "high_confidence_pct": "70.00%",
  "medium_confidence_pct": "22.40%",
  "low_confidence_pct": "7.60%"
}
```

### 4. Manual Review Queue
```json
// outputs/manual_review_queue.json
{
  "MI": {
    "cui": "C0027051",
    "name": "Myocardial Infarction",
    "confidence": 0.42,
    "tier": "low",
    "alternatives": [...]
  }
}
```

### 5. Pipeline Metrics
```json
// outputs/pipeline_metrics.json
{
  "stages": [
    {
      "name": "Stage 3.0: UMLS Data Loading",
      "duration_seconds": 62.5,
      "input_count": 0,
      "output_count": 4567234,
      "metrics": {
        "total_concepts": 4567234,
        "total_aliases": 15234567,
        "avg_aliases_per_concept": 3.34
      }
    },
    ...
  ],
  "total_duration_seconds": 324.7,
  "warnings_count": 12
}
```

### 6. Visualizations (nếu có matplotlib/seaborn)
```
outputs/visualizations/
├── stage_durations.png          # Bar chart: thời gian mỗi stage
├── confidence_distribution.png   # Histogram: phân phối confidence
├── candidate_funnel.png          # Funnel: số candidates qua các stages
└── semantic_type_breakdown.png   # Pie chart: phân phối semantic types
```

## Tối ưu hiệu suất (Performance Tuning)

### GPU Memory Issues
```yaml
# Giảm batch size nếu thiếu GPU memory:
candidate_generation.sapbert.batch_size: 64  # default: 256
cross_encoder.inference.batch_size: 16       # default: 32
```

### Tăng tốc độ
```yaml
# Giảm số candidates nếu muốn chạy nhanh hơn:
candidate_generation.sapbert.top_k: 32       # default: 64
candidate_generation.ensemble.final_k: 64    # default: 128
cluster_aggregation.output_k: 32             # default: 64
hard_negative_filtering.output_k: 16         # default: 32
```

### Tăng độ chính xác
```yaml
# Tăng số candidates và threshold:
candidate_generation.ensemble.final_k: 256
cluster_aggregation.output_k: 128
confidence.tiers.high: 0.85  # default: 0.75
```

## Troubleshooting

### Lỗi: "MRCONSO.RRF not found"
```bash
# Kiểm tra đường dẫn:
ls data/umls/META/MRCONSO.RRF

# Nếu file ở nơi khác, cập nhật config:
python -m gfmrag.workflow.stage3_umls_mapping \
  umls.files.mrconso=/path/to/MRCONSO.RRF \
  umls.files.mrsty=/path/to/MRSTY.RRF
```

### Lỗi: "CUDA out of memory"
```bash
# Chuyển sang CPU:
python -m gfmrag.workflow.stage3_umls_mapping general.device=cpu

# Hoặc giảm batch size:
python -m gfmrag.workflow.stage3_umls_mapping \
  candidate_generation.sapbert.batch_size=64 \
  cross_encoder.inference.batch_size=16
```

### Lỗi: "No input entities found"
```bash
# Kiểm tra path đến kg_clean.txt:
ls tmp/kg_construction/*/hotpotqa/kg_clean.txt

# Cập nhật path trong config:
python -m gfmrag.workflow.stage3_umls_mapping \
  input.kg_clean_path=tmp/kg_construction/run123/hotpotqa/kg_clean.txt
```

### Cache bị hỏng
```bash
# Xóa cache và rebuild:
rm -rf data/umls/processed/
python -m gfmrag.workflow.stage3_umls_mapping general.force_recompute=true
```

## Đánh giá chất lượng (Quality Metrics)

### Confidence Distribution (mong đợi)
- **High confidence (≥0.75):** 60-80% entities
- **Medium confidence (0.5-0.75):** 15-30% entities
- **Low confidence (<0.5):** 5-10% entities

### Review các trường hợp Low Confidence
```bash
# Xem danh sách entities cần review:
cat outputs/manual_review_queue.json | jq -r 'keys[]'

# Kiểm tra từng case:
cat outputs/manual_review_queue.json | jq '."entity_name"'
```

### Kiểm tra alternatives
```python
import json

with open('outputs/final_umls_mappings.json') as f:
    mappings = json.load(f)

# Entities có score margin nhỏ (không chắc chắn):
uncertain = {
    entity: data
    for entity, data in mappings.items()
    if data['confidence_factors']['score_margin'] < 0.1
}

print(f"Uncertain mappings: {len(uncertain)}")
```

## Configuration File

Chi tiết cấu hình tại: `gfmrag/workflow/config/stage3_umls_mapping.yaml`

Các thông số quan trọng:
- **SapBERT model:** `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- **Cross-encoder:** `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- **Ensemble method:** Reciprocal Rank Fusion (RRF)
- **Confidence tiers:** High (≥0.75), Medium (0.5-0.75), Low (<0.5)

## Tích hợp vào workflow (Integration)

### Workflow hoàn chỉnh:
```bash
# Stage 0: Type Inference
python -m gfmrag.workflow.stage2_entity_resolution stage=0

# Stage 1: Synonym Resolution
python -m gfmrag.workflow.stage2_entity_resolution stage=1

# Stage 2: Entity Resolution
python -m gfmrag.workflow.stage2_entity_resolution stage=2

# Stage 3: UMLS Mapping (mới!)
python -m gfmrag.workflow.stage3_umls_mapping

# Kết quả: kg_clean.txt + umls_mapping_triples.txt
```

## Thông tin thêm

- **Thời gian chạy (sau first-time setup):** 5-15 phút cho 500-1000 entities
- **GPU memory:** ~8-12GB (SapBERT + Cross-encoder)
- **Disk space:** ~25GB cho cache (UMLS + embeddings)
- **Accuracy (dự kiến):** 85-95% cho high-confidence mappings

---

**Developed for GFM-RAG Project**
**Version:** 1.0.0
**Last Updated:** 2026-01-04
