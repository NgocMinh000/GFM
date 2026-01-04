# Hướng Dẫn Chi Tiết: Chạy và Đánh Giá Stage 3 UMLS Mapping

## 📋 Mục Lục
1. [Prerequisites - Chuẩn bị](#1-prerequisites---chuẩn-bị)
2. [Cài Đặt Dependencies](#2-cài-đặt-dependencies)
3. [Download UMLS Data](#3-download-umls-data)
4. [Cấu Hình Pipeline](#4-cấu-hình-pipeline)
5. [Chạy Pipeline](#5-chạy-pipeline)
6. [Xem Kết Quả](#6-xem-kết-quả)
7. [Đánh Giá Chất Lượng](#7-đánh-giá-chất-lượng)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites - Chuẩn Bị

### 1.1 Kiểm Tra Hệ Thống

```bash
# Kiểm tra Python version (cần >= 3.8)
python --version

# Kiểm tra GPU (khuyến nghị có GPU)
nvidia-smi

# Kiểm tra disk space (cần ~50GB trống)
df -h
```

### 1.2 Dữ Liệu Cần Có

✅ **Bắt buộc:**
- `data/kg_clean.txt` - Knowledge Graph từ Stage 2
- UMLS files (MRCONSO.RRF, MRSTY.RRF, MRDEF.RRF)

✅ **Tự động tạo:**
- Pipeline sẽ tự động tạo các file cache và output

---

## 2. Cài Đặt Dependencies

### 2.1 Core Dependencies

```bash
# Cài đặt các package cần thiết
pip install torch transformers faiss-cpu scikit-learn tqdm numpy

# Nếu có GPU, dùng faiss-gpu thay vì faiss-cpu
pip install faiss-gpu
```

### 2.2 Visualization Dependencies (Tùy chọn)

```bash
# Để tạo biểu đồ đánh giá
pip install matplotlib seaborn
```

### 2.3 Verify Installation

```bash
python -c "
import torch
import transformers
import faiss
import sklearn
print('✓ All core dependencies installed')
"

python -c "
import matplotlib
import seaborn
print('✓ Visualization dependencies installed')
" 2>/dev/null || echo "⚠ Visualization libs not installed (optional)"
```

---

## 3. Download UMLS Data

### 3.1 Đăng Ký UMLS License

**QUAN TRỌNG:** UMLS yêu cầu license miễn phí

1. Truy cập: https://www.nlm.nih.gov/research/umls/
2. Tạo tài khoản UTS (UMLS Terminology Services)
3. Chấp nhận License Agreement
4. Download UMLS Metathesaurus

### 3.2 Download Full Release

```bash
# Tạo thư mục UMLS
mkdir -p data/umls

# Download UMLS 2024AB (hoặc version mới nhất)
# Sau khi login vào UTS, download file ZIP:
# - umls-2024AB-full.zip (~10GB)

# Giải nén
cd data/umls
unzip umls-2024AB-full.zip
cd 2024AB/META
```

### 3.3 Verify UMLS Files

```bash
# Kiểm tra các file cần thiết
ls -lh data/umls/2024AB/META/

# Phải có 3 files này:
# MRCONSO.RRF (~15GB)  - Concept names và synonyms
# MRSTY.RRF   (~500MB) - Semantic types
# MRDEF.RRF   (~300MB) - Definitions
```

**Nếu không có UMLS License:**
Có thể dùng UMLS subset nhỏ hơn cho testing (không khuyến nghị cho production):
```bash
# Download UMLS Sample (không cần license, chỉ cho test)
# https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/sample.html
```

---

## 4. Cấu Hình Pipeline

### 4.1 Kiểm Tra File Input

```bash
# Verify KG file tồn tại
ls -lh data/kg_clean.txt

# Xem vài dòng đầu
head -20 data/kg_clean.txt

# Đếm số entities
wc -l data/kg_clean.txt
```

### 4.2 Tạo Config File (Tùy chọn)

Pipeline có config mặc định tốt, nhưng bạn có thể tùy chỉnh:

```bash
# Tạo file config tùy chỉnh
cat > config/my_umls_config.yaml << 'EOF'
# Input paths
kg_clean_path: "./data/kg_clean.txt"
umls_data_dir: "./data/umls/2024AB/META"
output_root: "./tmp/umls_mapping"

# UMLS files
mrconso_path: "./data/umls/2024AB/META/MRCONSO.RRF"
mrsty_path: "./data/umls/2024AB/META/MRSTY.RRF"
mrdef_path: "./data/umls/2024AB/META/MRDEF.RRF"

# Stage 2: Candidate Generation
sapbert_model: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
sapbert_batch_size: 256  # Giảm nếu GPU memory thấp
sapbert_top_k: 64
ensemble_final_k: 128

# Stage 3: Cluster Aggregation
cluster_output_k: 64

# Stage 4: Hard Negative Filtering
hard_neg_similarity_threshold: 0.7
hard_neg_output_k: 32

# Stage 5: Cross-Encoder
cross_encoder_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# Stage 6: Confidence
confidence_high_threshold: 0.75
propagation_min_agreement: 0.8

# Runtime
device: "cuda"  # hoặc "cpu" nếu không có GPU
num_processes: 10
force_recompute: false
save_intermediate: true
EOF
```

### 4.3 Config Parameters Quan Trọng

| Parameter | Mô tả | Giá trị khuyến nghị |
|-----------|-------|---------------------|
| `sapbert_batch_size` | Batch size cho encoding | GPU: 256-512, CPU: 32-64 |
| `ensemble_final_k` | Số candidates sau ensemble | 128 (cân bằng recall/precision) |
| `confidence_high_threshold` | Ngưỡng high confidence | 0.75 (60%+ mappings sẽ high) |
| `device` | GPU hoặc CPU | "cuda" nếu có GPU |
| `save_intermediate` | Lưu kết quả trung gian | true (để debug) |

---

## 5. Chạy Pipeline

### 5.1 Chạy Toàn Bộ Pipeline (Khuyến Nghị)

```bash
# Cách 1: Dùng config mặc định
python run_umls_pipeline.py

# Cách 2: Dùng custom config
python run_umls_pipeline.py --config config/my_umls_config.yaml

# Cách 3: Override specific params
python run_umls_pipeline.py \
    --umls-dir data/umls/2024AB/META \
    --kg-file data/kg_clean.txt \
    --output-dir tmp/umls_mapping
```

### 5.2 Chạy Từng Stage Riêng Lẻ

**Stage 0: Load UMLS (chỉ chạy 1 lần)**
```bash
python run_umls_pipeline.py --stages stage0_umls_loading
```

**Stage 1: Preprocessing**
```bash
python run_umls_pipeline.py --stages stage1_preprocessing
```

**Stage 2 Setup: SapBERT + TF-IDF (chỉ chạy 1 lần, mất 2-3 giờ)**
```bash
python run_umls_pipeline.py --stages stage2_setup_sapbert stage2_setup_tfidf
```

**Stage 2-6: Main Pipeline**
```bash
python run_umls_pipeline.py --stages \
    stage2_candidate_generation \
    stage3_cluster_aggregation \
    stage4_hard_negative_filtering \
    stage5_cross_encoder_reranking \
    stage6_final_output
```

### 5.3 Monitor Progress

**Terminal 1: Chạy pipeline**
```bash
python run_umls_pipeline.py
```

**Terminal 2: Theo dõi logs**
```bash
# Xem logs real-time
tail -f tmp/umls_mapping/pipeline.log

# Hoặc dùng watch để refresh
watch -n 5 'tail -30 tmp/umls_mapping/pipeline.log'
```

**Terminal 3: Check status**
```bash
# Kiểm tra status từng 30s
watch -n 30 'python run_umls_pipeline.py --status'
```

### 5.4 Resume Nếu Bị Gián Đoạn

```bash
# Nếu pipeline bị dừng giữa chừng, resume từ checkpoint
python run_umls_pipeline.py --resume
```

### 5.5 Force Rerun Specific Stage

```bash
# Chạy lại stage cụ thể (bỏ qua cache)
python run_umls_pipeline.py \
    --stages stage3_cluster_aggregation \
    --force
```

---

## 6. Xem Kết Quả

### 6.1 Output Directory Structure

```
tmp/umls_mapping/
├── final_umls_mappings.json          # ⭐ FILE CHÍNH - Kết quả mapping
├── umls_mapping_triples.txt          # Format KG triples
├── mapping_statistics.json           # Thống kê tổng quan
├── manual_review_queue.json          # Mappings cần review thủ công
├── pipeline_metrics.json             # Metrics chi tiết
├── pipeline_report.txt               # Báo cáo văn bản
│
├── visualizations/                   # 📊 Biểu đồ đánh giá
│   ├── candidate_reduction_funnel.png
│   ├── confidence_distribution.png
│   ├── score_progression.png
│   ├── stage_timing.png
│   ├── cluster_statistics.png
│   ├── metric_heatmap.png
│   ├── quality_metrics.png
│   └── visualization_summary.txt
│
├── stage31_preprocessing/            # Kết quả Stage 1
│   ├── entities.txt
│   ├── synonym_clusters.json
│   └── normalized_entities.json
│
├── stage32_candidates.json           # Kết quả Stage 2 (128 candidates)
├── stage33_aggregated.json           # Kết quả Stage 3 (64 candidates)
├── stage34_filtered.json             # Kết quả Stage 4 (32 candidates)
├── stage35_reranked.json             # Kết quả Stage 5 (reranked)
│
└── cache/                            # Cache files (có thể xóa để chạy lại)
    ├── umls_concepts.pkl
    ├── umls_embeddings.pkl
    └── umls_faiss.index
```

### 6.2 Xem File Kết Quả Chính

**A. Final Mappings (JSON format)**
```bash
# Xem structure
head -50 tmp/umls_mapping/final_umls_mappings.json

# Đếm số mappings
jq 'length' tmp/umls_mapping/final_umls_mappings.json

# Xem 1 mapping example
jq 'to_entries | first' tmp/umls_mapping/final_umls_mappings.json
```

**Format của final_umls_mappings.json:**
```json
{
  "diabetes": {
    "cui": "C0020538",
    "name": "Diabetes Mellitus",
    "confidence": 0.89,
    "tier": "high",
    "alternatives": [
      {"cui": "C0011847", "name": "Diabetes", "score": 0.85},
      {"cui": "C0011849", "name": "Diabetes Mellitus, Insulin-Dependent", "score": 0.78}
    ],
    "cluster_size": 3,
    "is_propagated": false,
    "confidence_factors": {
      "score_margin": 0.42,
      "absolute_score": 0.89,
      "cluster_consensus": 0.85,
      "method_agreement": 0.80
    }
  }
}
```

**B. Mapping Statistics**
```bash
# Xem thống kê tổng quan
cat tmp/umls_mapping/mapping_statistics.json | jq .
```

**Output:**
```json
{
  "total_entities": 5000,
  "high_confidence": 3200,
  "medium_confidence": 1300,
  "low_confidence": 500,
  "propagated": 800,
  "high_confidence_pct": "64.00%",
  "medium_confidence_pct": "26.00%",
  "low_confidence_pct": "10.00%",
  "propagated_pct": "16.00%"
}
```

**C. KG Triples Format**
```bash
# Xem triples để thêm vào KG
head -20 tmp/umls_mapping/umls_mapping_triples.txt

# Format: entity | mapped_to_cui | CUI
# Example:
# diabetes | mapped_to_cui | C0020538
# hypertension | mapped_to_cui | C0020538
```

### 6.3 Xem Biểu Đồ Đánh Giá

```bash
# Mở thư mục visualizations
cd tmp/umls_mapping/visualizations

# List tất cả plots
ls -lh *.png

# Mở plots (macOS)
open *.png

# Mở plots (Linux với image viewer)
eog *.png
# hoặc
xdg-open *.png

# Mở plots (Windows)
start *.png
```

**7 Biểu Đồ Được Tạo:**

1. **candidate_reduction_funnel.png** - Funnel chart từ 128→64→32→1 candidates
2. **confidence_distribution.png** - Pie + bar charts phân bố confidence tiers
3. **score_progression.png** - Line chart điểm số qua các stages
4. **stage_timing.png** - Bar chart thời gian chạy từng stage
5. **cluster_statistics.png** - Thống kê cluster sizes
6. **metric_heatmap.png** - Heatmap tất cả metrics
7. **quality_metrics.png** - Dashboard so sánh actual vs target

### 6.4 Đọc Pipeline Report

```bash
# Đọc báo cáo văn bản
cat tmp/umls_mapping/pipeline_report.txt

# Hoặc dùng less để scroll
less tmp/umls_mapping/pipeline_report.txt
```

**Report bao gồm:**
- Overall summary (thời gian, warnings, errors)
- Chi tiết từng stage (duration, input/output, metrics)
- Warnings và errors (nếu có)

---

## 7. Đánh Giá Chất Lượng

### 7.1 Metrics Quan Trọng

**A. Confidence Distribution (Metric chính)**

```bash
# Xem confidence distribution
jq '.high_confidence_pct, .medium_confidence_pct, .low_confidence_pct' \
   tmp/umls_mapping/mapping_statistics.json
```

**✅ Target Quality:**
- **High Confidence ≥ 60%** (tốt nhất)
- **Medium Confidence: 20-30%**
- **Low Confidence < 20%**

**Ví dụ kết quả tốt:**
```
High:   64%  ✓ (target: >60%)
Medium: 26%  ✓ (target: 20-30%)
Low:    10%  ✓ (target: <20%)
```

**B. Average Confidence**

```bash
# Tính average confidence
jq '[.[] | .confidence] | add / length' \
   tmp/umls_mapping/final_umls_mappings.json
```

**✅ Target:** Average confidence > 0.65

**C. Score Margin (Gap giữa top-1 và top-2)**

```bash
# Xem score margin trong metrics
jq '.stages[] | select(.stage_name | contains("Stage 3.6")) | .metrics.avg_score_margin' \
   tmp/umls_mapping/pipeline_metrics.json
```

**✅ Target:** Average score margin > 0.20 (clear winner)

**D. Propagation Rate**

```bash
# Xem tỷ lệ propagated mappings
jq '.propagated_pct' tmp/umls_mapping/mapping_statistics.json
```

**✅ Target:** 10-30% (synonym clusters được leverage tốt)

### 7.2 Quality Checks

**A. Check Low Confidence Mappings**

```bash
# Xem các mappings có confidence thấp
jq 'to_entries | map(select(.value.tier == "low")) | length' \
   tmp/umls_mapping/final_umls_mappings.json

# List 10 low confidence mappings
jq 'to_entries | map(select(.value.tier == "low")) | .[0:10]' \
   tmp/umls_mapping/final_umls_mappings.json
```

**B. Check Manual Review Queue**

```bash
# Số lượng cần review thủ công
jq 'length' tmp/umls_mapping/manual_review_queue.json

# Xem 5 cases đầu tiên
jq 'to_entries | .[0:5]' tmp/umls_mapping/manual_review_queue.json
```

**C. Check Warnings**

```bash
# Xem warnings từ pipeline
jq '.stages[] | select(.warnings | length > 0) | {stage: .stage_name, warnings: .warnings}' \
   tmp/umls_mapping/pipeline_metrics.json
```

**D. Spot Check Random Samples**

```bash
# Random sample 10 mappings để verify thủ công
jq 'to_entries | map(select(.value.tier == "high")) | .[0:10] | .[] | {entity: .key, cui: .value.cui, name: .value.name, confidence: .value.confidence}' \
   tmp/umls_mapping/final_umls_mappings.json
```

### 7.3 Validation với Gold Standard (Nếu có)

Nếu bạn có gold standard annotations:

```python
import json

# Load predictions
with open('tmp/umls_mapping/final_umls_mappings.json') as f:
    predictions = json.load(f)

# Load gold standard
with open('gold_standard.json') as f:
    gold = json.load(f)

# Tính accuracy
correct = 0
total = 0
for entity, gold_cui in gold.items():
    if entity in predictions:
        pred_cui = predictions[entity]['cui']
        if pred_cui == gold_cui:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%}")
print(f"Correct: {correct}/{total}")
```

### 7.4 Error Analysis

**A. Entities Without Candidates**

```bash
# Tìm entities không tìm được CUI nào
grep "No candidates found for:" tmp/umls_mapping/pipeline.log | wc -l
```

**B. Cluster Disagreement**

```bash
# Tìm clusters có outliers
grep "outlier" tmp/umls_mapping/pipeline.log | head -20
```

**C. Hard Negatives**

```bash
# Xem hard negative penalties
jq '.stages[] | select(.stage_name | contains("Stage 3.4")) | .metrics.candidates_with_penalties' \
   tmp/umls_mapping/pipeline_metrics.json
```

---

## 8. Troubleshooting

### 8.1 Pipeline Fails

**Lỗi: CUDA Out of Memory**
```bash
# Solution 1: Giảm batch size
python run_umls_pipeline.py --batch-size 128

# Solution 2: Dùng CPU
python run_umls_pipeline.py --device cpu

# Solution 3: Tắt FP16
python run_umls_pipeline.py --no-amp
```

**Lỗi: UMLS files not found**
```bash
# Kiểm tra paths
ls -la data/umls/2024AB/META/MRCONSO.RRF

# Fix: Chỉ định đúng path
python run_umls_pipeline.py --umls-dir /đúng/path/to/umls/META
```

**Lỗi: Pipeline stopped giữa chừng**
```bash
# Resume từ checkpoint
python run_umls_pipeline.py --resume

# Nếu không work, reset và chạy lại
python run_umls_pipeline.py --reset --force
```

### 8.2 Low Quality Results

**High Confidence < 60%**

Nguyên nhân có thể:
1. UMLS data không match domain của bạn
2. Entity normalization quá aggressive
3. Threshold quá cao

Solutions:
```bash
# 1. Giảm confidence threshold
# Edit config: confidence_high_threshold: 0.65 (từ 0.75)

# 2. Kiểm tra preprocessing
head -100 tmp/umls_mapping/stage31_preprocessing/normalized_entities.json

# 3. Tăng số candidates
# Edit config: ensemble_final_k: 256 (từ 128)
```

**Nhiều Low Confidence Mappings**

```bash
# 1. Check Stage 2 candidate quality
jq '.stages[] | select(.stage_name | contains("Stage 3.2")) | .metrics.avg_top1_score' \
   tmp/umls_mapping/pipeline_metrics.json

# Nếu avg_top1_score < 0.6 → vấn đề ở candidate generation

# 2. Tune parameters
# Tăng sapbert_top_k và tfidf_top_k trong config
```

### 8.3 Performance Issues

**Stage 2 Setup quá chậm (>5 giờ)**

```bash
# 1. Verify GPU được sử dụng
nvidia-smi

# 2. Tăng batch size nếu GPU còn memory
python run_umls_pipeline.py --batch-size 512

# 3. Enable multi-GPU
python run_umls_pipeline.py --no-multi-gpu false
```

**Disk Space Đầy**

```bash
# Kiểm tra space
df -h tmp/umls_mapping/

# Xóa cache không cần thiết (sau khi đã có results)
rm -rf tmp/umls_mapping/cache/*.pkl

# Xóa intermediate files
rm tmp/umls_mapping/stage3*.json
```

### 8.4 Debug Mode

```bash
# Chạy với debug logging
export LOG_LEVEL=DEBUG
python run_umls_pipeline.py

# Hoặc chỉ chạy Stage 2 với sample nhỏ để test
# Edit kg_clean.txt để chỉ có 100 entities đầu
head -100 data/kg_clean.txt > data/kg_clean_sample.txt
python run_umls_pipeline.py --kg-file data/kg_clean_sample.txt
```

---

## 9. Tips & Best Practices

### 9.1 Lần Đầu Chạy

```bash
# 1. Test với sample nhỏ trước
head -500 data/kg_clean.txt > data/kg_clean_sample.txt
python run_umls_pipeline.py --kg-file data/kg_clean_sample.txt

# 2. Kiểm tra kết quả sample
cat tmp/umls_mapping/mapping_statistics.json

# 3. Nếu OK, chạy full dataset
python run_umls_pipeline.py
```

### 9.2 Optimize Performance

```bash
# GPU settings (nếu có GPU mạnh)
python run_umls_pipeline.py \
    --batch-size 512 \
    --device cuda

# CPU settings (nếu không có GPU)
python run_umls_pipeline.py \
    --batch-size 64 \
    --device cpu \
    --num-workers 8
```

### 9.3 Monitoring Long Runs

```bash
# Script để monitor và alert
#!/bin/bash
# monitor_pipeline.sh

while true; do
    STATUS=$(python run_umls_pipeline.py --status 2>&1 | grep "completed")
    echo "[$(date)] $STATUS"

    # Check nếu done
    if echo "$STATUS" | grep -q "6/6 stages completed"; then
        echo "✓ Pipeline DONE!"
        # Send notification (optional)
        # mail -s "Pipeline Done" you@email.com <<< "Pipeline completed"
        break
    fi

    sleep 300  # Check mỗi 5 phút
done
```

### 9.4 Backup Results

```bash
# Backup kết quả quan trọng
tar -czf umls_mapping_results_$(date +%Y%m%d).tar.gz \
    tmp/umls_mapping/final_umls_mappings.json \
    tmp/umls_mapping/mapping_statistics.json \
    tmp/umls_mapping/visualizations/ \
    tmp/umls_mapping/pipeline_report.txt

# Upload to cloud (optional)
# aws s3 cp umls_mapping_results_*.tar.gz s3://your-bucket/
```

---

## 10. Quick Reference

### Commands Cheat Sheet

```bash
# Chạy toàn bộ
python run_umls_pipeline.py

# Chạy với custom config
python run_umls_pipeline.py --config config/my_config.yaml

# Resume
python run_umls_pipeline.py --resume

# Check status
python run_umls_pipeline.py --status

# Reset
python run_umls_pipeline.py --reset

# Force rerun
python run_umls_pipeline.py --force

# Specific stages
python run_umls_pipeline.py --stages stage2_candidate_generation

# Monitor logs
tail -f tmp/umls_mapping/pipeline.log

# View results
jq . tmp/umls_mapping/mapping_statistics.json
```

### File Locations

```
Input:  data/kg_clean.txt
        data/umls/2024AB/META/*.RRF

Output: tmp/umls_mapping/final_umls_mappings.json
        tmp/umls_mapping/visualizations/*.png

Logs:   tmp/umls_mapping/pipeline.log
        tmp/umls_mapping/pipeline_report.txt
```

### Quality Targets

```
✓ High Confidence:      ≥ 60%
✓ Low Confidence:       < 20%
✓ Avg Confidence:       > 0.65
✓ Avg Score Margin:     > 0.20
✓ Propagation Rate:     10-30%
```

---

## 11. Liên Hệ & Tài Liệu

**Documentation:**
- `STAGE3_UMLS_MAPPING_ANALYSIS.md` - Phân tích chi tiết workflow
- `docs/STAGE3_METRICS_GUIDE.md` - Hướng dẫn metrics
- `docs/UMLS_MAPPING_PIPELINE.md` - Pipeline documentation

**UMLS Resources:**
- UMLS Home: https://www.nlm.nih.gov/research/umls/
- UMLS Download: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
- UMLS Documentation: https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/

**Papers:**
- SapBERT: https://arxiv.org/abs/2010.11784
- PubMedBERT: https://arxiv.org/abs/2007.15779

---

**Good luck! 🚀**

Nếu gặp vấn đề, hãy:
1. Check logs: `tmp/umls_mapping/pipeline.log`
2. Check status: `python run_umls_pipeline.py --status`
3. Xem troubleshooting section ở trên
