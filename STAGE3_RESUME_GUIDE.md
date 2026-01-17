# Stage 3 Resume Guide: Recalculate Scores & Continue

Hướng dẫn tính lại điểm Stage 3.5 với công thức mới và tiếp tục chạy Stage 3.6 mà không cần chạy lại từ đầu.

## 🎯 Mục đích

Sau khi thay đổi công thức tính điểm (từ 0.7/0.3 → 0.4/0.6) và thêm pre-filtering, bạn có thể:

1. **Tính lại điểm** từ kết quả Stage 3.5 hiện có
2. **Chạy tiếp Stage 3.6** (Confidence Scoring) với kết quả mới
3. **Tiết kiệm thời gian** - không cần chạy lại Stage 3.1-3.5 (tốn nhiều thời gian)

## 📁 Cấu trúc File

```
tmp/umls_mapping/
├── stage31_preprocessing.json       # Từ Stage 3.1 (entities + synonym clusters)
├── stage35_reranked.json            # Từ Stage 3.5 (kết quả CŨ)
├── stage35_reranked_recalculated.json  # Kết quả MỚI sau khi recalculate
├── final_umls_mappings_v2.json      # Final output mới
└── umls_mapping_triples_v2.txt      # KG triples mới
```

## 🚀 Cách sử dụng

### Phương án 1: Chỉ tính lại điểm (không chạy stage 3.6)

```bash
# Tính lại điểm với công thức mới
python recalculate_stage35_scores.py \
    --input tmp/umls_mapping/stage35_reranked.json \
    --output tmp/umls_mapping/stage35_reranked_v2.json \
    --backup

# Kết quả:
# - stage35_reranked.json.backup (backup file gốc)
# - stage35_reranked_v2.json (kết quả mới)
```

**Tham số:**
- `--input`: File input (default: `tmp/umls_mapping/stage35_reranked.json`)
- `--output`: File output (default: ghi đè input)
- `--backup`: Tạo backup trước khi ghi đè
- `--min-prev-score`: Threshold cho previous_score (default: 0.6)
- `--min-cross-score`: Threshold cho cross_encoder_score (default: 0.5)
- `--cross-weight`: Trọng số cross-encoder (default: 0.4)
- `--prev-weight`: Trọng số previous score (default: 0.6)

### Phương án 2: Tính lại điểm VÀ chạy tiếp Stage 3.6 (RECOMMENDED)

```bash
# Chạy toàn bộ: recalculate + stage 3.6
python resume_stage36.py \
    --output-dir tmp/umls_mapping \
    --min-prev-score 0.6 \
    --min-cross-score 0.5 \
    --cross-weight 0.4

# Kết quả:
# - stage35_reranked_recalculated.json (điểm mới)
# - final_umls_mappings_v2.json (final output)
# - umls_mapping_triples_v2.txt (KG triples)
# - mapping_statistics_v2.json (thống kê)
```

**Tham số:**
- `--output-dir`: Thư mục output (default: `tmp/umls_mapping`)
- `--stage35-file`: File stage 3.5 (default: `<output-dir>/stage35_reranked.json`)
- `--stage31-file`: File stage 3.1 (default: `<output-dir>/stage31_preprocessing.json`)
- `--min-prev-score`: Threshold cho previous_score (default: 0.6)
- `--min-cross-score`: Threshold cho cross_encoder_score (default: 0.5)
- `--cross-weight`: Trọng số cross-encoder (default: 0.4)
- `--config`: File config (default: `gfmrag/workflow/config/stage3_umls_mapping.yaml`)

## 📊 Công thức tính điểm

### OLD (0.7/0.3):
```python
final_score = cross_encoder_score × 0.7 + previous_score × 0.3
```

### NEW (0.4/0.6):
```python
final_score = cross_encoder_score × 0.4 + previous_score × 0.6
```

### Pre-filtering:
```python
# Loại bỏ candidates nếu:
if previous_score < 0.6:      # Quá thấp
    filtered_out
if cross_encoder_score < 0.5: # Cross-encoder không chắc chắn
    filtered_out
```

## 🔍 Ví dụ

### Ví dụ 1: False Positive (sẽ bị lọc)

**Trước:**
```json
"0 02": {
  "cui": "C0963088",
  "name": "il 023",
  "previous_score": 0.338,         ❌ < 0.6
  "cross_encoder_score": 0.647,
  "score": 0.554                   ✅ Pass (OLD)
}
```

**Sau:**
```
→ FILTERED OUT (previous_score < 0.6)
```

### Ví dụ 2: True Positive (điểm cao hơn)

**Trước:**
```json
"diuretics treatment": {
  "previous_score": 1.264,
  "cross_encoder_score": 0.602,
  "score": 0.800               # 0.602×0.7 + 1.264×0.3
}
```

**Sau:**
```json
"diuretics treatment": {
  "previous_score": 1.264,     ✅
  "cross_encoder_score": 0.602, ✅
  "score": 0.999               # 0.602×0.4 + 1.264×0.6 ⬆️ CAO HƠN!
}
```

## 📈 Kết quả mong đợi

### Filtering statistics:
- **Filtered**: 10-20% candidates (previous_score < 0.6 hoặc cross_encoder < 0.5)
- **Retained**: 80-90% candidates (chất lượng cao)

### Quality improvement:
- **False positives**: ⬇️ Giảm đáng kể (non-medical entities bị lọc)
- **True positives**: ⬆️ Điểm cao hơn (previous_score được trọng số cao hơn)
- **High confidence %**: ⬆️ Tăng từ 0.35% → 3-5% (dự kiến)

## ⚠️ Lưu ý

### 1. File dependencies

Script `resume_stage36.py` cần:
- ✅ `stage35_reranked.json` (Stage 3.5 output)
- ✅ `stage31_preprocessing.json` (Stage 3.1 output) - **BẮT BUỘC**

Nếu thiếu `stage31_preprocessing.json`, bạn cần:
- Export entities + synonym_clusters từ workflow hiện tại
- Hoặc chạy lại Stage 3.1-3.5 (với config mới)

### 2. Config updates

Đảm bảo config đã được cập nhật (đã commit):
```yaml
# stage3_umls_mapping.yaml
cross_encoder:
  pre_filtering:
    enabled: true
    min_previous_score: 0.6
    min_cross_encoder_score: 0.5

  score_weights:
    cross_encoder: 0.4
    previous_stage: 0.6
```

### 3. Thresholds tuning

Nếu kết quả chưa đạt, thử điều chỉnh thresholds:

**Strict hơn** (ít false positives):
```bash
python resume_stage36.py \
    --min-prev-score 0.7 \      # Tăng từ 0.6
    --min-cross-score 0.6       # Tăng từ 0.5
```

**Loose hơn** (ít false negatives):
```bash
python resume_stage36.py \
    --min-prev-score 0.5 \      # Giảm từ 0.6
    --min-cross-score 0.4       # Giảm từ 0.5
```

**Rebalance weights** (nếu cross-encoder đã fine-tuned):
```bash
python resume_stage36.py \
    --cross-weight 0.6 \        # Tăng nếu cross-encoder tốt hơn
    --min-prev-score 0.5
```

## 🎯 Next Steps

Sau khi chạy xong:

1. **Kiểm tra kết quả**:
   ```bash
   # Xem statistics
   cat tmp/umls_mapping/mapping_statistics_v2.json

   # So sánh với version cũ
   diff tmp/umls_mapping/mapping_statistics.json \
        tmp/umls_mapping/mapping_statistics_v2.json
   ```

2. **Validate một số mappings**:
   ```bash
   # Xem top candidates cho entity cụ thể
   python -c "import json; data=json.load(open('tmp/umls_mapping/final_umls_mappings_v2.json')); print(json.dumps(data['iv dose'], indent=2))"
   ```

3. **Nếu kết quả tốt**:
   - Sử dụng `final_umls_mappings_v2.json` cho downstream tasks
   - Hoặc rename thành `final_umls_mappings.json` (replace version cũ)

4. **Nếu kết quả chưa tốt**:
   - Điều chỉnh thresholds (xem phần "Thresholds tuning")
   - Fine-tune cross-encoder (xem `TRAINING_GUIDE.md`)
   - Hoặc điều chỉnh công thức weights

## 🆘 Troubleshooting

### Lỗi: File not found
```
FileNotFoundError: tmp/umls_mapping/stage31_preprocessing.json
```

**Giải pháp**: File stage 3.1 không tồn tại. Bạn cần:
1. Tìm file output của Stage 3.1 (có thể tên khác)
2. Hoặc chạy lại Stage 3.1 để tạo file

### Lỗi: Entities with ALL candidates filtered out
```
WARNING: Entities with ALL candidates filtered out: 150
```

**Giải pháp**: Thresholds quá strict, giảm xuống:
```bash
python resume_stage36.py --min-prev-score 0.5 --min-cross-score 0.4
```

### Lỗi: Import error
```
ModuleNotFoundError: No module named 'gfmrag'
```

**Giải pháp**: Chạy từ root directory của project:
```bash
cd /home/user/GFM
python resume_stage36.py
```

## 📚 Tài liệu liên quan

- `STAGE3_ARCHITECTURE.md` - Kiến trúc pipeline Stage 3
- `STAGE3_PHASE1_IMPROVEMENTS.md` - Phase 1 improvements
- `TRAINING_GUIDE.md` - Hướng dẫn training cross-encoder
- `gfmrag/umls_mapping/cross_encoder_reranker.py` - Implementation code

---

**Version**: 2.0
**Date**: 2026-01-17
**Author**: Claude (Stage 3 optimization)
