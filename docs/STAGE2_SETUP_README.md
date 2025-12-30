# Stage 2 Setup - SapBERT + TF-IDF

**ONE-TIME SETUP** cho Stage 2 UMLS Candidate Generation.

## Overview

Stage 2 Setup bao gồm 2 tasks ONE-TIME:
- **Task 2.1:** SapBERT embeddings + FAISS index (2-3 giờ GPU, 4-6 giờ CPU)
- **Task 2.2:** TF-IDF vectorizer (10-15 phút)

**Lưu ý:** Sau khi setup xong, **KHÔNG BAO GIỜ** cần chạy lại!

---

## Requirements

### Python Packages

```bash
pip install torch transformers scikit-learn numpy tqdm faiss-cpu
# Hoặc nếu có GPU:
pip install torch transformers scikit-learn numpy tqdm faiss-gpu
```

### Disk Space

- Total: ~25-30 GB
- SapBERT embeddings: ~12 GB
- FAISS index: ~12 GB
- TF-IDF matrix: ~500 MB
- Other files: ~100 MB

### Input Files (từ Stage 1)

- `./outputs/umls_concepts.pkl` (từ Task 1.4)

**Lưu ý về paths:** Nếu Stage 1 output ở path khác (ví dụ: `./tmp/umls_mapping/umls_cache/`), bạn có thể:

1. **Copy files:**
   ```bash
   mkdir -p ./outputs
   cp ./tmp/umls_mapping/umls_cache/umls_concepts.pkl ./outputs/
   ```

2. **Symlink:**
   ```bash
   mkdir -p ./outputs
   ln -s $(pwd)/tmp/umls_mapping/umls_cache/umls_concepts.pkl ./outputs/umls_concepts.pkl
   ```

---

## Usage

### 1. Run Task 2.1 (SapBERT Setup)

```bash
python scripts/task_2_1_sapbert_setup.py
```

**Runtime:** 2-3 giờ trên GPU, 4-6 giờ trên CPU

**Output:**
- `./outputs/umls_embeddings.pkl` (~12 GB)
- `./outputs/umls_faiss.index` (~12 GB)
- `./outputs/umls_cui_order.pkl`

### 2. Run Task 2.2 (TF-IDF Setup)

```bash
python scripts/task_2_2_tfidf_setup.py
```

**Runtime:** 10-15 phút

**Output:**
- `./outputs/tfidf_vectorizer.pkl`
- `./outputs/tfidf_matrix.pkl` (~500 MB)
- `./outputs/alias_to_cuis.pkl`
- `./outputs/all_aliases.pkl`

### 3. Validate Setup

```bash
python scripts/validate_stage2_setup.py
```

---

## Output Files

Tổng cộng **7 files** được tạo ra:

### Task 2.1 - SapBERT

| File | Size | Description |
|------|------|-------------|
| `umls_embeddings.pkl` | ~12 GB | Dict {CUI → embedding vector (768-dim)} |
| `umls_faiss.index` | ~12 GB | FAISS IndexFlatIP for fast similarity search |
| `umls_cui_order.pkl` | ~10 MB | List of CUIs in same order as FAISS index |

### Task 2.2 - TF-IDF

| File | Size | Description |
|------|------|-------------|
| `tfidf_vectorizer.pkl` | ~1 MB | Sklearn TfidfVectorizer (char trigrams) |
| `tfidf_matrix.pkl` | ~500 MB | Sparse matrix (>10M aliases × ~100K features) |
| `alias_to_cuis.pkl` | ~50 MB | Dict {alias → [CUI1, CUI2, ...]} |
| `all_aliases.pkl` | ~50 MB | List of all aliases (same order as matrix rows) |

---

## Success Criteria

✅ Validation script phải pass tất cả checks:

- [ ] All 7 files exist
- [ ] File sizes reasonable (see table above)
- [ ] FAISS index: >4M vectors, dimension 768
- [ ] TF-IDF matrix: (>10M rows, ~100K features)
- [ ] Embeddings count = CUI order count
- [ ] All aliases count >10M

---

## Technical Details

### SapBERT Encoding

- **Model:** `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- **Max length:** 64 tokens
- **Encoding:** CLS token embedding, L2 normalized
- **Dimension:** 768
- **Input:** Preferred names only (not all aliases)

### FAISS Index

- **Type:** IndexFlatIP (Inner Product)
- **Normalization:** L2 normalized vectors
- **Why IP?** After L2 normalization, Inner Product = Cosine Similarity
- **Speed:** Exact search, no approximation

### TF-IDF Vectorizer

- **Analyzer:** Character-level
- **N-grams:** Trigrams (3, 3)
- **Lowercase:** Yes
- **Min DF:** 2 (remove rare trigrams)
- **Max features:** 100,000
- **Input:** All aliases (including preferred names)

---

## Troubleshooting

### Out of Memory (OOM)

**For Task 2.1 (SapBERT):**
- Reduce batch size in encoding (currently processes one at a time)
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`

**For Task 2.2 (TF-IDF):**
- TF-IDF fit_transform is memory-intensive for 10M+ aliases
- Ensure at least 16GB RAM available
- Close other applications

### Slow Performance

**Task 2.1:**
- Use GPU if available (10-20x faster)
- Normal runtime: 2-3 hours GPU, 4-6 hours CPU

**Task 2.2:**
- Normal runtime: 10-15 minutes
- Most time spent in fit_transform

### FAISS Not Found

```bash
# CPU version (works everywhere)
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

---

## Next Steps

Sau khi setup hoàn tất và validation pass:

➡️ **Chuyển sang FILE 02b:** Stage 2 - Generate Candidates

Các files này sẽ được sử dụng bởi `CandidateGenerator` để tạo UMLS candidates cho entities.

---

## Notes

- **One-time only:** Scripts có thể mất vài giờ nhưng chỉ cần chạy 1 lần duy nhất
- **Reusable:** Các files này có thể được reuse cho nhiều datasets/experiments khác nhau
- **Cacheable:** Có thể copy/backup các files này để dùng lại sau
- **Path flexibility:** Scripts có thể chỉnh sửa paths nếu cần thiết

---

## File Locations

Theo requirements, tất cả files output ở `./outputs/`.

Nếu bạn muốn sử dụng path khác, sửa biến `output_dir` trong scripts:

```python
# In scripts/task_2_1_sapbert_setup.py and task_2_2_tfidf_setup.py
output_dir = Path('./outputs')  # Change this to your preferred path
```
