# 🚀 Quick Start - Stage 3: UMLS Mapping Pipeline

Chạy pipeline UMLS Mapping chỉ với một lệnh duy nhất!

## ⚡ Cách Nhanh Nhất

```bash
# Chỉ cần chạy một lệnh
python run_umls_pipeline.py
```

Xong! Pipeline sẽ tự động chạy tất cả các stage với cấu hình tối ưu sẵn.

## 📋 Prerequisites

Trước khi chạy, đảm bảo có:

1. **UMLS files** trong `data/umls/`:
   ```
   data/umls/MRCONSO.RRF
   data/umls/MRSTY.RRF
   data/umls/MRDEF.RRF
   ```

2. **Knowledge Graph** tại `data/kg_clean.txt`

3. **Dependencies** đã cài:
   ```bash
   pip install torch transformers faiss-cpu scikit-learn tqdm
   ```

## 🎯 Các Lệnh Phổ Biến

### Chạy toàn bộ pipeline
```bash
python run_umls_pipeline.py
```

### Tiếp tục từ checkpoint (nếu bị gián đoạn)
```bash
python run_umls_pipeline.py --resume
```

### Kiểm tra tiến trình
```bash
python run_umls_pipeline.py --status
```

### Chạy stage cụ thể
```bash
# Chỉ chạy preprocessing
python run_umls_pipeline.py --stages stage1_preprocessing

# Chạy Stage 2 setup
python run_umls_pipeline.py --stages stage2_setup_sapbert stage2_setup_tfidf
```

### Dùng custom paths
```bash
python run_umls_pipeline.py \
  --umls-dir /path/to/umls \
  --kg-file /path/to/kg_clean.txt \
  --output-dir /path/to/output
```

### Reset và chạy lại từ đầu
```bash
python run_umls_pipeline.py --reset
python run_umls_pipeline.py --force
```

## 🎨 Các Cách Chạy Khác

### Cách 1: Root-level runner (Recommended)
```bash
python run_umls_pipeline.py
```

### Cách 2: Workflow directory
```bash
python workflow/stage3umlsmapping.py
```

### Cách 3: Python module
```bash
python -m workflow.stage3umlsmapping
```

Cả 3 cách đều giống nhau, chọn cách nào cũng được!

## ⏱️ Thời Gian Chạy (Ước Tính)

Với cấu hình tối ưu (default):

```
Stage 0: UMLS Loading         → 5-10 phút
Stage 1: Preprocessing         → 2-5 phút
Stage 2: SapBERT Setup         → 25-40 phút ⚡ (đã tối ưu 3-6x!)
Stage 2: TF-IDF Setup          → 3-5 phút
Stage 2: Generate Candidates   → 5-10 phút
Stages 3-4: Aggregate + Filter → 3-5 phút
Stage 5: Reranking             → 2-3 phút
Stage 6: Final Output          → 1-2 phút

TỔNG: ~60-90 phút
```

## 📁 Kết Quả Đầu Ra

Sau khi chạy xong, kết quả trong `tmp/umls_mapping/`:

```
tmp/umls_mapping/
├── final_umls_mappings.json          # ⭐ File chính
├── stage6_with_confidence.json       # Có confidence scores
├── stage5_reranked.json              # Sau reranking
├── stage4_filtered.json              # Sau filtering
├── stage3_aggregated.json            # Sau aggregation
├── stage2_candidates.json            # Candidates ban đầu
└── cache/
    ├── umls_concepts.pkl
    ├── umls_embeddings.pkl
    └── umls_faiss.index
```

## 🔧 Tùy Chỉnh Performance

### Tăng batch size (nếu có GPU mạnh)
```bash
python run_umls_pipeline.py --batch-size 4096
```

### Giảm batch size (nếu bị out of memory)
```bash
python run_umls_pipeline.py --batch-size 1024
```

### Tắt FP16 (nếu GPU không hỗ trợ)
```bash
python run_umls_pipeline.py --no-amp
```

### Tắt multi-GPU
```bash
python run_umls_pipeline.py --no-multi-gpu
```

## ✅ Validation

Sau khi pipeline chạy xong, validate kết quả:

```bash
# Validation tổng thể
python scripts/final_validation.py

# Validation Stage 1
python scripts/validate_stage1.py

# Validation Stage 2 setup
python scripts/validate_stage2_setup.py
```

## 🆘 Troubleshooting

### Lỗi: CUDA out of memory
```bash
# Giảm batch size
python run_umls_pipeline.py --batch-size 512
```

### Lỗi: UMLS files not found
```bash
# Kiểm tra files
ls -la data/umls/

# Hoặc chỉ định path khác
python run_umls_pipeline.py --umls-dir /path/to/umls
```

### Lỗi: Pipeline bị dừng giữa chừng
```bash
# Tiếp tục từ checkpoint
python run_umls_pipeline.py --resume
```

### Muốn chạy lại từ đầu
```bash
python run_umls_pipeline.py --reset --force
```

## 📚 Tài Liệu Chi Tiết

Xem thêm trong:
- `docs/DEPLOYMENT_GUIDE.md` - Hướng dẫn deployment đầy đủ
- `workflow/README.md` - Workflow documentation
- `docs/QUICK_OPTIMIZATION_GUIDE.md` - Optimization guide

## 💡 Tips

1. **Lần đầu chạy**: Dùng `--status` để theo dõi tiến trình
   ```bash
   # Terminal 1: Chạy pipeline
   python run_umls_pipeline.py

   # Terminal 2: Check status
   watch -n 10 python run_umls_pipeline.py --status
   ```

2. **Debug**: Xem logs chi tiết
   ```bash
   tail -f tmp/umls_mapping/pipeline.log
   ```

3. **Tiết kiệm thời gian**: Cache được lưu tự động, lần chạy sau sẽ nhanh hơn

4. **Interrupt an toàn**: Nhấn Ctrl+C để dừng, sau đó `--resume` để tiếp tục

## 🎉 Quick Check

Test nhanh xem đã setup đúng chưa:

```bash
# 1. Check prerequisites
python run_umls_pipeline.py --status

# 2. Test run first stage only
python run_umls_pipeline.py --stages stage0_umls_loading

# 3. If OK, run full pipeline
python run_umls_pipeline.py
```

---

**Bất kỳ vấn đề gì, check `docs/DEPLOYMENT_GUIDE.md` hoặc `workflow/README.md`**
