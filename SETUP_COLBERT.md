# Hướng dẫn Setup ColBERT Model

## Vấn đề đã gặp và Giải pháp

### 1. Lỗi FAISS Clustering (CRITICAL - ĐÃ FIX)

**Vấn đề:**
```
RuntimeError: Error in void faiss::Clustering::train_encoded(...):
Error: 'nx >= k' failed: Number of training points (11) should be at least
as large as number of clusters (32)
```

**Nguyên nhân:**
- `compute_pairwise_similarity()` index 2 entities để tính similarity
- 2 entities → ~8-16 token embeddings
- FAISS mặc định cần ≥32-64 clusters
- Không thể cluster 11 điểm thành 32 clusters

**Giải pháp ✅:**
- **KHÔNG dùng indexing nữa** cho pairwise similarity
- Truy cập ColBERT encoder trực tiếp
- Tính MaxSim thủ công bằng torch operations
- Hoàn toàn tránh FAISS clustering

**Code đã fix tại:** `gfmrag/kg_construction/entity_linking_model/colbert_el_model.py:259-362`

### 2. Model Download & Caching

**Vấn đề:**
- ColBERT model cần tải từ HuggingFace (~450MB)
- ❌ Không thể commit model lên Git (quá lớn)
- ❌ Download mỗi lần chạy rất chậm
- ❌ Lỗi proxy/network khi tải từ HuggingFace (chỉ trong Claude Code container)

**Giải pháp ✅:**
- **RAGatouille tự động cache model** sau lần đầu download
- Chạy `setup_colbert_cache.py` một lần để tải model vào cache
- Các lần sau tự động dùng cache (NHANH ⚡)

## Về HuggingFace Token

### ❓ Tôi có HF_TOKEN trong .env, có giúp gì không?

**TL;DR:** HF token **KHÔNG giải quyết** lỗi proxy/network.

### Khi nào cần HF token:

✅ **Private models** - Models riêng tư
✅ **Gated models** - Models yêu cầu chấp nhận terms
✅ **Rate limiting** - Tăng giới hạn download

### Khi nào KHÔNG cần HF token:

❌ **Public models** - `colbert-ir/colbertv2.0` là public
❌ **Lỗi proxy** - Token không giải quyết "403 Tunnel connection failed"
❌ **Network issues** - Token chỉ cho authentication, không fix network

### Code đã hỗ trợ HF token:

Nếu có token trong `.env`:
```bash
HF_TOKEN=hf_your_token_here
```

Code tự động dùng token khi download (nhưng không giải quyết proxy issue).

## Cách thực hiện

### Bước 1: Setup model cache (chỉ cần 1 lần)

```bash
# Chạy script để tải model vào cache của RAGatouille
poetry run python setup_colbert_cache.py
```

Hoặc nếu không dùng Poetry:
```bash
python setup_colbert_cache.py
```

**Kết quả:**
- Model được lưu trong cache của RAGatouille: `~/.cache/huggingface/`
- Kích thước: ~450MB
- Tự động reuse cho các lần chạy sau

### Bước 2: Code tự động dùng cached model

Code đã được cập nhật để **tự động dùng cached model từ RAGatouille**:

```python
# Khi khởi tạo ColbertELModel
model = ColbertELModel()

# Lần đầu: Download model và cache
# Các lần sau: Tự động dùng cache (NHANH ⚡)
```

**Logic tự động:**
1. RAGatouille check cache tại `~/.cache/huggingface/`
2. Nếu có → load từ cache (NHANH ⚡)
3. Nếu không → download từ HF và cache lại

### Bước 3: Test

```bash
# Chạy test ColBERT pairwise similarity
poetry run python test_colbert_similarity.py
```

**Test này sẽ:**
- Load ColBERT model từ cache
- Test `compute_pairwise_similarity()` với các entity pairs
- **KHÔNG gặp lỗi FAISS clustering nữa** (đã fix bằng direct encoding)

## Lợi ích

✅ **Không cần internet** khi chạy (sau khi đã tải model)
✅ **Nhanh hơn nhiều** (không phải download lại)
✅ **Không làm phình Git repository** (models/ trong .gitignore)
✅ **Tương thích với CI/CD** (tải model trong build step)

## Cấu trúc thư mục

```
GFM/
├── ~/.cache/huggingface/      # ← Model cache (tự động bởi RAGatouille)
│   └── hub/
│       └── models--colbert-ir--colbertv2.0/
├── setup_colbert_cache.py     # ← Script setup cache (chỉ chạy 1 lần)
├── test_colbert_similarity.py # ← Test pairwise similarity
└── gfmrag/
    └── kg_construction/
        └── entity_linking_model/
            └── colbert_el_model.py  # ← Auto dùng cached model + direct encoding
```

### Chi tiết các thay đổi trong code:

**`colbert_el_model.py`:**
1. ✅ `__init__()`: Sử dụng RAGatouille cache, không cần local model path
2. ✅ `compute_pairwise_similarity()`: **FIX FAISS error** - dùng direct encoding thay vì indexing

## Troubleshooting

### ✅ Lỗi FAISS Clustering (ĐÃ FIX)

```
RuntimeError: Number of training points (11) should be at least as large as number of clusters (32)
```

**Giải pháp:** ĐÃ FIX trong code mới - `compute_pairwise_similarity()` không dùng indexing nữa.

### Lỗi: Network/Proxy khi download

**Lưu ý:** Lỗi proxy chỉ xảy ra trong Claude Code container, KHÔNG ảnh hưởng server thật.

**Trên server thật:**
- Chạy `python setup_colbert_cache.py` để tải model vào cache
- Các lần sau tự động dùng cache

**Nếu vẫn gặp network issue:**
- **Option 1:** Tải trên máy có internet, copy `~/.cache/huggingface/` sang
- **Option 2:** Dùng HuggingFace mirror hoặc VPN
- **Option 3:** Đặt `HF_TOKEN` trong `.env` (nếu dùng gated models)

### Lỗi: Disk space

Model cần ~450MB trong cache. Kiểm tra disk space:
```bash
df -h ~
```

## Dành cho CI/CD

Trong CI pipeline, thêm step setup cache:

```yaml
# .github/workflows/test.yml
steps:
  - name: Setup ColBERT cache
    run: |
      poetry run python setup_colbert_cache.py

  - name: Run tests
    run: |
      poetry run python test_colbert_similarity.py
```

Hoặc cache model giữa các runs để tránh download lại:

```yaml
- name: Cache ColBERT model
  uses: actions/cache@v3
  with:
    path: ~/.cache/huggingface
    key: colbert-v2.0-${{ runner.os }}
```

## Tóm tắt các fix

### Fix 1: FAISS Clustering Error ✅
- **Vấn đề:** Indexing 2 entities → không đủ data points cho clustering
- **Fix:** `compute_pairwise_similarity()` dùng **direct encoding** thay vì indexing
- **Kết quả:** Hoàn toàn tránh FAISS, tính MaxSim trực tiếp bằng torch

### Fix 2: Model Caching ✅
- **Vấn đề:** Download model mỗi lần chạy, lỗi proxy/network
- **Fix:** Dùng **RAGatouille cache** tại `~/.cache/huggingface/`
- **Kết quả:** Download 1 lần, reuse vĩnh viễn

## Notes

- Model chỉ cần setup cache **1 lần duy nhất**
- Mỗi developer chạy `setup_colbert_cache.py` trên máy mình
- CI/CD chạy setup trong build step hoặc dùng cache
- Cache tự động được RAGatouille quản lý
- **KHÔNG cần commit model files** vào Git
