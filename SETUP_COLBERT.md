# Hướng dẫn Setup ColBERT Model

## Vấn đề

ColBERT model cần tải từ HuggingFace (~450MB), nhưng:
- ❌ Không thể commit model lên Git (quá lớn)
- ❌ Download mỗi lần chạy rất chậm
- ❌ Lỗi proxy/network khi tải từ HuggingFace

## Giải pháp

**Tải model về local một lần, sau đó code tự động dùng local model.**

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

### Bước 1: Tải model về local

```bash
# Chạy script download (chỉ cần 1 lần)
poetry run python download_colbert_model.py
```

Hoặc nếu không dùng Poetry:
```bash
python download_colbert_model.py
```

**Kết quả:**
- Model được lưu tại: `models/colbert/colbertv2.0/`
- Kích thước: ~450MB
- Không được commit vào Git (đã có trong `.gitignore`)

### Bước 2: Code tự động dùng local model

Code đã được cập nhật để **tự động phát hiện và dùng local model**:

```python
# Khi khởi tạo ColbertELModel
model = ColbertELModel()

# Nếu models/colbert/colbertv2.0/ tồn tại → dùng local
# Nếu không → download từ HuggingFace
```

**Logic tự động:**
1. Kiểm tra `models/colbert/colbertv2.0/` có tồn tại không
2. Nếu có → load từ local (NHANH ⚡)
3. Nếu không → download từ HF (chậm, cần internet)

### Bước 3: Test

```bash
# Chạy test ColBERT
poetry run python test_colbert_similarity.py
```

## Lợi ích

✅ **Không cần internet** khi chạy (sau khi đã tải model)
✅ **Nhanh hơn nhiều** (không phải download lại)
✅ **Không làm phình Git repository** (models/ trong .gitignore)
✅ **Tương thích với CI/CD** (tải model trong build step)

## Cấu trúc thư mục

```
GFM/
├── models/                    # ← Không commit vào Git
│   └── colbert/
│       └── colbertv2.0/      # ← Model files ở đây
│           ├── config.json
│           ├── pytorch_model.bin
│           └── tokenizer files...
├── download_colbert_model.py  # ← Script tải model
├── test_colbert_similarity.py
└── gfmrag/
    └── kg_construction/
        └── entity_linking_model/
            └── colbert_el_model.py  # ← Tự động dùng local model
```

## Troubleshooting

### Lỗi: Module 'transformers' not found

```bash
poetry add transformers
# hoặc
pip install transformers
```

### Lỗi: Network/Proxy khi download

**Option 1:** Tải trên máy có internet, copy folder `models/` sang
**Option 2:** Dùng HuggingFace mirror hoặc VPN
**Option 3:** Download manual từ https://huggingface.co/colbert-ir/colbertv2.0

### Lỗi: Disk space

Model cần ~450MB. Kiểm tra disk space:
```bash
df -h .
```

## Dành cho CI/CD

Trong CI pipeline, thêm step tải model:

```yaml
# .github/workflows/test.yml
steps:
  - name: Download ColBERT model
    run: |
      poetry run python download_colbert_model.py

  - name: Run tests
    run: |
      poetry run python test_colbert_similarity.py
```

Hoặc cache model giữa các runs:

```yaml
- name: Cache ColBERT model
  uses: actions/cache@v3
  with:
    path: models/colbert
    key: colbert-v2.0
```

## Notes

- Model chỉ cần tải **1 lần duy nhất**
- Mỗi developer tải local trên máy mình
- CI/CD tải trong build step hoặc dùng cache
- **KHÔNG commit `models/` vào Git** (đã có trong .gitignore)
