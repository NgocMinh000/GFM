# 📥 Quick Reference: Download UMLS về Remote Server

## 🚀 Cách Nhanh Nhất (Recommended)

### Option 1: Chạy Script Tự Động

```bash
cd /home/user/GFM

# Chạy script setup tự động
bash scripts/setup_umls.sh
```

Script sẽ tự động:
- ✅ Check dependencies và disk space
- ✅ Prompt nhập UTS credentials
- ✅ Download UMLS (~8GB)
- ✅ Extract files cần thiết
- ✅ Cleanup để tiết kiệm dung lượng
- ✅ Verify installation

**⏱️ Thời gian**: ~30-60 phút (tùy internet)

---

### Option 2: Quick Download (RRF only)

Nếu chỉ cần 3 files RRF (~5GB thay vì 40GB):

```bash
cd /home/user/GFM

# Chạy quick download script
bash scripts/quick_download_umls.sh
```

Script sẽ hỏi username/password và download chỉ files cần thiết.

---

### Option 3: Manual Commands

#### Bước 1: Tạo thư mục

```bash
cd /home/user/GFM
mkdir -p data/umls
cd data/umls
```

#### Bước 2: Download với curl

```bash
# Thay YOUR_USERNAME và YOUR_PASSWORD
curl -u "YOUR_USERNAME:YOUR_PASSWORD" \
  -C - \
  -o umls-2024AA-full.zip \
  "https://download.nlm.nih.gov/umls/kss/2024AA/umls-2024AA-full.zip"
```

#### Bước 3: Extract

```bash
# Extract main archive
unzip umls-2024AA-full.zip

# Tìm và extract mmsys
cd 2024AA-full
unzip mmsys.zip

# Copy RRF files
cp META/MRCONSO.RRF ../
cp META/MRSTY.RRF ../
cp META/MRDEF.RRF ../

# Cleanup
cd ..
rm -rf 2024AA-full umls-2024AA-full.zip
```

#### Bước 4: Verify

```bash
ls -lh data/umls/*.RRF
```

Expected:
```
MRCONSO.RRF  ~4.5 GB
MRSTY.RRF    ~50 MB
MRDEF.RRF    ~500 MB
```

---

## 📋 Prerequisites

### 1. Đăng ký UTS Account (miễn phí)

```
https://uts.nlm.nih.gov/uts/signup-login
```

- Click "Request a UTS Account"
- Điền thông tin
- Chọn Purpose: Research
- Accept license agreement
- Verify email

**⏱️ Thời gian**: 5 phút (kích hoạt ngay)

### 2. Check Server Requirements

```bash
# Check disk space (cần ít nhất 10GB free)
df -h /home/user/GFM

# Check dependencies
which curl unzip

# Install nếu thiếu
sudo apt-get install curl unzip
```

---

## 🔧 Nếu Gặp Lỗi

### Lỗi: Authentication failed

```bash
# Kiểm tra username/password
# Đăng nhập web để confirm account đã active
# https://uts.nlm.nih.gov
```

### Lỗi: No space left on device

```bash
# Check dung lượng
df -h

# Dọn dẹp nếu cần
rm -rf /tmp/*
docker system prune -a  # Nếu dùng Docker
```

### Lỗi: Download bị ngắt giữa chừng

```bash
# Chạy lại script - sẽ tự động resume
bash scripts/setup_umls.sh

# Hoặc dùng curl với -C - để resume
curl -C - -u "user:pass" -o umls.zip [URL]
```

### Lỗi: Permission denied

```bash
# Check quyền thư mục
ls -la data/

# Sửa quyền nếu cần
chmod 755 data/umls
```

---

## ✅ Verify Installation

```bash
cd /home/user/GFM

# Check files
ls -lh data/umls/

# Should show:
# MRCONSO.RRF
# MRSTY.RRF
# MRDEF.RRF

# Test pipeline
python run_umls_pipeline.py --stages stage0_umls_loading
```

Nếu thành công sẽ thấy:

```
✓ Prerequisites validation passed
✓ Loading MRCONSO.RRF...
✓ Loading MRSTY.RRF...
✓ Loading MRDEF.RRF...
✅ Loaded 4,523,671 concepts
```

---

## 🎯 Quick Commands

```bash
# Download tự động (recommended)
bash scripts/setup_umls.sh

# Download nhanh (RRF only)
bash scripts/quick_download_umls.sh

# Manual download với credentials từ env
export UTS_USER="your_username"
export UTS_PASS="your_password"
bash scripts/quick_download_umls.sh

# Check status sau download
ls -lh data/umls/*.RRF
wc -l data/umls/*.RRF

# Test ngay
python run_umls_pipeline.py --stages stage0_umls_loading
```

---

## 📍 File Locations

Sau khi download xong:

```
/home/user/GFM/
└── data/
    └── umls/
        ├── MRCONSO.RRF    # 4.5 GB - Main concepts
        ├── MRSTY.RRF      # 50 MB  - Semantic types
        └── MRDEF.RRF      # 500 MB - Definitions
```

---

## 📚 Chi Tiết Hơn

Xem tài liệu đầy đủ:

```bash
cat docs/UMLS_DOWNLOAD_GUIDE.md
```

Hoặc xem online: `docs/UMLS_DOWNLOAD_GUIDE.md`

---

## 💡 Tips

1. **Dùng screen/tmux** để tránh mất kết nối:
   ```bash
   screen -S umls
   bash scripts/setup_umls.sh
   # Ctrl+A, D để detach
   ```

2. **Monitor progress**:
   ```bash
   # Terminal khác
   watch -n 5 'ls -lh data/umls/'
   ```

3. **Save credentials** (optional):
   ```bash
   # Trong ~/.bashrc hoặc ~/.zshrc
   export UTS_USER="your_username"
   export UTS_PASS="your_password"
   ```

4. **Resume download nếu ngắt**:
   ```bash
   # Script tự động resume
   bash scripts/setup_umls.sh
   ```

---

**🎉 Sau khi download xong, chạy pipeline:**

```bash
python run_umls_pipeline.py
```
