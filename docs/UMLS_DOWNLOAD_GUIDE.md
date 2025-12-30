# Hướng Dẫn Download UMLS Resources Về Remote Server

Hướng dẫn chi tiết để download UMLS database về remote server khi làm việc qua VSCode Remote SSH.

## 📋 Tổng Quan

UMLS (Unified Medical Language System) là hệ thống thuật ngữ y tế của US National Library of Medicine. Để sử dụng, bạn cần:

1. ✅ Đăng ký tài khoản UMLS (miễn phí)
2. ✅ Download UMLS Full Release
3. ✅ Extract các file RRF cần thiết
4. ✅ Đặt vào đúng thư mục trong project

## 🔐 Bước 1: Đăng Ký Tài Khoản UMLS (Chỉ làm 1 lần)

### 1.1. Truy cập trang đăng ký

```
https://uts.nlm.nih.gov/uts/signup-login
```

### 1.2. Tạo tài khoản mới

- Click **"Request a UTS Account"**
- Điền thông tin cá nhân
- Chọn **Purpose of Use**: Research (hoặc phù hợp với mục đích của bạn)
- Đồng ý với UMLS Metathesaurus License Agreement
- Submit form

### 1.3. Xác nhận email

- Check email để xác nhận tài khoản
- Đăng nhập vào UTS với tài khoản mới tạo

**⏱️ Thời gian**: Tài khoản được kích hoạt ngay lập tức sau khi xác nhận email

## 📥 Bước 2: Download UMLS Full Release

### 2.1. Truy cập trang download

Sau khi đăng nhập UTS:

```
https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
```

Hoặc trực tiếp:

```
https://download.nlm.nih.gov/umls/kss/
```

### 2.2. Chọn version

Recommendation: **UMLS 2024AA** (version mới nhất)

```
https://download.nlm.nih.gov/umls/kss/2024AA/umls-2024AA-full.zip
```

**Kích thước**: ~6-8 GB (nén), ~30-40 GB (giải nén)

### 2.3. Download về remote server

Bạn có **3 cách** để download về remote server:

## 🚀 Phương Pháp Download

### **Phương Pháp 1: Download Trực Tiếp Trên Remote Server (Khuyến Nghị)**

Nhanh nhất nếu remote server có internet tốt.

#### Bước 1: SSH vào remote server

Trong VSCode, mở terminal (Ctrl + ` hoặc Cmd + `), terminal sẽ tự động SSH vào remote server.

#### Bước 2: Tạo thư mục và di chuyển vào đó

```bash
cd /home/user/GFM
mkdir -p data/umls
cd data/umls
```

#### Bước 3: Download bằng wget (cần authentication)

**Option A: Sử dụng wget với UMLS API Key**

```bash
# Lấy API key từ https://uts.nlm.nih.gov/uts/profile (sau khi login)
# Tại mục "API Authentication"

# Download với API key
wget --post-data "apikey=YOUR_UMLS_API_KEY" \
  "https://download.nlm.nih.gov/umls/kss/2024AA/umls-2024AA-full.zip"
```

**Option B: Download với session cookie (dễ hơn)**

```bash
# Bước 1: Trên máy local, login vào UMLS website
# Bước 2: Copy download link trực tiếp

# Sử dụng curl với authentication
curl -u "YOUR_UTS_USERNAME:YOUR_UTS_PASSWORD" \
  -o umls-2024AA-full.zip \
  "https://download.nlm.nih.gov/umls/kss/2024AA/umls-2024AA-full.zip"
```

**Option C: Sử dụng UTS Download Tool (Recommended)**

```bash
# Download UTS Download Tool
wget https://download.nlm.nih.gov/umls/kss/downloads/uts-download.jar

# Chạy download tool với credentials
java -jar uts-download.jar \
  -username YOUR_UTS_USERNAME \
  -password YOUR_UTS_PASSWORD \
  -version 2024AA

# Tool sẽ tự động download và extract
```

#### Bước 4: Giải nén

```bash
# Giải nén file zip
unzip umls-2024AA-full.zip

# Di chuyển vào thư mục chứa META files
cd 2024AA-full
unzip mmsys.zip
cd META
```

Các file cần thiết sẽ ở trong thư mục `META/`:
- MRCONSO.RRF
- MRSTY.RRF
- MRDEF.RRF

---

### **Phương Pháp 2: Download Local → Upload Lên Server**

Nếu remote server có internet chậm hoặc bị hạn chế.

#### Bước 1: Download về máy local

1. Truy cập https://download.nlm.nih.gov/umls/kss/2024AA/
2. Login với UTS account
3. Download `umls-2024AA-full.zip` về máy (6-8 GB)

#### Bước 2: Upload lên remote server qua VSCode

**Option A: Sử dụng VSCode Remote Explorer**

1. Trong VSCode, mở **Explorer** (Ctrl+Shift+E)
2. Right-click vào thư mục `data/umls`
3. Chọn **"Upload..."**
4. Chọn file `umls-2024AA-full.zip` từ máy local

**Option B: Sử dụng SCP command**

```bash
# Trên máy local (Terminal riêng, không phải VSCode terminal)
scp umls-2024AA-full.zip username@remote-server:/home/user/GFM/data/umls/
```

#### Bước 3: Giải nén trên server

```bash
# Trong VSCode terminal (đã SSH vào remote)
cd /home/user/GFM/data/umls
unzip umls-2024AA-full.zip
cd 2024AA-full
unzip mmsys.zip
```

---

### **Phương Pháp 3: Download Chỉ Các File Cần Thiết (Nhanh Nhất)**

Nếu bạn chỉ cần 3 files RRF (thay vì toàn bộ 40GB).

UMLS không cung cấp download riêng lẻ, nhưng bạn có thể:

#### Option A: Tự động extract chỉ files cần thiết

```bash
cd /home/user/GFM/data/umls

# Download full zip
wget [DOWNLOAD_URL] -O umls-2024AA-full.zip

# Extract chỉ files cần thiết (nhanh hơn)
unzip -p umls-2024AA-full.zip "*/META/MRCONSO.RRF" > MRCONSO.RRF
unzip -p umls-2024AA-full.zip "*/META/MRSTY.RRF" > MRSTY.RRF
unzip -p umls-2024AA-full.zip "*/META/MRDEF.RRF" > MRDEF.RRF

# Xóa file zip để tiết kiệm dung lượng
rm umls-2024AA-full.zip
```

#### Option B: Download subset từ NLM FTP (nếu có)

Một số version UMLS có sẵn trên FTP:

```bash
# Check FTP directory
curl -l ftp://ftp.nlm.nih.gov/umls/

# Download nếu có subset
# (Không phải lúc nào cũng có sẵn)
```

---

## 📂 Bước 3: Tổ Chức Files

Sau khi download và extract, đảm bảo cấu trúc thư mục như sau:

```
/home/user/GFM/
└── data/
    └── umls/
        ├── MRCONSO.RRF    # ~4.5 GB
        ├── MRSTY.RRF      # ~50 MB
        └── MRDEF.RRF      # ~500 MB
```

### Script tự động copy files

```bash
#!/bin/bash
# copy_umls_files.sh

cd /home/user/GFM/data/umls

# Tìm và copy các file RRF cần thiết
find . -name "MRCONSO.RRF" -exec cp {} . \;
find . -name "MRSTY.RRF" -exec cp {} . \;
find . -name "MRDEF.RRF" -exec cp {} . \;

# Xóa các thư mục tạm
rm -rf 2024AA-full/
rm -f umls-2024AA-full.zip

echo "✅ UMLS files ready!"
ls -lh *.RRF
```

Chạy script:

```bash
chmod +x copy_umls_files.sh
./copy_umls_files.sh
```

---

## ✅ Bước 4: Verify Download

Kiểm tra các files đã đúng:

```bash
cd /home/user/GFM/data/umls

# Kiểm tra file tồn tại
ls -lh MRCONSO.RRF MRSTY.RRF MRDEF.RRF

# Kiểm tra số dòng (ước tính cho 2024AA)
wc -l MRCONSO.RRF  # ~15-17 triệu dòng
wc -l MRSTY.RRF    # ~1.5-2 triệu dòng
wc -l MRDEF.RRF    # ~500k-1 triệu dòng

# Xem nội dung 10 dòng đầu
head MRCONSO.RRF
head MRSTY.RRF
head MRDEF.RRF
```

Expected output:

```
MRCONSO.RRF:  4.5 GB,  ~16 triệu dòng
MRSTY.RRF:    50 MB,   ~1.8 triệu dòng
MRDEF.RRF:    500 MB,  ~800k dòng
```

---

## 🎯 Bước 5: Test Pipeline

Sau khi có files, test ngay:

```bash
cd /home/user/GFM

# Test với Stage 0 (UMLS loading)
python run_umls_pipeline.py --stages stage0_umls_loading
```

Nếu thành công, bạn sẽ thấy:

```
✓ Prerequisites validation passed
✓ Directory ready: ./tmp/umls_mapping
🚀 Initializing UMLS Mapping Pipeline...

Stage 0: UMLS Loading
  Loading MRCONSO.RRF...
  Loading MRSTY.RRF...
  Loading MRDEF.RRF...
  ✅ Loaded 4,523,671 concepts
  ✅ Loaded 1,834,582 semantic types
  ✅ Created cache files
```

---

## 🔧 Troubleshooting

### Issue 1: wget download bị lỗi 401/403

**Nguyên nhân**: Cần authentication

**Giải pháp**:

```bash
# Option A: Sử dụng curl với credentials
curl -u "username:password" -o umls.zip [DOWNLOAD_URL]

# Option B: Download manual rồi upload
# Xem Phương Pháp 2 ở trên
```

### Issue 2: Không đủ dung lượng trên server

**Kiểm tra dung lượng**:

```bash
df -h /home/user/GFM/data
```

**Giải pháp**:

1. **Download trực tiếp các file RRF** (Phương Pháp 3) - chỉ cần ~5GB thay vì 40GB
2. **Xóa file zip sau khi extract**:
   ```bash
   rm umls-2024AA-full.zip
   rm -rf 2024AA-full/
   ```
3. **Mount external storage** nếu server cho phép

### Issue 3: Giải nén quá lâu

**Giải pháp**: Sử dụng parallel extraction

```bash
# Thay vì unzip thông thường, dùng pigz (parallel gzip)
sudo apt-get install pigz unzip

# Extract nhanh hơn
pigz -dc umls-2024AA-full.zip | tar -x
```

### Issue 4: Upload qua VSCode bị timeout

**Giải pháp**: Split file thành chunks nhỏ hơn

```bash
# Trên máy local
split -b 1G umls-2024AA-full.zip umls-part-

# Upload từng part qua VSCode
# Sau đó trên server:
cat umls-part-* > umls-2024AA-full.zip
```

### Issue 5: Không có quyền truy cập UMLS

**Giải pháp**:

- Kiểm tra tài khoản UTS đã được approve chưa
- License agreement phải được accept
- Một số tổ chức cần approval từ administrator

---

## 📊 Comparison: Các Phương Pháp Download

| Phương Pháp | Tốc độ | Dung lượng cần | Độ phức tạp | Khuyến nghị |
|-------------|--------|----------------|-------------|-------------|
| **1. Direct wget/curl** | ⭐⭐⭐⭐⭐ Nhanh nhất | 40 GB | Dễ | ✅ Nếu server có internet tốt |
| **2. Local → Upload** | ⭐⭐ Chậm | 8 GB transfer | Trung bình | Nếu server internet chậm |
| **3. Extract RRF only** | ⭐⭐⭐⭐ Nhanh | 5 GB | Dễ | ✅ Tiết kiệm dung lượng |

---

## 🚀 Quick Start Script

Script hoàn chỉnh để download và setup UMLS:

```bash
#!/bin/bash
# setup_umls.sh

echo "🚀 UMLS Setup for Remote Server"
echo "================================"

# Configuration
PROJECT_ROOT="/home/user/GFM"
UMLS_DIR="$PROJECT_ROOT/data/umls"
UMLS_VERSION="2024AA"

# Prompt for credentials
read -p "Enter UTS Username: " UTS_USER
read -sp "Enter UTS Password: " UTS_PASS
echo ""

# Create directory
mkdir -p "$UMLS_DIR"
cd "$UMLS_DIR"

# Download with curl
echo "📥 Downloading UMLS $UMLS_VERSION..."
curl -u "$UTS_USER:$UTS_PASS" \
  -o "umls-$UMLS_VERSION-full.zip" \
  "https://download.nlm.nih.gov/umls/kss/$UMLS_VERSION/umls-$UMLS_VERSION-full.zip"

# Extract
echo "📦 Extracting files..."
unzip "umls-$UMLS_VERSION-full.zip"
cd "$UMLS_VERSION-full"
unzip mmsys.zip

# Copy RRF files
echo "📂 Copying RRF files..."
cp META/MRCONSO.RRF "$UMLS_DIR/"
cp META/MRSTY.RRF "$UMLS_DIR/"
cp META/MRDEF.RRF "$UMLS_DIR/"

# Cleanup
echo "🧹 Cleaning up..."
cd "$UMLS_DIR"
rm -rf "$UMLS_VERSION-full"
rm "umls-$UMLS_VERSION-full.zip"

# Verify
echo "✅ Verification:"
ls -lh MRCONSO.RRF MRSTY.RRF MRDEF.RRF
echo ""
echo "Files ready at: $UMLS_DIR"
echo ""
echo "Next step: Run pipeline"
echo "  cd $PROJECT_ROOT"
echo "  python run_umls_pipeline.py"
```

Sử dụng:

```bash
chmod +x setup_umls.sh
./setup_umls.sh
```

---

## 📚 Tài Liệu Tham Khảo

- **UMLS Homepage**: https://www.nlm.nih.gov/research/umls/
- **UTS (Account Management)**: https://uts.nlm.nih.gov/
- **Download Center**: https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html
- **UMLS Reference Manual**: https://www.ncbi.nlm.nih.gov/books/NBK9676/
- **File Formats**: https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/index.html

---

## 💡 Tips

1. **Cache credentials**: Lưu API key vào environment variable
   ```bash
   export UMLS_API_KEY="your_key_here"
   ```

2. **Resume download**: Dùng `wget -c` hoặc `curl -C -` để resume nếu bị ngắt
   ```bash
   curl -C - -u "$USER:$PASS" -o umls.zip [URL]
   ```

3. **Monitor progress**: Dùng `pv` để xem tiến trình
   ```bash
   sudo apt-get install pv
   pv umls-2024AA-full.zip | unzip -
   ```

4. **Background download**: Chạy trong screen/tmux để tránh mất kết nối
   ```bash
   screen -S umls_download
   ./setup_umls.sh
   # Ctrl+A, D để detach
   # screen -r umls_download để attach lại
   ```

---

**🎉 Hoàn thành! Bạn đã sẵn sàng chạy UMLS Mapping Pipeline!**
