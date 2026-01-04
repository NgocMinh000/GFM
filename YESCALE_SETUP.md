# YEScale API Setup Guide

## 🐛 Problem

Khi chạy Stage 2 Entity Resolution, bạn gặp lỗi:

```
[2025-12-10 04:00:18,491][__main__][WARNING] - LLM inference failed for entity 'autosomal dominant ataxias':
The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable

[LangChain] Using OpenAI API (official)
```

**Nguyên nhân:** YEScale API credentials chưa được cấu hình đúng.

---

## ✅ Giải Pháp

### **Bước 1: Lấy YEScale API Key**

1. Truy cập YEScale dashboard
2. Tạo hoặc copy API key của bạn (dạng `sk-xxxxx`)
3. Lưu lại API key này

### **Bước 2: Set Environment Variables**

#### **Option A: Tạm thời (chỉ session hiện tại)**

```bash
# Trong terminal, chạy:
export YESCALE_API_BASE_URL="https://api.yescale.io/v1/chat/completions"
export YESCALE_API_KEY="sk-your-actual-api-key-here"

# Verify
echo $YESCALE_API_BASE_URL
echo $YESCALE_API_KEY
```

#### **Option B: Vĩnh viễn (recommended)**

Thêm vào file `~/.bashrc` (hoặc `~/.zshrc` nếu dùng zsh):

```bash
# Mở file
nano ~/.bashrc

# Thêm vào cuối file:
export YESCALE_API_BASE_URL="https://api.yescale.io/v1/chat/completions"
export YESCALE_API_KEY="sk-your-actual-api-key-here"

# Lưu file (Ctrl+O, Enter, Ctrl+X)

# Reload bashrc
source ~/.bashrc

# Verify
echo $YESCALE_API_BASE_URL
echo $YESCALE_API_KEY
```

### **Bước 3: Verify Configuration**

```bash
# Run test script
bash test_yescale_setup.sh

# Expected output if configured correctly:
# ✅ YESCALE_API_BASE_URL is set
# ✅ YESCALE_API_KEY is set
# ✅ Configuration COMPLETE
```

---

## 🔧 YEScale API Endpoint

**Correct endpoint:**
```
https://api.yescale.io/v1/chat/completions
```

**Important:** Đây là endpoint đầy đủ, không cần append thêm path nào.

**API Format (theo YEScale docs):**
```python
import requests
import json

url = "https://api.yescale.io/v1/chat/completions"

payload = json.dumps({
   "model": "gpt-4o-mini",
   "messages": [
      {
         "role": "user",
         "content": "Hello!"
      }
   ],
   "max_tokens": 1000
})

headers = {
   'Accept': 'application/json',
   'Authorization': f'Bearer {YOUR_API_KEY}',
   'Content-Type': 'application/json'
}

response = requests.post(url, headers=headers, data=payload)
print(response.text)
```

---

## 📊 How It Works Now

### **With YEScale Configured:**

```python
# Stage 0 Type Inference uses 3 methods:
1. Pattern-Based (weight: 0.2)
2. Relationship-LLM (weight: 0.4) ← Uses YEScale API ✅
3. Zero-shot (weight: 0.4)

# All 3 methods work
# Weighted voting: 0.2 + 0.4 + 0.4 = 1.0
```

### **Without YEScale Configured:**

```python
# Stage 0 Type Inference uses 2 methods:
1. Pattern-Based (weight: 0.6)  ← Increased weight
2. Zero-shot (weight: 0.4)

# LLM method skipped (graceful degradation)
# Warning logged once at startup
# Weighted voting: 0.6 + 0.4 = 1.0
```

---

## 🚀 Running Stage 2

### **After Configuration:**

```bash
# Activate environment
conda activate gfm-rag

# Verify YEScale setup
bash test_yescale_setup.sh

# Clear cache (important!)
rm -rf tmp/entity_resolution/stage0_*

# Run Stage 2
python -m gfmrag.workflow.stage2_entity_resolution
```

**Expected logs if configured correctly:**
```
================================================================================
STAGE 0: ENHANCED TYPE INFERENCE (4-Step Hybrid)
================================================================================
✅ Initialized YEScale LLM for relationship inference: https://api.yescale.io/v1/chat/completions
Method: hybrid
Processing 691 unique entities...
Architecture: Pattern → Relationship-LLM → Zero-shot → Hybrid Decision
Type inference (4-step): 100%|████████████████| 691/691 [04:02<00:00]
```

**Expected logs if NOT configured:**
```
================================================================================
STAGE 0: ENHANCED TYPE INFERENCE (4-Step Hybrid)
================================================================================
[WARNING] YEScale API not configured (YESCALE_API_BASE_URL or YESCALE_API_KEY missing).
Skipping LLM-based relationship inference.
Set YESCALE_API_BASE_URL and YESCALE_API_KEY environment variables to enable.

Method: hybrid
Processing 691 unique entities...
Architecture: Pattern → (LLM skipped) → Zero-shot → Hybrid Decision
Type inference (4-step): 100%|████████████████| 691/691 [02:30<00:00]
```

---

## ❓ FAQ

### **Q: Tôi có thể dùng OpenAI API key thay vì YEScale không?**

A: Không được. Code hiện tại yêu cầu `YESCALE_API_BASE_URL` để xác định endpoint. Nếu chỉ có `OPENAI_API_KEY` mà không có `YESCALE_API_BASE_URL`, LLM method sẽ bị skip.

### **Q: Điều gì xảy ra nếu tôi không set YEScale credentials?**

A:
- ✅ Code vẫn chạy (không crash)
- ⚠️  LLM-based relationship inference bị skip
- ✅ Pattern + Zero-shot vẫn hoạt động
- ⚠️  Type classification có thể kém chính xác hơn (thiếu 40% weight từ LLM)

### **Q: Làm sao tôi biết YEScale đang được dùng?**

A: Kiểm tra logs khi chạy Stage 2. Nếu thấy:
```
✅ Initialized YEScale LLM for relationship inference: https://api.yescale.io/v1/chat/completions
```
→ YEScale đang được dùng ✅

Nếu thấy:
```
[WARNING] YEScale API not configured...
```
→ YEScale KHÔNG được dùng ❌

### **Q: API key của tôi có bị lộ trong logs không?**

A: Không. API key không bao giờ được log. Chỉ có URL endpoint được log (không chứa sensitive info).

---

## 🔍 Debugging

### **Test 1: Check Environment Variables**

```bash
bash test_yescale_setup.sh
```

### **Test 2: Python Test**

```python
import os

yescale_url = os.environ.get("YESCALE_API_BASE_URL")
yescale_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")

print(f"URL: {yescale_url}")
print(f"Key: {yescale_key[:10] if yescale_key else None}...")

if yescale_url and yescale_key:
    print("✅ Configuration OK")
else:
    print("❌ Configuration MISSING")
```

### **Test 3: Manual API Call**

```bash
curl -X POST "https://api.yescale.io/v1/chat/completions" \
  -H "Accept: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

---

## 📝 Checklist

Before running Stage 2, make sure:

- [ ] `YESCALE_API_BASE_URL` is set
- [ ] `YESCALE_API_KEY` (or `OPENAI_API_KEY`) is set
- [ ] `test_yescale_setup.sh` passes
- [ ] Old cache cleared: `rm -rf tmp/entity_resolution/stage0_*`
- [ ] `conda activate gfm-rag` activated

---

## 🎯 Summary

| Configuration | Pattern | LLM | Zero-shot | Total Weight |
|---------------|---------|-----|-----------|--------------|
| **With YEScale** | 0.2 | 0.4 ✅ | 0.4 | 1.0 |
| **Without YEScale** | 0.6 | skipped ❌ | 0.4 | 1.0 |

**Recommendation:** Cấu hình YEScale để có kết quả tốt nhất (LLM method thường chính xác hơn pattern-based).

---

## 📚 Related Files

- **test_yescale_setup.sh**: Script kiểm tra configuration
- **BUGFIXES_STAGE0_STAGE2.md**: Bug fixes documentation
- **STAGE0_ENHANCED.md**: Stage 0 architecture details
- **gfmrag/workflow/stage2_entity_resolution.py:437-475**: LLM initialization code

---

**Commit:** `cc1ec76`
**File:** `gfmrag/workflow/stage2_entity_resolution.py`
**Status:** ✅ Fixed and tested
