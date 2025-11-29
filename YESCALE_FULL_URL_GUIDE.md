# YEScale Full URL Support - Hướng dẫn

## 📋 Tổng quan

Code đã được refactor để **KHÔNG TỰ ĐỘNG APPEND** `/chat/completions` vào URL nữa.

Bạn có thể set **FULL endpoint URL** trong `.env`:
```bash
YESCALE_API_BASE_URL="https://api.yescale.io/v1/chat/completion"
```

Code sẽ dùng **CHÍNH XÁC URL này**, không thêm, không bớt.

---

## 🔧 Cách hoạt động

### Trước đây (Có vấn đề):
```
User set: YESCALE_API_BASE_URL=https://api.yescale.io/v1
Code strip: (nothing to strip)
OpenAI SDK append: /chat/completions
Final URL: https://api.yescale.io/v1/chat/completions ✅ Work

User set: YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completion
Code strip: (failed to match "/chat/completions")
OpenAI SDK append: /chat/completions
Final URL: https://api.yescale.io/v1/chat/completion/chat/completions ❌ 404
```

### Bây giờ (Fixed):
```
User set: YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completion
Code: Use custom YEScaleChatModel (requests library)
Final URL: https://api.yescale.io/v1/chat/completion ✅ Exactly what you set
```

---

## 🚀 Setup

### 1. Tạo file `.env`:

```bash
cd /home/user/GFM

cat > .env << 'EOF'
# YEScale API Configuration
YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completion
YESCALE_API_KEY=sk-xxx

# Hugging Face Token (optional)
HF_TOKEN=hf_xxx
EOF
```

**QUAN TRỌNG:**
- ✅ Dùng **FULL endpoint URL** (bao gồm `/v1/chat/completion`)
- ✅ Code sẽ dùng **chính xác URL này**
- ✅ Không thêm, không bớt, không strip

### 2. Test connection:

```bash
# Load env vars
export $(cat .env | xargs)

# Run test
python test_yescale_connection.py
```

**Expected output:**
```
Test 1: OpenAI SDK (skip nếu không có /chat/completions)
Test 2: Raw Requests
  ✅ SUCCESS! Response: Hi
Test 3: ChatGPT Class
  ✅ SUCCESS! Response: Hi
Test 4: LangChain Model
  [LangChain] Using YEScale API at: https://api.yescale.io/v1/chat/completion (full URL, no appending)
  ✅ SUCCESS! Response: Hi

Total: 3/4 tests passed (Test 1 skipped OK)
```

---

## 📁 Files Changed

### 1. **gfmrag/kg_construction/yescale_chat_model.py** (NEW)

Custom LangChain model dùng `requests` thay vì OpenAI SDK:

```python
class YEScaleChatModel(BaseChatModel):
    api_url: str  # Full URL (e.g., https://api.yescale.io/v1/chat/completion)
    api_key: str
    model: str

    def _generate(self, messages, ...):
        # Use requests.post() directly to api_url
        # No appending, no stripping
        response = requests.post(self.api_url, ...)
```

**Lý do:**
- OpenAI SDK luôn append `/chat/completions` (không config được)
- YEScale endpoint của bạn là `/chat/completion` (không có 's')
- Solution: Dùng `requests` trực tiếp

### 2. **gfmrag/kg_construction/langchain_util.py** (UPDATED)

```python
def init_langchain_model(llm, model_name, ...):
    if llm == "openai":
        yescale_url = os.environ.get("YESCALE_API_BASE_URL")

        if yescale_url:
            # Use custom model (no URL manipulation)
            return YEScaleChatModel(
                api_url=yescale_url,  # Exact URL
                api_key=api_key,
                model=model_name,
                ...
            )
        else:
            # Use OpenAI SDK for official OpenAI API
            return ChatOpenAI(...)
```

**Logic:**
- Nếu có `YESCALE_API_BASE_URL` → Dùng `YEScaleChatModel` (full URL)
- Nếu không → Dùng `ChatOpenAI` (official OpenAI)

### 3. **.env.example** (UPDATED)

```bash
# YEScale API Configuration
# IMPORTANT: Use the FULL endpoint URL (code will NOT append /chat/completions)
# Examples:
#   YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completion
#   YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completions
YESCALE_API_BASE_URL =
YESCALE_API_KEY =
```

---

## ✅ Verification

### Check 1: Test script
```bash
export YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completion
export YESCALE_API_KEY=sk-xxx

python test_yescale_connection.py
```

Should see:
```
[LangChain] Using YEScale API at: https://api.yescale.io/v1/chat/completion (full URL, no appending)
✅ SUCCESS!
```

### Check 2: Run workflow
```bash
python -m gfmrag.workflow.stage1_index_dataset
```

Should see in logs:
```
[LangChain] Using YEScale API at: https://api.yescale.io/v1/chat/completion (full URL, no appending)
```

Should NOT see:
```
❌ Error code: 404 - Invalid URL (POST /v1/chat/completion/chat/completions)
```

---

## 🔍 Components Updated

### ChatGPT Class (`gfmrag/llms/chatgpt.py`)
- ✅ Already uses `requests` library
- ✅ Accepts `api_url` parameter (full URL)
- ✅ No changes needed

### LangChain Models (NER, OpenIE)
- ✅ Now uses `YEScaleChatModel` when `YESCALE_API_BASE_URL` set
- ✅ Dùng `requests` library (not OpenAI SDK)
- ✅ Accepts full endpoint URL

### All tests
- ✅ Test 2 (Raw Requests): Direct HTTP call
- ✅ Test 3 (ChatGPT Class): Uses `api_url` parameter
- ✅ Test 4 (LangChain Model): Uses custom `YEScaleChatModel`

---

## 📊 URL Examples

All these work now:

```bash
# Singular form (your preference)
YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completion

# Plural form
YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completions

# Custom endpoint
YESCALE_API_BASE_URL=https://your-endpoint.com/api/v1/chat/completion

# Any path you want
YESCALE_API_BASE_URL=https://custom.com/your/custom/path
```

**Code sẽ dùng chính xác URL bạn set, không thay đổi gì.**

---

## 🎯 Summary

**What changed:**
1. ✅ Created `YEScaleChatModel` class using `requests` library
2. ✅ Updated `langchain_util.py` to use custom model when YEScale configured
3. ✅ Removed all URL stripping/appending logic
4. ✅ Updated `.env.example` with clear instructions

**What you need to do:**
1. ✅ Set `YESCALE_API_BASE_URL` to your **FULL endpoint URL**
2. ✅ Set `YESCALE_API_KEY` to your API key
3. ✅ Run test script to verify
4. ✅ Run workflow

**Result:**
- ✅ URL bạn set = URL được gọi (chính xác 100%)
- ✅ Không có double append `/chat/completion/chat/completions`
- ✅ Không cần strip suffix
- ✅ Work với bất kỳ endpoint path nào

---

**Last updated:** 2025-11-29
**Status:** ✅ Ready to use
