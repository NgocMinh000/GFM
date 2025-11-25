# Hướng dẫn tích hợp YEScale LLM API

## Giới thiệu

Repo này hiện đã hỗ trợ kết nối với YEScale API - một API tương thích với OpenAI. Bạn có thể sử dụng YEScale thay vì OpenAI API bằng cách cấu hình các biến môi trường.

## Cấu hình

### Bước 1: Tạo file `.env`

Sao chép file `.env.example` thành `.env`:

```bash
cp .env.example .env
```

### Bước 2: Cấu hình API credentials

Mở file `.env` và điền thông tin YEScale của bạn:

```bash
# YEScale API Configuration
YESCALE_API_BASE_URL=https://your-yescale-endpoint.com/v1
YESCALE_API_KEY=your-yescale-api-key-here
```

**Lưu ý:**
- `YESCALE_API_BASE_URL`: Full URL endpoint của YEScale API (bao gồm `/chat/completions`)
- `YESCALE_API_KEY`: API key bạn nhận được từ YEScale

### Bước 3: Hiểu cách hoạt động (QUAN TRỌNG!)

**YEScale API = OpenAI API** (100% tương thích format)

Theo tài liệu API của YEScale, endpoint chat completions là:
```
POST /v1/chat/completions
Authorization: Bearer {{YOUR_API_KEY}}
Body: {"model": "gpt-4o", "messages": [...], "max_tokens": 1000}
```

**THAY ĐỔI MỚI:** Code đã được refactor để sử dụng `requests` library thay vì OpenAI SDK!

Khi bạn cấu hình:
```bash
YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completions
```

Code sẽ:
1. ✅ **ChatGPT class**: Gọi trực tiếp đến URL này bằng `requests.post()`
2. ✅ **LangChain models**: Tự động strip `/chat/completions`, để OpenAI SDK append lại
3. ✅ Thêm header: `Authorization: Bearer {YESCALE_API_KEY}`
4. ✅ Format request/response giống hệt OpenAI API

**⚠️ Lưu ý về URL format:**
- ✅ **ĐÚNG**: `YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completions` (FULL URL)
- ❌ **SAI**: `YESCALE_API_BASE_URL=https://api.yescale.io/v1` (thiếu `/chat/completions`)

**Lý do:**
- `ChatGPT` class sử dụng URL trực tiếp (không append gì)
- `LangChain` models tự động strip `/chat/completions` trước khi pass vào SDK

## Cách hoạt động

Khi bạn cấu hình `YESCALE_API_BASE_URL`, hệ thống sẽ tự động:

1. **Ưu tiên sử dụng YEScale**: Nếu `YESCALE_API_BASE_URL` được set, tất cả các request sẽ được gửi đến YEScale thay vì OpenAI
2. **Tương thích với OpenAI API**: YEScale API tương thích 100% với OpenAI API format
3. **Fallback linh hoạt**: Nếu không có `YESCALE_API_BASE_URL`, hệ thống sẽ sử dụng OpenAI API như bình thường

### Thứ tự ưu tiên API Key

Hệ thống sẽ tìm kiếm API key theo thứ tự:
1. `YESCALE_API_KEY` (nếu có)
2. `OPENAI_API_KEY` (nếu không có YEScale key)

## Giải thích kỹ thuật

### Refactor: Từ OpenAI SDK sang requests library

**Vấn đề trước đó:**
- OpenAI SDK tự động append `/chat/completions` vào base_url
- User set `YESCALE_API_BASE_URL=https://api.yescale.io/v1/chat/completions` (full URL)
- Kết quả: `https://api.yescale.io/v1/chat/completions/chat/completions` → 404 Error!

**Giải pháp mới:**

**1. YEScale API = OpenAI API (cùng format)**

Từ tài liệu YEScale:
```python
# YEScale Request
POST /v1/chat/completions
Headers: Authorization: Bearer {{API_KEY}}
Body: {"model": "gpt-4o", "messages": [...]}

# YEScale Response
{"id": "...", "choices": [{"message": {"content": "..."}}], "usage": {...}}
```

Đây **chính xác là format của OpenAI API**! Không có gì khác biệt.

**2. ChatGPT class: Sử dụng requests library**

Code mới trong `gfmrag/llms/chatgpt.py`:
```python
import requests
import json

# Init với full URL
self.api_url = os.environ.get("YESCALE_API_BASE_URL")  # Full URL
self.api_key = os.environ.get("YESCALE_API_KEY")

# Gọi API bằng requests
headers = {
    'Authorization': f'Bearer {self.api_key}',
    'Content-Type': 'application/json'
}

payload = {
    "model": self.model_name,
    "messages": message,
    "temperature": 0.0
}

response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
data = response.json()
result = data['choices'][0]['message']['content']
```

**3. LangChain models: Strip /chat/completions tự động**

Code trong `gfmrag/kg_construction/langchain_util.py`:
```python
base_url = os.environ.get("YESCALE_API_BASE_URL")

# Strip /chat/completions vì OpenAI SDK sẽ append lại
if base_url and base_url.endswith("/chat/completions"):
    base_url = base_url[:-len("/chat/completions")]

# Bây giờ pass vào ChatOpenAI
client = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name)
```

**Kết quả:**
- ✅ `ChatGPT` class gọi trực tiếp full URL
- ✅ `LangChain` strip `/chat/completions` trước, để SDK append lại
- ✅ Cả 2 đều hoạt động hoàn hảo với YEScale!

### So sánh implementation

**Cách 1: Requests library (ChatGPT class - MỚI)**
```python
import requests
import json

url = "https://api.yescale.io/v1/chat/completions"  # Full URL
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Hello!"}]}

response = requests.post(url, headers=headers, data=json.dumps(payload))
data = response.json()
print(data['choices'][0]['message']['content'])
```

**Cách 2: OpenAI SDK với strip (LangChain - MỚI)**
```python
from langchain_openai import ChatOpenAI

# Strip /chat/completions từ full URL
base_url = "https://api.yescale.io/v1"  # Stripped

client = ChatOpenAI(api_key="your-key", base_url=base_url, model="gpt-4o-mini")
response = client.invoke("Hello!")
print(response.content)
```

**Kết quả: Cả 2 đều hoạt động!** Không còn lỗi 404.

## Sử dụng

### 1. Với module `gfmrag.llms.ChatGPT`

```python
from gfmrag.llms import ChatGPT

# Tự động sử dụng YEScale nếu đã cấu hình trong .env
llm = ChatGPT(model_name_or_path="gpt-4o-mini")

# Hoặc chỉ định trực tiếp trong code (sử dụng full URL)
llm = ChatGPT(
    model_name_or_path="gpt-4o-mini",
    api_key="your-yescale-key",
    api_url="https://api.yescale.io/v1/chat/completions"  # Full URL!
)

# Generate text
response = llm.generate_sentence("Hello, how are you?")
print(response)
```

### 2. Với LangChain models (NER, OpenIE)

Các module sử dụng LangChain (`LLMNERModel`, `LLMOPENIEModel`) sẽ tự động đọc từ biến môi trường:

```python
from gfmrag.kg_construction.ner_model import LLMNERModel

# Tự động sử dụng YEScale nếu YESCALE_API_BASE_URL được set
ner_model = LLMNERModel(
    llm_api="openai",  # Vẫn dùng "openai" vì YEScale tương thích OpenAI
    model_name="gpt-4o-mini",
    max_tokens=300
)

entities = ner_model("Bill Gates founded Microsoft")
print(entities)  # ['bill gates', 'microsoft']
```

### 3. Với YAML config files

Các file config trong `gfmrag/workflow/config/` không cần thay đổi. Chỉ cần set biến môi trường:

```yaml
# gfmrag/workflow/config/ner_model/llm_ner_model.yaml
_target_: gfmrag.kg_construction.ner_model.LLMNERModel
llm_api: openai  # Giữ nguyên "openai"
model_name: gpt-4o-mini
max_tokens: 300
```

Hệ thống sẽ tự động redirect đến YEScale nếu biến môi trường được cấu hình.

## Test kết nối

Để kiểm tra xem YEScale có hoạt động không:

```python
import os
from openai import OpenAI

# Kiểm tra biến môi trường
print(f"Base URL: {os.environ.get('YESCALE_API_BASE_URL')}")
print(f"API Key: {os.environ.get('YESCALE_API_KEY')[:10]}...")

# Test kết nối
client = OpenAI(
    api_key=os.environ.get('YESCALE_API_KEY'),
    base_url=os.environ.get('YESCALE_API_BASE_URL')
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

## YEScale API Specification

Theo tài liệu bạn cung cấp, YEScale API endpoint:

```
POST /v1/chat/completions
```

### Request Parameters

- `model` (required): Model ID (ví dụ: "gpt-4o", "gpt-4o-mini")
- `messages` (required): List of messages
  ```json
  [
    {"role": "user", "content": "Hello!"}
  ]
  ```
- `temperature` (optional): 0-2, default depends on model
- `max_tokens` (optional): Maximum tokens to generate
- `top_p` (optional): Nucleus sampling
- `stream` (optional): Enable streaming
- `stop` (optional): Stop sequences
- `presence_penalty` (optional): -2.0 to 2.0
- `frequency_penalty` (optional): -2.0 to 2.0

### Response Format

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Response text here"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

## Troubleshooting

### Lỗi: Connection Error

Kiểm tra:
- `YESCALE_API_BASE_URL` có đúng format không (phải bao gồm `/v1`)
- Network có thể kết nối đến YEScale endpoint không

### Lỗi: Authentication Failed

Kiểm tra:
- `YESCALE_API_KEY` có đúng không
- API key còn hạn sử dụng không

### Lỗi: Model not found

Kiểm tra:
- Model name có được YEScale hỗ trợ không (ví dụ: "gpt-4o-mini", "gpt-4o")
- Xem danh sách models được hỗ trợ trong tài liệu YEScale

## Chuyển đổi giữa OpenAI và YEScale

### Sử dụng YEScale

```bash
# .env
YESCALE_API_BASE_URL=https://api.yescale.com/v1
YESCALE_API_KEY=your-key
```

### Chuyển về OpenAI

Chỉ cần comment hoặc xóa `YESCALE_API_BASE_URL`:

```bash
# .env
# YESCALE_API_BASE_URL=https://api.yescale.com/v1  # Comment out
# YESCALE_API_KEY=your-key
OPENAI_API_KEY=sk-proj-xxx
```

Hệ thống sẽ tự động fallback về OpenAI API.

## Tài liệu tham khảo

- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [YEScale API Documentation](https://yescale.com/docs) (nếu có)
