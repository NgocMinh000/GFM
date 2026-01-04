# LLM Initialization Pattern - Best Practice

## 📚 Overview

Hướng dẫn này mô tả **best practice** để khởi tạo LLM trong GFM codebase, dựa trên pattern đã được verify trong `llm_openie_model.py`.

---

## ✅ Recommended Pattern (llm_openie_model.py)

### **Step 1: Initialize LLM Client**

```python
from gfmrag.kg_construction.langchain_util import init_langchain_model

# In __init__ or initialization method:
self.client = init_langchain_model(
    llm="openai",  # Will auto-detect YEScale or use OpenAI
    model_name="gpt-4o-mini",
    temperature=0.0,
)
```

**How init_langchain_model() works:**
1. Checks `YESCALE_API_BASE_URL` environment variable
2. If set → Uses `YEScaleChatModel` (custom implementation)
3. If not set → Uses `ChatOpenAI` (official OpenAI SDK)
4. Automatically handles API key from `YESCALE_API_KEY` or `OPENAI_API_KEY`

### **Step 2: Invoke with Instance-Based Handling**

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from gfmrag.kg_construction.utils import extract_json_dict

# Prepare messages
messages = [HumanMessage(content=prompt)]

# Instance-based invocation
if isinstance(self.client, ChatOpenAI):
    # OpenAI: Use JSON mode (reliable, enforced JSON output)
    response = self.client.invoke(
        messages,
        temperature=0,
        max_tokens=1024,
        response_format={"type": "json_object"},  # ← Key feature
    )
    response_content = response.content
    result = eval(response_content)  # Safe since JSON mode guarantees valid JSON

else:
    # YEScale or other models: Parse JSON manually
    response = self.client.invoke(
        messages,
        temperature=0,
        max_tokens=1024,
    )
    response_content = response.content
    result = extract_json_dict(response_content)  # Robust JSON extraction

# Use result
entity_type = result.get("type", "other")
confidence = result.get("confidence", 0.5)
```

---

## 🎯 Why This Pattern?

### **1. Auto-Detection**
- No manual env var checks needed
- `init_langchain_model()` handles all logic
- Works with both YEScale and OpenAI automatically

### **2. JSON Mode for OpenAI**
- OpenAI's `response_format={"type": "json_object"}` **guarantees** valid JSON
- No need for manual parsing, regex extraction, or error handling
- Model output will ALWAYS be valid JSON

### **3. Robust JSON Parsing for YEScale**
- YEScale doesn't support JSON mode
- `extract_json_dict()` handles:
  - Extra text before/after JSON
  - Malformed JSON (attempts to fix)
  - Missing quotes, trailing commas, etc.
- Same utility used throughout codebase

### **4. Consistent Pattern**
- Same approach used in:
  - `llm_openie_model.py` (Stage 1 OpenIE)
  - `stage2_entity_resolution.py` (Stage 0 type inference)
- Easy to maintain and debug

---

## 📋 Complete Example

### **Example 1: Stage 1 OpenIE (llm_openie_model.py)**

```python
class LLMOPENIEModel(BaseOPENIEModel):
    def __init__(self, llm_api: str = "openai", model_name: str = "gpt-4o-mini"):
        # Step 1: Initialize
        self.client = init_langchain_model(llm_api, model_name)

    def ner(self, text: str) -> list:
        # Step 2: Invoke with instance check
        messages = ner_prompts.format_prompt(user_input=text).to_messages()

        if isinstance(self.client, ChatOpenAI):
            # OpenAI: JSON mode
            response = self.client.invoke(
                messages,
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            result = eval(response.content)
        else:
            # YEScale or others: Parse JSON
            response = self.client.invoke(messages, temperature=0)
            result = extract_json_dict(response.content)

        return result.get("named_entities", [])
```

### **Example 2: Stage 0 Type Inference (stage2_entity_resolution.py)**

```python
def _infer_type_relationship_llm(self, entity: str) -> Dict:
    # Step 1: Initialize (cached)
    if not hasattr(self, '_llm_cache'):
        # Check if API key available
        api_key = os.environ.get("YESCALE_API_KEY") or os.environ.get("OPENAI_API_KEY")

        if not api_key:
            logger.warning("LLM API not configured. Skipping.")
            self._llm_cache = None
        else:
            # Use init_langchain_model
            self._llm_cache = init_langchain_model(
                llm="openai",
                model_name="gpt-4o-mini",
                temperature=0.0,
            )

    if self._llm_cache is None:
        return {"type": "other", "confidence": 0.3}

    # Step 2: Invoke with instance check
    llm = self._llm_cache
    messages = [HumanMessage(content=prompt)]

    if isinstance(llm, ChatOpenAI):
        # OpenAI: JSON mode
        response = llm.invoke(
            messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = eval(response.content)
    else:
        # YEScale: Parse JSON
        response = llm.invoke(messages, temperature=0)
        result = extract_json_dict(response.content)

    return {
        "type": result.get("type", "other"),
        "confidence": result.get("confidence", 0.5)
    }
```

---

## ❌ Anti-Patterns (What NOT to Do)

### **Anti-Pattern 1: Direct YEScaleChatModel Initialization**

```python
# ❌ BAD: Manual initialization
from gfmrag.kg_construction.yescale_chat_model import YEScaleChatModel

yescale_url = os.environ.get("YESCALE_API_BASE_URL")
yescale_key = os.environ.get("YESCALE_API_KEY")

if not yescale_url or not yescale_key:
    raise ValueError("Missing credentials!")

self.llm = YEScaleChatModel(
    api_url=yescale_url,
    api_key=yescale_key,
    model="gpt-4o-mini"
)
```

**Problems:**
- Manual env var checking (error-prone)
- No fallback to OpenAI
- Duplicate code across files
- Inconsistent with rest of codebase

**Better:**
```python
# ✅ GOOD: Use init_langchain_model
self.llm = init_langchain_model("openai", "gpt-4o-mini")
```

### **Anti-Pattern 2: Manual JSON Parsing Without extract_json_dict**

```python
# ❌ BAD: Manual string manipulation
response_text = response.content.strip()
if "{" in response_text and "}" in response_text:
    json_start = response_text.index("{")
    json_end = response_text.rindex("}") + 1
    json_str = response_text[json_start:json_end]
    result = json.loads(json_str)
```

**Problems:**
- Doesn't handle malformed JSON
- Fails on nested braces
- No error recovery

**Better:**
```python
# ✅ GOOD: Use extract_json_dict
from gfmrag.kg_construction.utils import extract_json_dict

result = extract_json_dict(response.content)
```

### **Anti-Pattern 3: No Instance Checking**

```python
# ❌ BAD: Assume all models support JSON mode
response = llm.invoke(
    messages,
    response_format={"type": "json_object"}  # YEScale doesn't support this!
)
```

**Problems:**
- YEScale doesn't support `response_format`
- Will crash or be ignored

**Better:**
```python
# ✅ GOOD: Instance-based handling
if isinstance(llm, ChatOpenAI):
    # Use JSON mode
    response = llm.invoke(messages, response_format={"type": "json_object"})
else:
    # Parse manually
    response = llm.invoke(messages)
```

---

## 🔧 Utilities

### **extract_json_dict() - Robust JSON Extraction**

Location: `gfmrag/kg_construction/utils.py`

**What it does:**
- Extracts JSON from text with extra content
- Handles malformed JSON (attempts to fix)
- Returns empty dict `{}` if parsing fails

**Example:**
```python
from gfmrag.kg_construction.utils import extract_json_dict

# Input: "Here is the result: {"type": "drug", "confidence": 0.85} Hope this helps!"
result = extract_json_dict(text)
# Output: {"type": "drug", "confidence": 0.85}

# Input: "No JSON here"
result = extract_json_dict(text)
# Output: {}
```

---

## 📊 Comparison: OpenAI vs YEScale

| Feature | OpenAI (ChatOpenAI) | YEScale (YEScaleChatModel) |
|---------|---------------------|----------------------------|
| **JSON Mode** | ✅ Supported (`response_format`) | ❌ Not supported |
| **API Endpoint** | `https://api.openai.com/v1/chat/completions` | `https://api.yescale.io/v1/chat/completions` |
| **Initialization** | Via `init_langchain_model()` | Via `init_langchain_model()` |
| **JSON Parsing** | `eval(response.content)` | `extract_json_dict(response.content)` |
| **Env Vars** | `OPENAI_API_KEY` | `YESCALE_API_BASE_URL` + `YESCALE_API_KEY` |

---

## 🎯 Checklist for New LLM Usage

When adding new LLM calls to the codebase:

- [ ] Use `init_langchain_model()` for initialization
- [ ] Cache LLM instance (don't re-initialize per call)
- [ ] Check instance type: `if isinstance(llm, ChatOpenAI)`
- [ ] Use JSON mode for OpenAI: `response_format={"type": "json_object"}`
- [ ] Use `extract_json_dict()` for non-OpenAI models
- [ ] Handle gracefully if API key missing
- [ ] Add clear logging about which API is used
- [ ] Test with both YEScale and OpenAI (if applicable)

---

## 📚 Related Files

- **`gfmrag/kg_construction/langchain_util.py`**: `init_langchain_model()` implementation
- **`gfmrag/kg_construction/yescale_chat_model.py`**: `YEScaleChatModel` class
- **`gfmrag/kg_construction/utils.py`**: `extract_json_dict()` utility
- **`gfmrag/kg_construction/openie_model/llm_openie_model.py`**: Reference implementation
- **`gfmrag/workflow/stage2_entity_resolution.py`**: Stage 0 LLM usage

---

## 🚀 Summary

**Best Practice:**
1. ✅ Use `init_langchain_model("openai", "gpt-4o-mini")`
2. ✅ Check instance: `isinstance(llm, ChatOpenAI)`
3. ✅ OpenAI → JSON mode, YEScale → `extract_json_dict()`
4. ✅ Cache LLM instance
5. ✅ Graceful degradation if no API key

**Follow this pattern for:**
- Consistency across codebase
- Auto-detection of YEScale vs OpenAI
- Robust JSON parsing
- Better error handling
- Easier maintenance

**Reference:** `llm_openie_model.py` for complete working example.
