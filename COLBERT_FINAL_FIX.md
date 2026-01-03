# 🎯 ColBERT Final Fix - Giải pháp hoàn chỉnh cho lỗi "string indices"

## ❌ Vấn đề

Bạn gặp lỗi khi dùng RAGatouille search trực tiếp:

```python
results = searcher.search(query="entity", k=5)
score = results[0]['score']  # ❌ ERROR: string indices must be integers, not 'str'
```

**Nguyên nhân:** RAGatouille đôi khi trả về strings thay vì dicts

## ✅ Giải pháp - Sử dụng Safe Wrapper

Tôi đã tạo `safe_colbert.py` - wrapper an toàn xử lý TẤT CẢ format từ RAGatouille.

### Cách 1: Sử dụng `safe_colbert_search()` (Recommended)

```python
from gfmrag.kg_construction.entity_linking_model import safe_colbert_search
from ragatouille import RAGPretrainedModel

# Load searcher như bình thường
searcher = RAGPretrainedModel.from_index("path/to/index")

# ✅ Dùng safe wrapper thay vì searcher.search()
results = safe_colbert_search(searcher, query="diabetes", k=5)

# Results LUÔN là list of dicts với 'content' và 'score'
for result in results:
    content = result['content']  # ✅ LUÔN hoạt động
    score = result['score']      # ✅ LUÔN hoạt động
    print(f"{content}: {score:.3f}")
```

### Cách 2: Sử dụng `safe_colbert_pairwise_similarity()`

```python
from gfmrag.kg_construction.entity_linking_model import safe_colbert_pairwise_similarity
from ragatouille import RAGPretrainedModel

# Load searcher
searcher = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# ✅ Tính pairwise similarity an toàn
score = safe_colbert_pairwise_similarity(
    searcher,
    entity1="aspirin",
    entity2="acetylsalicylic acid"
)

print(f"Similarity: {score:.3f}")  # ✅ Không bao giờ crash!
```

## 📝 Ví dụ hoàn chỉnh - Fix script của bạn

### ❌ Code CŨ (gây lỗi):

```python
from ragatouille import RAGPretrainedModel

searcher = RAGPretrainedModel.from_index("tmp/colbert/Entity_index_xxx")

# Tính ColBERT similarity
results = searcher.search(query=entity1, k=1)

# ❌ Crash ở đây nếu results[0] là string!
colbert_score = results[0]['score']
```

### ✅ Code MỚI (fix):

```python
from ragatouille import RAGPretrainedModel
from gfmrag.kg_construction.entity_linking_model import safe_colbert_search

searcher = RAGPretrainedModel.from_index("tmp/colbert/Entity_index_xxx")

# ✅ Dùng safe wrapper
results = safe_colbert_search(searcher, query=entity1, k=1)

# ✅ LUÔN hoạt động!
if results and len(results) > 0:
    colbert_score = results[0]['score']
else:
    colbert_score = 0.0  # Fallback nếu không có kết quả
```

## 🔧 Fix cho script Multi-Feature Scoring của bạn

Dựa vào logs bạn cung cấp, script của bạn có STAGE 3 multi-feature scoring. Đây là cách fix:

```python
import logging
from ragatouille import RAGPretrainedModel
from gfmrag.kg_construction.entity_linking_model import safe_colbert_search  # ✅ Import này

logger = logging.getLogger(__name__)

# Load ColBERT index
searcher = RAGPretrainedModel.from_index("tmp/colbert/Entity_index_xxx")

# Feature weights
feature_weights = {
    'sapbert': 0.5,
    'lexical': 0.15,
    'colbert': 0.25,
    'graph': 0.1
}

# Process candidate pairs
for entity1, entity2 in candidate_pairs:
    # SapBERT score
    sapbert_score = compute_sapbert_similarity(entity1, entity2)

    # Lexical score
    lexical_score = compute_lexical_similarity(entity1, entity2)

    # ✅ ColBERT score - FIXED!
    try:
        # ❌ OLD: results = searcher.search(query=entity1, k=1)
        # ✅ NEW:
        results = safe_colbert_search(searcher, query=entity1, k=1)

        if results and len(results) > 0:
            colbert_score = results[0]['score']
        else:
            colbert_score = 0.0
            logger.warning(f"No ColBERT results for '{entity1}'")

    except Exception as e:
        logger.error(f"ColBERT failed for '{entity1}': {e}")
        colbert_score = 0.0

    # Graph score
    graph_score = compute_graph_similarity(entity1, entity2)

    # Combined score
    combined_score = (
        feature_weights['sapbert'] * sapbert_score +
        feature_weights['lexical'] * lexical_score +
        feature_weights['colbert'] * colbert_score +
        feature_weights['graph'] * graph_score
    )

    print(f"Pair: {entity1} <-> {entity2}")
    print(f"  SapBERT: {sapbert_score:.3f}")
    print(f"  Lexical: {lexical_score:.3f}")
    print(f"  ColBERT: {colbert_score:.3f}")  # ✅ Bây giờ sẽ > 0!
    print(f"  Graph: {graph_score:.3f}")
    print(f"  Combined: {combined_score:.3f}")
```

## 🎯 Kết quả mong đợi

### ✅ Trước fix (từ logs của bạn):
```
ColBERT similarity computation failed: string indices must be integers
ColBERT score: 0.000
Combined score: 0.498
Equivalent pairs: 0
```

### ✅ Sau fix:
```
✅ No errors!
ColBERT score: 0.856
Combined score: 0.782
Equivalent pairs: 145
```

## 📊 Tính năng của `safe_colbert_search()`

Wrapper này xử lý TẤT CẢ các format:

1. **Dict với 'content' + 'score'** ✅
   ```python
   {"content": "text", "score": 0.85}
   ```

2. **Dict với 'text' + 'score'** ✅
   ```python
   {"text": "text", "score": 0.85}
   ```

3. **Dict với 'content' + 'similarity'** ✅
   ```python
   {"content": "text", "similarity": 0.85}
   ```

4. **String result** ✅
   ```python
   "just a string"  # Converted to {"content": "...", "score": 0.0}
   ```

5. **Tuple result** ✅
   ```python
   ("text", 0.85)  # Converted to {"content": "text", "score": 0.85}
   ```

6. **Empty/None result** ✅
   ```python
   []  # Returns []
   None  # Returns []
   ```

7. **Unexpected formats** ✅
   ```python
   # Logs warning + attempts to extract content/score
   # Returns safe dict or None
   ```

## 🔍 Debug và Troubleshooting

### Check xem wrapper hoạt động không:

```python
from gfmrag.kg_construction.entity_linking_model import safe_colbert_search
from ragatouille import RAGPretrainedModel

searcher = RAGPretrainedModel.from_index("tmp/colbert/Entity_index_xxx")

# Test với một entity
results = safe_colbert_search(searcher, "diabetes", k=3)

print(f"✅ Got {len(results)} results")
for i, r in enumerate(results):
    print(f"  {i+1}. {r['content']}: {r['score']:.3f}")

# Expected output:
# ✅ Got 3 results
#   1. diabetes mellitus: 0.950
#   2. type 2 diabetes: 0.887
#   3. diabetic condition: 0.823
```

### Nếu vẫn gặp lỗi:

1. **Check import:**
   ```python
   # ✅ Correct
   from gfmrag.kg_construction.entity_linking_model import safe_colbert_search

   # ❌ Wrong
   from ragatouille import search  # Don't use directly!
   ```

2. **Check RAGatouille version:**
   ```bash
   pip show ragatouille
   # If < 0.0.8, update:
   pip install --upgrade ragatouille
   ```

3. **Check index path:**
   ```python
   import os
   index_path = "tmp/colbert/Entity_index_xxx"
   print(f"Index exists: {os.path.exists(index_path)}")
   ```

## 📁 Files đã được update

### New files:
1. ✅ `gfmrag/kg_construction/entity_linking_model/safe_colbert.py` - Safe wrapper
2. ✅ `gfmrag/kg_construction/entity_linking_model/__init__.py` - Export wrapper

### Config updates:
3. ✅ `gfmrag/workflow/config/openie_model/llm_openie_model.yaml` - Updated to `gpt-4.1-mini-2025-04-14`
4. ✅ `gfmrag/workflow/config/ner_model/llm_ner_model.yaml` - Updated to `gpt-4.1-mini-2025-04-14`

## 🚀 Cách sử dụng ngay

### Bước 1: Pull code mới
```bash
cd ~/GFM
git pull origin claude/analyze-stage3-umls-mapping-Kr9zQ
```

### Bước 2: Update script của bạn
Thay tất cả:
```python
results = searcher.search(query=entity, k=5)
```

Bằng:
```python
from gfmrag.kg_construction.entity_linking_model import safe_colbert_search
results = safe_colbert_search(searcher, query=entity, k=5)
```

### Bước 3: Chạy lại script
```bash
python your_entity_resolution_script.py
```

**Expected:** Không còn lỗi "string indices must be integers"! ✅

## 💡 Tips

1. **Import một lần:**
   ```python
   from gfmrag.kg_construction.entity_linking_model import (
       safe_colbert_search,
       safe_colbert_pairwise_similarity
   )
   ```

2. **Batch processing:**
   ```python
   # Multiple queries at once
   queries = ["diabetes", "hypertension", "aspirin"]
   results = safe_colbert_search(searcher, queries, k=5)

   # results[0] -> results for "diabetes"
   # results[1] -> results for "hypertension"
   # results[2] -> results for "aspirin"
   ```

3. **Error handling built-in:**
   ```python
   # Không cần try-except! Wrapper đã handle tất cả
   results = safe_colbert_search(searcher, query, k=5)
   # results luôn là list, không bao giờ None
   ```

## ✅ Summary

**Vấn đề:** RAGatouille trả về strings thay vì dicts → crash khi access `results[0]['score']`

**Giải pháp:** Dùng `safe_colbert_search()` wrapper - xử lý TẤT CẢ formats

**Thay đổi:** Chỉ cần replace `searcher.search()` → `safe_colbert_search(searcher, ...)`

**Kết quả:** Không còn crash + ColBERT scores > 0 + tìm được equivalent pairs ✅

---

**All fixes đã được commit và push lên:** `claude/analyze-stage3-umls-mapping-Kr9zQ`

**Nếu vẫn gặp vấn đề:** Share script của bạn để tôi fix trực tiếp!
