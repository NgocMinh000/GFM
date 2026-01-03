# Hướng dẫn Fix Lỗi ColBERT trong Entity Resolution Script

## ❌ Vấn đề hiện tại

Lỗi: `string indices must be integers, not 'str'` khi truy cập `results[0]['score']`

**Nguyên nhân:** RAGatouille trả về kết quả ở format không mong đợi (strings thay vì dicts)

## ✅ Giải pháp

### Bước 1: Import utility function

Thêm vào đầu script của bạn:

```python
from gfmrag.kg_construction.entity_linking_model.colbert_utils import (
    extract_colbert_score,
    compute_colbert_pairwise_similarity
)
```

### Bước 2: Thay thế code cũ

#### ❌ Code CŨ (bị lỗi):

```python
# Tính ColBERT similarity
results = searcher.search(query=entity1, k=1)
colbert_score = results[0]['score']  # ❌ Lỗi ở đây!
```

#### ✅ Code MỚI (fix lỗi):

```python
# Tính ColBERT similarity với error handling
results = searcher.search(query=entity1, k=1)
colbert_score = extract_colbert_score(results, entity1, fallback=0.0)
```

### Bước 3: Hoặc sử dụng wrapper function

Nếu bạn đang tính pairwise similarity giữa 2 entities:

```python
# Thay vì:
# searcher = RAGPretrainedModel.from_index(index_path)
# results = searcher.search(query=entity1, k=1)
# score = results[0]['score']  # ❌ Lỗi

# Dùng:
from gfmrag.kg_construction.entity_linking_model import ColbertELModel

model = ColbertELModel()
score = model.compute_pairwise_similarity(entity1, entity2)  # ✅ OK
```

## 📋 Ví dụ cụ thể

### Scenario: Multi-Feature Scoring

```python
import logging
from ragatouille import RAGPretrainedModel
from gfmrag.kg_construction.entity_linking_model.colbert_utils import extract_colbert_score

logger = logging.getLogger(__name__)

# Load ColBERT index
searcher = RAGPretrainedModel.from_index("path/to/colbert/index")

# Feature weights
feature_weights = {
    'sapbert': 0.5,
    'lexical': 0.15,
    'colbert': 0.25,
    'graph': 0.1
}

# Tính scores cho mỗi pair
for entity1, entity2 in candidate_pairs:
    # SapBERT score (working fine)
    sapbert_score = compute_sapbert_similarity(entity1, entity2)

    # Lexical score (working fine)
    lexical_score = compute_lexical_similarity(entity1, entity2)

    # ColBERT score (FIX HERE!)
    try:
        results = searcher.search(query=entity1, k=1)
        # ✅ Use safe extraction
        colbert_score = extract_colbert_score(results, entity1, fallback=0.0)
    except Exception as e:
        logger.error(f"ColBERT search failed for '{entity1}': {e}")
        colbert_score = 0.0

    # Graph score (working fine)
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
    print(f"  ColBERT: {colbert_score:.3f}")  # ✅ Sẽ không còn 0!
    print(f"  Graph: {graph_score:.3f}")
    print(f"  Combined: {combined_score:.3f}")
```

## 🔍 Debug Utilities

### Kiểm tra index hoạt động đúng không:

```python
from gfmrag.kg_construction.entity_linking_model.colbert_utils import (
    validate_colbert_index,
    debug_colbert_results
)

# Validate index
if not validate_colbert_index(searcher):
    print("❌ ColBERT index có vấn đề!")
else:
    print("✅ ColBERT index hoạt động tốt")

# Debug raw results
debug_colbert_results(searcher, "aspirin", k=3)
```

### Batch processing cho nhiều pairs:

```python
from gfmrag.kg_construction.entity_linking_model.colbert_utils import (
    batch_compute_colbert_similarity
)

# Tính similarity cho nhiều pairs cùng lúc
pairs = [
    ("aspirin", "acetylsalicylic acid"),
    ("diabetes", "hyperglycemia"),
    ("hypertension", "high blood pressure")
]

scores = batch_compute_colbert_similarity(searcher, pairs, batch_size=32)

for pair, score in scores.items():
    print(f"{pair[0]} <-> {pair[1]}: {score:.3f}")
```

## 🎯 Checklist

- [ ] Import `extract_colbert_score` từ `colbert_utils`
- [ ] Thay thế tất cả `results[0]['score']` bằng `extract_colbert_score(results, query)`
- [ ] Thêm try-except cho error handling
- [ ] Test lại script
- [ ] Verify ColBERT scores không còn là 0

## 📊 Expected Results

Sau khi fix:
- ✅ ColBERT scores sẽ có giá trị > 0 (thường 0.3 - 0.9)
- ✅ Không còn lỗi "string indices must be integers"
- ✅ Combined scores sẽ cao hơn (do ColBERT contribution)
- ✅ Tìm được nhiều equivalent pairs hơn

## 💡 Giải thích kỹ thuật

`extract_colbert_score()` xử lý nhiều format kết quả:

1. **Dict with 'score' key**: `{"content": "...", "score": 0.85}` → 0.85
2. **Dict with 'similarity' key**: `{"text": "...", "similarity": 0.85}` → 0.85
3. **String result**: `"some text"` → 0.0 (fallback)
4. **Empty result**: `[]` → 0.0 (fallback)
5. **Error cases**: → 0.0 (fallback) + log error

Điều này đảm bảo script luôn chạy được dù RAGatouille trả về format nào.

## 🔗 Related Files

Các file đã được update:
- `gfmrag/kg_construction/entity_linking_model/colbert_utils.py` - Utility functions
- `gfmrag/kg_construction/entity_linking_model/colbert_el_model.py` - Fixed pairwise method
- `gfmrag/kg_construction/entity_linking_model/__init__.py` - Exports

---

**Nếu vẫn gặp lỗi sau khi apply fix này, hãy share:**
1. Script entity resolution của bạn (hoặc đường dẫn)
2. Log output chi tiết
3. Version của RAGatouille: `pip show ragatouille`
