# STAGE 3 PHASE 2: CROSS-ENCODER FINE-TUNING & ADVANCED OPTIMIZATIONS

## 📋 OVERVIEW

**Date**: 2026-01-16
**Status**: 🚧 IN PROGRESS
**Priority**: 🔴 CRITICAL (Biggest impact on quality)
**Expected Impact**: High confidence 5-10% → 40-60% (+30-50% improvement)

---

## ⚠️ CURRENT BOTTLENECK

### Problem: Cross-Encoder Zero-Shot Performance
```json
{
  "cross_encoder_model": "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR",
  "performance": {
    "avg_score": 0.58,
    "target_score": 0.80,
    "gap": "-27.5%"
  },
  "root_cause": "Zero-shot inference (not trained on UMLS entity linking)",
  "impact": "Poor final reranking → Low confidence mappings"
}
```

**Why This Matters**:
- Cross-encoder is the **final decision maker** (Step 3.5)
- Current model was pre-trained on general biomedical text, **NOT** entity linking
- Without fine-tuning: Cannot distinguish between similar candidates effectively

---

## 🎯 PHASE 2 COMPONENTS

### 2.1. Cross-Encoder Fine-Tuning (CRITICAL)
**Impact**: +30-50% high confidence
**Timeline**: 1.5 days implementation + 4-6 hours training

**Approach**:
```python
# Training objective: Binary classification
Input: (entity_mention, candidate_cui_name)
Output: [0 or 1] (is_correct_match)

Example positive:
  entity: "aspirin"
  candidate: "Aspirin [CUI: C0004057]"
  label: 1

Example hard negative:
  entity: "aspirin"
  candidate: "Aspirin Allergy [CUI: C0004058]"  # Similar but wrong
  label: 0
```

**Training Data Sources**:
1. **MedMentions** (Primary)
   - 4,000 PubMed abstracts
   - 352,496 entity mentions
   - Linked to UMLS 2017AA
   - High quality gold labels

2. **BC5CDR** (Secondary)
   - Chemical-Disease Relations
   - 1,500 PubMed abstracts
   - 15,935 chemical mentions
   - 12,852 disease mentions

**Hard Negative Mining Strategy**:
```python
# For each positive (entity, correct_cui):
# Generate 3 types of negatives:

1. Semantic negatives (high similarity, wrong CUI):
   - Use FAISS to find top-20 similar CUIs
   - Filter out correct CUI
   - Select top-5 as hard negatives

2. Type negatives (wrong semantic type):
   - If entity is "drug", select disease/symptom CUIs
   - Ensures type discrimination

3. Random negatives (easy negatives for stability):
   - Sample 2-3 random CUIs
```

**Model Architecture**:
```
Base Model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
Input: [CLS] entity_mention [SEP] cui_name [SEP]
Output: Binary classification head (sigmoid)
Loss: Binary Cross-Entropy + Hard Negative Contrastive Loss
```

**Training Configuration**:
```yaml
model:
  base: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  max_length: 128

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 3
  warmup_ratio: 0.1
  gradient_accumulation_steps: 2

data:
  positive_samples: 352496  # MedMentions
  hard_negatives_per_positive: 5
  easy_negatives_per_positive: 2
  total_samples: ~2.4M

hard_negative_mining:
  similarity_threshold: 0.85  # Only very similar CUIs
  max_candidates: 20
  top_k_hard: 5
```

**Expected Performance**:
| Metric | Before | After Fine-Tuning |
|--------|--------|-------------------|
| Avg Cross-Encoder Score | 0.58 | 0.85+ |
| High Confidence % | 5-10% | 40-60% |
| Score Margin | 0.10 | 0.25+ |
| Type Match Accuracy | 75% | 90%+ |

---

### 2.2. Confidence Recalibration
**Impact**: +5-10% calibration accuracy
**Timeline**: 0.5 days

**Problem**: Raw model scores != True probabilities
```python
# Example miscalibration:
model_score = 0.85
actual_accuracy = 0.68  # Overconfident!
```

**Solution: Platt Scaling**
```python
from sklearn.linear_model import LogisticRegression

# Step 1: Collect (score, label) pairs on validation set
val_scores = [0.85, 0.72, 0.91, ...]  # Model outputs
val_labels = [1, 0, 1, ...]           # True labels

# Step 2: Fit Platt scaler
platt_scaler = LogisticRegression()
platt_scaler.fit(val_scores.reshape(-1, 1), val_labels)

# Step 3: Apply calibration
calibrated_prob = platt_scaler.predict_proba(score)[0][1]

# Result:
model_score = 0.85 → calibrated_prob = 0.72 (matches actual accuracy!)
```

**Integration Point**: After cross-encoder scoring (Step 3.5)

---

### 2.3. Adaptive Threshold Tuning
**Impact**: +3-5% high confidence
**Timeline**: 0.5 days

**Problem**: Single confidence threshold (0.80) too strict/lenient for all types

**Solution: Per-Type Thresholds**
```python
# Learned from validation data:
confidence_thresholds = {
    "drug": 0.85,      # Strict (many similar drug names)
    "disease": 0.78,   # Moderate
    "gene": 0.90,      # Very strict (gene symbols ambiguous)
    "symptom": 0.70,   # Lenient (fewer confusions)
    "anatomy": 0.80,   # Moderate
    "procedure": 0.82, # Moderate-strict
    "other": 0.75      # Default
}

# Usage:
entity_type = get_entity_type(entity)
threshold = confidence_thresholds[entity_type]
tier = "high" if confidence >= threshold else ...
```

**Threshold Optimization**:
- Use validation set to find optimal threshold per type
- Maximize F1 score at each threshold
- Constraint: Coverage >= 80% for each type

---

## 📂 NEW FILES TO CREATE

### 1. Training Data Preparation
```
gfmrag/umls_mapping/training/
├── __init__.py
├── data_loader.py              # MedMentions/BC5CDR loader
├── hard_negative_miner.py      # Generate hard negatives
└── dataset.py                  # PyTorch Dataset class
```

### 2. Model Training
```
gfmrag/umls_mapping/training/
├── cross_encoder_trainer.py    # Fine-tuning script
├── config/
│   └── training_config.yaml    # Hyperparameters
└── models/
    └── binary_cross_encoder.py # Model wrapper
```

### 3. Calibration & Tuning
```
gfmrag/umls_mapping/
├── calibration.py              # Platt scaling
└── adaptive_thresholds.py      # Per-type thresholds
```

### 4. Integration
```
gfmrag/workflow/
└── config/stage3_umls_mapping.yaml  # Add fine-tuned model path
```

---

## 🚀 IMPLEMENTATION ROADMAP

### Day 1: Data Preparation & Hard Negative Mining
- ✅ Download MedMentions dataset
- ✅ Parse and convert to training format
- ✅ Implement hard negative mining with FAISS
- ✅ Create PyTorch Dataset class
- ✅ Verify data quality (sample inspection)

**Deliverable**: 2.4M training samples ready

### Day 2: Model Fine-Tuning
- ✅ Implement binary cross-encoder model
- ✅ Training script with hard negative loss
- ✅ Train for 3 epochs (~4-6 hours on GPU)
- ✅ Evaluate on validation set
- ✅ Save fine-tuned checkpoint

**Deliverable**: Fine-tuned cross-encoder model

### Day 3: Calibration & Integration
- ✅ Implement Platt scaling calibration
- ✅ Optimize per-type confidence thresholds
- ✅ Integrate into Stage 3 pipeline
- ✅ Run full pipeline test
- ✅ Measure quality improvements

**Deliverable**: Fully integrated Phase 2 system

---

## 📊 SUCCESS METRICS

### Quantitative Targets
| Metric | Baseline (Phase 1) | Phase 2 Target | Success Criteria |
|--------|-------------------|----------------|------------------|
| High Confidence % | 5-10% | 40-60% | ≥ 40% |
| Score Margin | 0.10-0.12 | 0.25+ | ≥ 0.20 |
| Overall Confidence | 0.64-0.66 | 0.75+ | ≥ 0.70 |
| Cross-Encoder Score | 0.58 | 0.85+ | ≥ 0.80 |
| Type Match Accuracy | 75% | 90%+ | ≥ 85% |

### Qualitative Checks
- [ ] Manually inspect top-100 high-confidence mappings → 95%+ correct
- [ ] Check hard cases (ambiguous entities) → Improved discrimination
- [ ] Semantic type consistency ≥ 95%
- [ ] Coverage at τ=0.70 ≥ 80%

---

## ⚙️ CONFIGURATION UPDATES

### Training Config
```yaml
# gfmrag/umls_mapping/training/config/training_config.yaml
model:
  name: "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
  checkpoint_dir: "models/cross_encoder_finetuned"

dataset:
  medmentions:
    path: "data/MedMentions/full"
    split: "train"
    use_full: true

  bc5cdr:
    path: "data/BC5CDR"
    use_chemicals: true
    use_diseases: true

hard_negatives:
  faiss_index: "tmp/umls_umls_faiss_index"
  similarity_threshold: 0.85
  top_k: 20
  negatives_per_positive: 5
  type_negatives: 2
  random_negatives: 2

training:
  batch_size: 32
  learning_rate: 2e-5
  epochs: 3
  max_length: 128
  warmup_ratio: 0.1
  weight_decay: 0.01
  gradient_accumulation_steps: 2
  fp16: true  # Mixed precision training

  loss:
    positive_weight: 1.0
    hard_negative_weight: 1.5  # Emphasize hard negatives
    easy_negative_weight: 0.5
```

### Pipeline Integration Config
```yaml
# gfmrag/workflow/config/stage3_umls_mapping.yaml (additions)
cross_encoder:
  model_path: "models/cross_encoder_finetuned"  # Fine-tuned model
  use_calibration: true
  calibration_path: "models/platt_scaler.pkl"
  batch_size: 32
  max_length: 128

confidence:
  use_adaptive_thresholds: true
  thresholds_path: "models/adaptive_thresholds.json"
  fallback_threshold: 0.75
```

---

## 🔬 VALIDATION STRATEGY

### Split Strategy
```python
# MedMentions splitting:
train_split = 80%  # 3,200 abstracts
val_split = 10%    # 400 abstracts
test_split = 10%   # 400 abstracts

# Stratified by:
- Entity type distribution
- CUI frequency (common vs rare)
- Difficulty (easy vs hard negatives)
```

### Evaluation Metrics
1. **Classification Metrics** (on test set):
   - Precision / Recall / F1
   - ROC-AUC
   - PR-AUC (more informative for imbalanced data)

2. **Ranking Metrics** (end-to-end):
   - MRR (Mean Reciprocal Rank)
   - Recall@K (K=1, 5, 10, 32)
   - NDCG (Normalized Discounted Cumulative Gain)

3. **Confidence Metrics**:
   - ECE (Expected Calibration Error)
   - Brier Score
   - Reliability diagram

---

## 📚 REFERENCES

### Datasets
- **MedMentions**: Mohan & Li (2019), "MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts"
  - [GitHub](https://github.com/chanzuckerberg/MedMentions)
  - [Paper](https://arxiv.org/abs/1902.09476)

- **BC5CDR**: Li et al. (2016), "BioCreative V CDR task corpus"
  - [BioCreative](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/)

### Methods
- **Hard Negative Mining**: Xiong et al. (2020), "Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval"
- **Platt Scaling**: Platt (1999), "Probabilistic Outputs for Support Vector Machines"
- **Cross-Encoder**: Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"

### Biomedical NLP
- **SapBERT**: Liu et al. (2021), "Self-Alignment Pretraining for Biomedical Entity Representations"
- **PubMedBERT**: Gu et al. (2021), "Domain-Specific Language Model Pretraining for Biomedical Natural Language Processing"

---

## 🐛 RISKS & MITIGATION

### Risk 1: Overfitting on MedMentions
**Mitigation**:
- Use BC5CDR as additional validation set
- Early stopping with patience=2
- Dropout=0.1 in classification head
- Monitor validation metrics closely

### Risk 2: UMLS Version Mismatch
**Problem**: MedMentions uses UMLS 2017AA, current system uses 2020AB
**Mitigation**:
- Map CUIs from 2017AA → 2020AB using MRCONSO.RRF
- Handle deprecated CUIs gracefully
- Use CUI normalization

### Risk 3: GPU Memory Limitations
**Mitigation**:
- Gradient accumulation (effective batch size = 32×2 = 64)
- Mixed precision training (fp16)
- Reduce batch size if OOM occurs
- Use gradient checkpointing if needed

### Risk 4: Training Time Too Long
**Estimated**: 4-6 hours on single V100 GPU
**Mitigation**:
- Use multi-GPU if available (DataParallel)
- Cache hard negatives (pre-mine once)
- Start with 1 epoch baseline, scale to 3 if needed

---

## ✅ TESTING CHECKLIST

### Before Training
- [ ] MedMentions dataset downloaded and parsed
- [ ] UMLS FAISS index loaded successfully
- [ ] Hard negative mining produces diverse negatives
- [ ] Dataset class returns correct tensor shapes
- [ ] DataLoader batching works correctly

### During Training
- [ ] Loss decreases steadily
- [ ] Validation metrics improve
- [ ] No NaN/Inf losses
- [ ] GPU utilization > 80%
- [ ] Training completes in < 8 hours

### After Training
- [ ] Model checkpoint saved correctly
- [ ] Validation F1 ≥ 0.85
- [ ] Validation ROC-AUC ≥ 0.92
- [ ] Cross-encoder avg score ≥ 0.80 on test set

### Integration Testing
- [ ] Fine-tuned model loads in pipeline
- [ ] Inference speed acceptable (< 0.5s per batch)
- [ ] End-to-end quality metrics meet targets
- [ ] No runtime errors during full pipeline run

---

## 📈 EXPECTED TIMELINE

| Day | Tasks | Deliverables |
|-----|-------|--------------|
| **Day 1** | Data prep + hard negative mining | 2.4M training samples |
| **Day 2** | Model training (4-6h GPU) | Fine-tuned checkpoint |
| **Day 3** | Calibration + integration + testing | Fully integrated system |
| **Total** | 2-3 days | Phase 2 complete |

---

**Author**: GFM-RAG Team
**Version**: 1.0
**Last Updated**: 2026-01-16
**Status**: 🚧 IN PROGRESS

**Next Action**: Begin Day 1 implementation (data preparation)
