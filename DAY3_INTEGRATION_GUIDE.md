# Day 3: Integration, Calibration & Testing Guide

**Complete guide for integrating fine-tuned cross-encoder into Stage 3 pipeline**

---

## Overview

After completing Day 2 training, Day 3 focuses on:
1. **Calibration**: Improve probability estimates with Platt scaling
2. **Adaptive Thresholds**: Tune thresholds per entity type
3. **Integration**: Use fine-tuned model in Stage 3 pipeline
4. **Testing**: Verify improvements

**Prerequisites:**
- ✅ Day 2 training complete
- ✅ Model checkpoint at `models/cross_encoder_finetuned/checkpoint-best`
- ✅ Validation F1 ≥ 0.90

---

## Step 1: Calibration & Threshold Tuning

### What is Calibration?

Calibration ensures predicted probabilities match actual frequencies:
- **Uncalibrated**: Model predicts 0.8 confidence, but only correct 60% of the time
- **Calibrated**: Model predicts 0.8 confidence, and is correct 80% of the time

### What are Adaptive Thresholds?

Different entity types require different thresholds:
- **Disease** entities (common, easier): threshold = 0.45
- **Pharmacologic Substance** (rare, harder): threshold = 0.65

This improves precision/recall tradeoff per type.

### Run Calibration & Tuning

```bash
# Activate environment
conda activate gfm-training

# Run calibration and threshold tuning (takes ~10-15 minutes)
python -m gfmrag.umls_mapping.training.calibrate_and_tune \
    --model_path models/cross_encoder_finetuned/checkpoint-best \
    --val_data tmp/training/medmentions_splits.pkl \
    --output_dir models/calibration \
    --calibration_method compare \
    --threshold_objective f1

# This will:
# 1. Compare calibration methods (Platt, Temperature, Isotonic)
# 2. Select best method (usually Platt scaling)
# 3. Tune thresholds per entity type (optimize F1)
# 4. Save calibrator and threshold tuner
```

### Expected Output

```
================================================================================
STEP 1: PROBABILITY CALIBRATION
================================================================================

Calibration fitted:
  ECE before: 0.0234
  ECE after:  0.0156
  Improvement: 0.0078

================================================================================
CALIBRATION METHOD COMPARISON
================================================================================
Method          ECE        Brier      NLL
--------------------------------------------------------------------------------
uncalibrated    0.0234     0.0876     0.2341
platt           0.0156     0.0812     0.2198
temperature     0.0162     0.0823     0.2214
isotonic        0.0149     0.0798     0.2176
================================================================================

Best calibration method: isotonic

================================================================================
STEP 2: ADAPTIVE THRESHOLD TUNING
================================================================================

Type 'Amino Acid, Peptide, or Protein': threshold=0.487, F1=0.9123, samples=12453
Type 'Antibiotic': threshold=0.523, F1=0.8967, samples=3421
Type 'Disease or Syndrome': threshold=0.456, F1=0.9245, samples=45678
...

================================================================================
ADAPTIVE THRESHOLD SUMMARY
================================================================================
Total entity types: 127
  Tuned thresholds: 89
  Default thresholds: 38

Threshold distribution (tuned types):
  Min:    0.387
  Q1:     0.468
  Median: 0.512
  Q3:     0.567
  Max:    0.689

Top 5 types with LOWEST thresholds (easiest to link):
  Disease or Syndrome                      0.387 (F1=0.9456)
  Sign or Symptom                          0.412 (F1=0.9234)
  Therapeutic or Preventive Procedure      0.434 (F1=0.9123)
  ...

Top 5 types with HIGHEST thresholds (hardest to link):
  Enzyme                                   0.689 (F1=0.8567)
  Nucleic Acid, Nucleoside, or Nucleotide  0.652 (F1=0.8723)
  Receptor                                 0.634 (F1=0.8834)
  ...

================================================================================
```

### Verify Generated Files

```bash
ls -lh models/calibration/
# Expected files:
# - platt_scaler.pkl (or isotonic_calibrator.pkl, temperature_scaler.pkl)
# - adaptive_tuner.pkl
# - adaptive_tuner.hierarchy.pkl
# - val_negatives_cache.pkl
```

---

## Step 2: Integration Testing

Test that everything works together before deploying.

### Run Integration Test

```bash
# Test with default entities
python -m gfmrag.umls_mapping.training.integration_test \
    --model_path models/cross_encoder_finetuned/checkpoint-best \
    --calibrator_path models/calibration/platt_scaler.pkl \
    --threshold_tuner_path models/calibration/adaptive_tuner.pkl

# Test with custom entities
python -m gfmrag.umls_mapping.training.integration_test \
    --model_path models/cross_encoder_finetuned/checkpoint-best \
    --calibrator_path models/calibration/platt_scaler.pkl \
    --threshold_tuner_path models/calibration/adaptive_tuner.pkl \
    --test_entities "diabetic neuropathy,myocardial infarction,hypertension,chronic kidney disease" \
    --output results/integration_test.json
```

### Expected Output

```
================================================================================
RERANKER STATISTICS
================================================================================
Model loaded:              True
Calibration enabled:       True
Calibrator loaded:         True
Adaptive thresholds:       True
Threshold tuner loaded:    True
Device:                    cuda

Threshold Statistics:
  Num types:        127
  Num tuned:        89
  Threshold range:  0.387 - 0.689
  Threshold median: 0.512
================================================================================

Testing entity: 'diabetic neuropathy'
  Top match: Diabetic Neuropathies (CUI: C0011882)
  Score: 0.9234
  Score margin: 0.3456
  Confidence: high

Testing entity: 'myocardial infarction'
  Top match: Myocardial Infarction (CUI: C0027051)
  Score: 0.9567
  Score margin: 0.4123
  Confidence: high

...

================================================================================
BATCH TEST RESULTS
================================================================================
Entities tested:         10
Successfully mapped:     10 (100.0%)
Failed to map:           0 (0.0%)

Average score:           0.8967
Median score:            0.9123
Average margin:          0.3234
Median margin:           0.3456
High confidence (≥0.7):  90.0%
================================================================================

================================================================================
EXPECTED IMPROVEMENTS (from Phase 2 Plan)
================================================================================
High Confidence Mappings:
  Baseline:   5-10%
  Fine-Tuned: 40-60%
  Improvement: +30-50%

Cross-Encoder Score:
  Baseline:   0.58
  Fine-Tuned: 0.85+
  Improvement: +47%

Score Margin:
  Baseline:   0.10-0.12
  Fine-Tuned: 0.25+
  Improvement: +100-150%
================================================================================

ACTUAL RESULTS (Fine-Tuned):
  Avg Score:      0.8967
  Avg Margin:     0.3234
  High Conf %:    90.0%
```

### ✅ Success Criteria

Integration test passes if:
- [x] All components loaded successfully
- [x] Average score ≥ 0.85
- [x] Average margin ≥ 0.25
- [x] High confidence % ≥ 40%

---

## Step 3: Production Integration

### Option A: Update Existing Pipeline (Recommended)

Modify your Stage 3 pipeline to use the fine-tuned model:

```python
# In your Stage 3 pipeline script (e.g., stage3_umls_mapping.py)

# OLD:
from gfmrag.umls_mapping.cross_encoder_reranker import CrossEncoderReranker
self.cross_encoder = CrossEncoderReranker(self.config)

# NEW:
from gfmrag.umls_mapping.cross_encoder_finetuned import FineTunedCrossEncoderReranker
self.cross_encoder = FineTunedCrossEncoderReranker(
    model_path="models/cross_encoder_finetuned/checkpoint-best",
    calibrator_path="models/calibration/platt_scaler.pkl",
    threshold_tuner_path="models/calibration/adaptive_tuner.pkl",
    device="cuda",
)
```

### Option B: Config-Based Integration

Add to your config file:

```yaml
# config/stage3_config.yaml

cross_encoder:
  use_finetuned: true
  model_path: "models/cross_encoder_finetuned/checkpoint-best"
  calibrator_path: "models/calibration/platt_scaler.pkl"
  threshold_tuner_path: "models/calibration/adaptive_tuner.pkl"
  device: "cuda"
  batch_size: 32
```

Then in code:

```python
from gfmrag.umls_mapping.cross_encoder_finetuned import FineTunedCrossEncoderReranker

if config.cross_encoder.use_finetuned:
    self.cross_encoder = FineTunedCrossEncoderReranker.from_config(config)
else:
    self.cross_encoder = CrossEncoderReranker(config)
```

---

## Step 4: End-to-End Pipeline Test

Run full Stage 3 pipeline on test data.

### Prepare Test Data

```bash
# Use a small subset for testing
head -n 100 data/stage2_output.jsonl > data/stage2_test_subset.jsonl
```

### Run Pipeline

```bash
# Run Stage 3 with fine-tuned model
python -m gfmrag.workflow.stage3_umls_mapping \
    --input data/stage2_test_subset.jsonl \
    --output data/stage3_test_output_finetuned.jsonl \
    --config config/stage3_config.yaml \
    --use_finetuned true

# Compare with baseline (optional)
python -m gfmrag.workflow.stage3_umls_mapping \
    --input data/stage2_test_subset.jsonl \
    --output data/stage3_test_output_baseline.jsonl \
    --config config/stage3_config.yaml \
    --use_finetuned false
```

### Analyze Results

```python
import json

# Load results
with open("data/stage3_test_output_finetuned.jsonl") as f:
    finetuned = [json.loads(line) for line in f]

with open("data/stage3_test_output_baseline.jsonl") as f:
    baseline = [json.loads(line) for line in f]

# Compare high confidence mappings
finetuned_high = [m for m in finetuned if m["confidence_tier"] == "high"]
baseline_high = [m for m in baseline if m["confidence_tier"] == "high"]

print(f"High confidence mappings:")
print(f"  Baseline:   {len(baseline_high)}/{len(baseline)} ({len(baseline_high)/len(baseline)*100:.1f}%)")
print(f"  Fine-Tuned: {len(finetuned_high)}/{len(finetuned)} ({len(finetuned_high)/len(finetuned)*100:.1f}%)")
print(f"  Improvement: +{(len(finetuned_high)/len(finetuned) - len(baseline_high)/len(baseline))*100:.1f}%")

# Compare average scores
import numpy as np
finetuned_scores = [m["cross_encoder_score"] for m in finetuned]
baseline_scores = [m["cross_encoder_score"] for m in baseline]

print(f"\nAverage cross-encoder score:")
print(f"  Baseline:   {np.mean(baseline_scores):.4f}")
print(f"  Fine-Tuned: {np.mean(finetuned_scores):.4f}")
print(f"  Improvement: +{(np.mean(finetuned_scores) - np.mean(baseline_scores))*100:.1f}%")
```

---

## Step 5: Production Deployment

### Checklist

Before deploying to production:

- [ ] Integration tests pass (Step 2)
- [ ] End-to-end pipeline test completes (Step 4)
- [ ] Results show expected improvements:
  - [ ] High confidence % increased by 30-50%
  - [ ] Average cross-encoder score ≥ 0.85
  - [ ] Score margins ≥ 0.25
- [ ] Model files backed up:
  - [ ] `models/cross_encoder_finetuned/checkpoint-best/`
  - [ ] `models/calibration/platt_scaler.pkl`
  - [ ] `models/calibration/adaptive_tuner.pkl`
- [ ] Documentation updated
- [ ] Team notified of changes

### Deploy

1. **Copy model files to production server:**

```bash
# On local machine
tar -czf finetuned_models.tar.gz \
    models/cross_encoder_finetuned/checkpoint-best \
    models/calibration

# Transfer to production
scp finetuned_models.tar.gz user@production-server:/path/to/models/

# On production server
cd /path/to/models/
tar -xzf finetuned_models.tar.gz
```

2. **Update production config:**

```yaml
# production_config.yaml
cross_encoder:
  use_finetuned: true
  model_path: "/path/to/models/cross_encoder_finetuned/checkpoint-best"
  calibrator_path: "/path/to/models/calibration/platt_scaler.pkl"
  threshold_tuner_path: "/path/to/models/calibration/adaptive_tuner.pkl"
```

3. **Restart pipeline:**

```bash
# Stop current pipeline
pkill -f stage3_umls_mapping

# Start with new config
python -m gfmrag.workflow.stage3_umls_mapping \
    --config production_config.yaml \
    --input production_input.jsonl \
    --output production_output.jsonl
```

4. **Monitor metrics:**

```bash
# Watch logs for improvements
tail -f logs/stage3_pipeline.log | grep "high confidence"

# Check TensorBoard (if logging enabled)
tensorboard --logdir logs/production --port 6007
```

---

## Monitoring & Maintenance

### Key Metrics to Track

Monitor these metrics in production:

1. **High Confidence Mappings %**
   - Target: 40-60% (up from 5-10%)
   - Alert if drops below 30%

2. **Average Cross-Encoder Score**
   - Target: 0.85+ (up from 0.58)
   - Alert if drops below 0.80

3. **Average Score Margin**
   - Target: 0.25+ (up from 0.10-0.12)
   - Alert if drops below 0.20

4. **Calibration Error (ECE)**
   - Target: <0.03 (well-calibrated)
   - Alert if exceeds 0.05

5. **Throughput**
   - Target: Similar to baseline (~10-20% slower due to better model)
   - Alert if drops by >30%

### Regular Checks

**Daily:**
- Monitor high confidence percentage
- Check for errors in logs

**Weekly:**
- Review calibration metrics
- Analyze low confidence cases

**Monthly:**
- Evaluate model drift (score degradation over time)
- Consider retraining if performance degrades >5%

### Retraining Schedule

Retrain the model when:
- New medical entities emerge (major UMLS update)
- Performance degrades >5% from baseline
- New domain-specific requirements
- At least 6 months since last training

---

## Troubleshooting

### Issue: Low High Confidence %

**Symptoms:**
- High confidence mappings <30%
- Below expected 40-60% range

**Causes:**
- Calibration not applied correctly
- Thresholds too aggressive
- Model loaded incorrectly

**Solutions:**

1. **Verify calibration loaded:**
   ```python
   reranker = FineTunedCrossEncoderReranker(...)
   stats = reranker.get_statistics()
   print(stats)  # Check calibrator_loaded=True
   ```

2. **Adjust thresholds:**
   ```python
   # Re-tune with different objective
   python -m gfmrag.umls_mapping.training.calibrate_and_tune \
       --threshold_objective recall  # Lower thresholds
   ```

3. **Check model checkpoint:**
   ```bash
   # Verify model file exists and is not corrupted
   ls -lh models/cross_encoder_finetuned/checkpoint-best/pytorch_model.bin
   # Should be ~400-500 MB
   ```

### Issue: Scores Too Low

**Symptoms:**
- Average cross-encoder score <0.80
- Expected ≥0.85

**Causes:**
- Wrong model loaded (baseline instead of fine-tuned)
- Model undertrained

**Solutions:**

1. **Verify fine-tuned model:**
   ```python
   import torch
   state_dict = torch.load("models/cross_encoder_finetuned/checkpoint-best/pytorch_model.bin")
   print(len(state_dict))  # Should show all model parameters
   ```

2. **Check training metrics:**
   ```bash
   # Review training log for final F1
   grep "Val F1" logs/training.log | tail -1
   # Should show ≥0.90
   ```

3. **Try best epoch:**
   ```bash
   # If using checkpoint-epoch-3, try checkpoint-best
   ls models/cross_encoder_finetuned/checkpoint-best/
   ```

### Issue: Slow Performance

**Symptoms:**
- Pipeline 2-3x slower than baseline
- Expected only 10-20% slower

**Causes:**
- CPU inference (should use GPU)
- Inefficient batching
- Model not optimized

**Solutions:**

1. **Verify GPU usage:**
   ```python
   reranker = FineTunedCrossEncoderReranker(..., device="cuda")
   print(reranker.device)  # Should show 'cuda'
   ```

2. **Increase batch size:**
   ```python
   reranker = FineTunedCrossEncoderReranker(..., batch_size=64)  # Was 32
   ```

3. **Enable TorchScript (optional):**
   ```python
   # Compile model for faster inference
   model = torch.jit.script(reranker.model)
   ```

---

## Next Steps (Optional Improvements)

After successful Day 3 integration, consider:

### 1. Ensemble Methods
Combine multiple models for better accuracy:
- Fine-tuned cross-encoder (current)
- SapBERT bi-encoder (Stage 2)
- TF-IDF (Stage 2)
- Weighted voting

### 2. Active Learning
Improve model with user feedback:
- Collect corrections from users
- Retrain with high-quality examples
- Focus on low confidence cases

### 3. Multi-Task Learning
Train on related tasks simultaneously:
- Entity linking (current)
- Semantic type classification
- Relation extraction

### 4. Domain Adaptation
Fine-tune for specific domains:
- Clinical notes
- Research papers
- Patient forums

---

## Summary

Day 3 completes Phase 2 integration:

✅ **Calibration** - Probabilities match actual confidence
✅ **Adaptive Thresholds** - Optimized per entity type
✅ **Integration** - Fine-tuned model in production
✅ **Testing** - Verified improvements

**Expected Results:**
- High confidence: **40-60%** (was 5-10%)
- Cross-encoder score: **0.85+** (was 0.58)
- Score margin: **0.25+** (was 0.10-0.12)

**Next Phase:**
Phase 3 (Advanced Optimization) - See `STAGE3_PHASE1_IMPROVEMENTS.md` for roadmap.

---

## Resources

- **Phase 2 Plan**: `STAGE3_PHASE2_PLAN.md`
- **Training Guide**: `TRAINING_GUIDE.md`
- **Architecture Docs**: `ARCHITECTURE.md`
- **API Reference**: Code docstrings in each module

**Support:**
- GitHub Issues: [Report bugs]
- Contact: [Your team contact]

---

**Congratulations on completing Day 3! 🎉**

Your UMLS mapping quality has significantly improved through fine-tuning, calibration, and adaptive thresholds.
