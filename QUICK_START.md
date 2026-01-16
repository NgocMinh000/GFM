# Phase 2 Quick Start Guide

**Fast track to get fine-tuned cross-encoder running**

---

## Prerequisites

- GPU with 12+ GB VRAM (recommended: V100, A100)
- 32+ GB RAM
- CUDA 11.0+
- Python 3.8+
- MedMentions dataset downloaded
- UMLS data loaded

---

## Option 1: Automated (Recommended)

Run the complete pipeline automatically:

```bash
# 1. Activate environment
conda activate gfm-training

# 2. Run full pipeline (takes ~6-7 hours)
./run_full_training.sh

# This will:
# - Prepare data (Day 1)
# - Train model (Day 2) - ~6 hours
# - Calibrate & tune thresholds (Day 3) - ~15 min
# - Test integration
# - Generate all artifacts
```

**Output:**
- Model: `models/cross_encoder_finetuned/checkpoint-best/`
- Calibrator: `models/calibration/platt_scaler.pkl`
- Thresholds: `models/calibration/adaptive_tuner.pkl`
- Results: `results/phase2/`

---

## Option 2: Manual (Step-by-Step)

### Step 1: Train Model (Day 2)

```bash
python -m gfmrag.umls_mapping.training.cross_encoder_trainer \
    --config gfmrag/umls_mapping/training/config/training_config.yaml \
    --output_dir models/cross_encoder_finetuned
```

**Expected time:** 4-6 hours on V100
**Expected result:** Val F1 ≥ 0.90

### Step 2: Calibrate (Day 3)

```bash
python -m gfmrag.umls_mapping.training.calibrate_and_tune \
    --model_path models/cross_encoder_finetuned/checkpoint-best \
    --val_data tmp/training/medmentions_splits.pkl \
    --output_dir models/calibration
```

**Expected time:** 10-15 minutes
**Expected result:** ECE improvement ~35%

### Step 3: Test Integration

```bash
python -m gfmrag.umls_mapping.training.integration_test \
    --model_path models/cross_encoder_finetuned/checkpoint-best \
    --calibrator_path models/calibration/platt_scaler.pkl \
    --threshold_tuner_path models/calibration/adaptive_tuner.pkl
```

**Expected result:** Avg score ≥ 0.85, High confidence ≥ 40%

---

## Option 3: Use Pre-Trained (If Available)

If you have a pre-trained model:

```bash
# 1. Copy model files
cp -r /path/to/pretrained/checkpoint-best models/cross_encoder_finetuned/
cp /path/to/pretrained/platt_scaler.pkl models/calibration/
cp /path/to/pretrained/adaptive_tuner.pkl models/calibration/

# 2. Test it works
python -m gfmrag.umls_mapping.training.integration_test \
    --model_path models/cross_encoder_finetuned/checkpoint-best \
    --calibrator_path models/calibration/platt_scaler.pkl \
    --threshold_tuner_path models/calibration/adaptive_tuner.pkl
```

---

## Integration into Stage 3

### Minimal Change

In your Stage 3 pipeline file:

```python
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

### Config-Based (Recommended)

Use the production config:

```python
import yaml
from gfmrag.umls_mapping.cross_encoder_finetuned import FineTunedCrossEncoderReranker

# Load config
with open("gfmrag/umls_mapping/training/config/production_config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize reranker
if config["stage3"]["use_finetuned"]:
    reranker = FineTunedCrossEncoderReranker(
        model_path=config["model"]["finetuned_model_path"],
        calibrator_path=config["calibration"]["calibrator_path"],
        threshold_tuner_path=config["adaptive_thresholds"]["threshold_tuner_path"],
        device=config["inference"]["device"],
        batch_size=config["inference"]["batch_size"],
    )
else:
    # Fallback to baseline
    from gfmrag.umls_mapping.cross_encoder_reranker import CrossEncoderReranker
    reranker = CrossEncoderReranker(config)
```

---

## Verify Results

After training, check these metrics:

```bash
# 1. Training metrics (from logs)
grep "Val F1" logs/training.log | tail -1
# Should show: Val F1: 0.9000+

# 2. Evaluation results
cat results/phase2/evaluation/evaluation_results.json | python -m json.tool
# Look for:
# - "f1": 0.90+
# - "roc_auc": 0.95+
# - "ece": <0.03

# 3. Integration test
cat results/phase2/integration_test.json | python -m json.tool
# Look for:
# - "avg_score": 0.85+
# - "high_confidence_pct": 40.0+
```

---

## Troubleshooting

### Training is slow
```bash
# Check GPU usage
nvidia-smi

# If GPU not used, verify CUDA:
python -c "import torch; print(torch.cuda.is_available())"
```

### Out of memory
```yaml
# In training_config.yaml, reduce batch size:
training:
  batch_size: 16  # Was 32
  gradient_accumulation_steps: 4  # Was 2
```

### Low performance
```bash
# Check if using fine-tuned model (not baseline):
python -c "
from gfmrag.umls_mapping.cross_encoder_finetuned import FineTunedCrossEncoderReranker
r = FineTunedCrossEncoderReranker(
    model_path='models/cross_encoder_finetuned/checkpoint-best',
    calibrator_path='models/calibration/platt_scaler.pkl',
    threshold_tuner_path='models/calibration/adaptive_tuner.pkl',
)
stats = r.get_statistics()
print(stats)
# Verify: model_loaded=True, calibrator_loaded=True, threshold_tuner_loaded=True
"
```

---

## Expected Timeline

| Task | Time | Bottleneck |
|------|------|------------|
| Data preparation | 5-10 min | Disk I/O |
| Model training | 4-6 hours | GPU compute |
| Calibration | 10-15 min | CPU |
| Testing | 5-10 min | GPU |
| **Total** | **~6-7 hours** | Training |

**GPU Comparison:**
- V100 (16GB): ~5-6 hours
- A100 (40GB): ~3-4 hours
- RTX 3090: ~6-7 hours
- CPU only: 3-5 days (not recommended)

---

## Success Criteria

Training is successful if:
- ✅ Training completes without errors
- ✅ Val F1 ≥ 0.90
- ✅ ROC-AUC ≥ 0.95
- ✅ ECE < 0.03
- ✅ Integration tests pass
- ✅ High confidence % ≥ 40%
- ✅ Avg cross-encoder score ≥ 0.85
- ✅ Avg score margin ≥ 0.25

---

## Next Steps

After successful training:

1. **Backup model files:**
   ```bash
   tar -czf phase2_models.tar.gz \
       models/cross_encoder_finetuned \
       models/calibration
   ```

2. **Deploy to production:**
   - Copy model files to production server
   - Update Stage 3 pipeline code
   - Test on real data
   - Monitor metrics

3. **Monitor in production:**
   - High confidence percentage
   - Average cross-encoder score
   - Calibration error
   - Throughput

---

## Documentation

- **Detailed Training:** `TRAINING_GUIDE.md`
- **Integration:** `DAY3_INTEGRATION_GUIDE.md`
- **Complete Summary:** `PHASE2_COMPLETION_SUMMARY.md`
- **Phase 2 Plan:** `STAGE3_PHASE2_PLAN.md`

---

## Support

If you encounter issues:

1. Check logs: `logs/training.log`
2. Review TensorBoard: `tensorboard --logdir tmp/training/tensorboard`
3. See troubleshooting sections in guides above
4. Check integration test output
5. Open GitHub issue with error logs

---

**Ready to start?** Run `./run_full_training.sh` 🚀
