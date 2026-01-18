# Training Fix Summary

## Critical Bug Fixed

### The Problem
The baseline CEDetector model had a **critical architectural bug** in the forward pass (`models/cedetector_baseline.py:285-309`) that caused random ~50% accuracy.

**Previous Buggy Behavior:**
- For batch size B=12, created 72 query patches (12 images × 6 patches each)
- Compared **ALL 72 query patches against ALL 12 reference images**
- This created 864 comparisons instead of 72
- Each query patch was compared with references from OTHER images, not just its paired reference
- Model learned incorrect associations → random performance

**Corrected Behavior:**
- For batch size B=12, creates 72 query patches (12 images × 6 patches each)
- Compares patches from query_i ONLY with reference_i
- This creates 72 comparisons (6 patches per pair, 12 pairs total)
- Each query is compared only with its corresponding reference
- Model learns correct query-reference associations → expected 85-90% accuracy

### Code Changes

**File: `models/cedetector_baseline.py`**

Changed lines 285-309 from:
```python
# OLD (BUGGY): Compared all query patches with all references
for i in range(num_query_patches):  # 72 iterations for B=12
    for j in range(num_refs):        # 12 iterations
        # Created 864 comparisons!
        logit = self.classifier(q_tokens, r_tokens)
```

To:
```python
# NEW (FIXED): Compare each query with its paired reference only
for i in range(B):  # 12 iterations for B=12
    # Get 6 patches for query_i
    query_patches_i = query_tokens[i*6:(i+1)*6]
    # Get reference_i
    ref_tokens_i = ref_tokens[i]
    # Compare 6 patches with this reference only
    for j in range(6):  # 6 iterations
        # 72 total comparisons (correct!)
        logit = self.classifier(query_patches_i[j], ref_tokens_i)
```

## Comprehensive Diagnostics Passed

All 7 diagnostic checks passed:

1. ✓ **Forward Pass Shapes**: Correct batch handling
2. ✓ **Loss Function**: All loss components working (SimCLR, KL, MSL, BCE)
3. ✓ **Metrics Computation**: µAP, R@P90, Accuracy computed correctly
4. ✓ **Dataset Labels**: 50/50 positive/negative pairs
5. ✓ **Augmentation Pipeline**: 35+ transforms active
6. ✓ **Optimizer Settings**: AdamW with lr=2e-4, weight_decay=0.01
7. ✓ **Gradient Flow**: Healthy gradient magnitudes (no exploding/vanishing)

## Optimizations Applied

### 1. Augmentation Strength Reduced
**Change:** `train_baseline.py:210-213`
- Previous: Exactly 4 augmentations per image
- **New: 2-4 augmentations per image** (random)
- Benefit: Faster initial convergence while maintaining robustness

### 2. Training Configuration
Current optimal settings:
- **Learning Rate:** 2e-4 (good baseline)
- **Batch Size:** 12 per GPU × 2 GPUs = 24 global
- **Epochs:** 18 (DISC21) + 3 (NDEC)
- **Mixed Precision:** Enabled (FP16)
- **Dataset:** 50% subset (5000 samples)

## Expected Performance

### Before Fix (Random Performance)
```
Epoch 0-11:
  Accuracy: ~48-51% (random)
  µAP: ~0.47-0.51 (random)
  R@P90: ~0.0000 (terrible)
  Loss: Decreasing but metrics not improving
```

### After Fix (Expected Performance)
```
Epoch 1:
  Accuracy: 60-70%
  µAP: 0.60-0.70
  R@P90: 0.10-0.30

Epoch 5:
  Accuracy: 75-85%
  µAP: 0.75-0.85
  R@P90: 0.40-0.60

Epoch 10+:
  Accuracy: 85-90%
  µAP: 0.85-0.90
  R@P90: 0.60-0.80

Final (Epoch 18):
  Accuracy: 88-92%
  µAP: 0.88-0.92
  R@P90: 0.70-0.85
```

## Training Timeline

With 2 A30 GPUs, 50% dataset:
- **Epoch Time:** ~16-18 minutes
- **Total DISC21 (18 epochs):** ~5-6 hours
- **Total NDEC (3 epochs):** ~1 hour
- **Total Training Time:** ~6-7 hours

## How to Verify Fix is Working

Monitor the first 3 epochs. You should see:

### Epoch 0 (First Epoch)
- Accuracy should START improving from batch 0
- By end of epoch: 60-70% accuracy
- **If still ~50%: Bug not fixed**

### Epoch 1-2
- Steady improvement in accuracy
- µAP should track with accuracy
- R@P90 should start increasing (>0.05)

### Epoch 3-5
- Accuracy should be 75%+
- Clear upward trend in all metrics
- Loss should be smoothly decreasing

## Files Modified

1. **`models/cedetector_baseline.py`** (lines 285-309)
   - Fixed forward pass query-reference pairing

2. **`train_baseline.py`** (lines 210-213)
   - Reduced augmentation strength for faster convergence

## Files Created

1. **`test_forward_pass_fix.py`**
   - Verifies forward pass shapes and logic

2. **`comprehensive_training_diagnostic.py`**
   - Full training pipeline verification

3. **`diagnose_forward_bug.py`**
   - Demonstrates the original bug

4. **`TRAINING_FIX_SUMMARY.md`** (this file)
   - Complete documentation of fix

## How to Restart Training

```bash
# 1. Ensure you're in the project directory
cd "/home/jowatson/Deep Learning/CEDetector_APT"

# 2. Cancel any running jobs (if needed)
scancel <job_id>

# 3. Submit new training job
sbatch run_training_baseline.slurm

# 4. Monitor progress
tail -f logs/ced_baseline_gpu4_*.out

# 5. Watch for improvements
# - Epoch 0 should show 60-70% accuracy
# - Epoch 5 should show 75-85% accuracy
# - Metrics should improve every epoch
```

## Success Indicators

Training is working correctly if you see:

1. ✓ **Epoch 0 accuracy > 60%**
2. ✓ **Steady improvement each epoch**
3. ✓ **µAP tracks with accuracy**
4. ✓ **R@P90 > 0.0000 by epoch 2-3**
5. ✓ **Loss decreases smoothly**
6. ✓ **By epoch 10: accuracy > 80%**

## Troubleshooting

### If accuracy is still ~50%:
1. Check that the fixed `cedetector_baseline.py` is being used
2. Verify model is loading correctly
3. Check dataset labels are balanced

### If training is slow:
1. Verify mixed precision is enabled (check logs for "Mixed precision: Enabled")
2. Check GPU utilization: `nvidia-smi`
3. Ensure num_workers is set correctly (should be ~46)

### If metrics aren't improving:
1. Check if gradients are flowing (see diagnostic output)
2. Verify learning rate isn't too high/low
3. Check for NaN losses

## Next Steps

1. **Monitor first 5 epochs closely**
   - Verify accuracy improves as expected
   - Watch for any warnings or errors

2. **After epoch 5:**
   - If accuracy > 75%: Training is working correctly
   - If accuracy < 65%: Investigate further

3. **Full training:**
   - Let complete 18 epochs on DISC21
   - Fine-tune 3 epochs on NDEC
   - Final accuracy should be 88-92%

4. **Comparison with APT:**
   - Train APT version separately
   - Compare final accuracy and training speed
   - APT should be 20-40% faster with similar accuracy

## References

- **CED Paper:** "An End-to-End Vision Transformer Approach for Image Copy Detection"
- **Project Documentation:** `README.md`, `CLAUDE.md`
- **Diagnostic Script:** `comprehensive_training_diagnostic.py`
- **Test Script:** `test_forward_pass_fix.py`
