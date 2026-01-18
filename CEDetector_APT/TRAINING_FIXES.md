# Training Fixes - Low Accuracy and R@P90=0 Resolution

## Problem Summary

The training showed critical issues:
- **Accuracy**: ~50% (random guessing for binary classification)
- **R@P90**: 0.0 (model cannot achieve 90% precision at any threshold)
- **Loss**: 0.0 in logs (incorrect metric tracking)

## Root Causes Identified

### 1. Loss Logging Bug
**Location**: `train.py:158`

**Issue**: The code tried to log `metrics.get('loss', 0.0)` but `MetricsTracker.compute()` returns `'avg_loss'`, not `'loss'`.

**Impact**: Loss appeared as 0.0 in all training logs, making it impossible to track training progress.

**Fix**:
```python
# Before
'loss': metrics.get('loss', 0.0),

# After
'loss': metrics.get('avg_loss', 0.0),
```

### 2. Logits Defaulting to Zeros
**Location**: `train.py:93`

**Issue**: When the model didn't return 'logits' key, it defaulted to zeros:
```python
logits = outputs.get('logits', torch.zeros(query.size(0), device=device))
```

**Impact**:
- `torch.sigmoid(zeros) = 0.5` for all predictions
- All predictions identical → accuracy ~50%
- No precision-recall variation → R@P90 = 0

**Fix**:
```python
# Before
logits = outputs.get('logits', torch.zeros(...))

# After
if 'logits' not in outputs:
    raise ValueError(f"Model output missing 'logits' key. Available keys: {outputs.keys()}")
logits = outputs['logits']
```

This forces an immediate error if the model isn't working correctly, rather than silently producing garbage results.

### 3. Missing Validation Error Handling
**Location**: `train.py:169-189`

**Issue**: Validation used `outputs.get('scores', zeros)` which had the same problem as training.

**Fix**: Changed to use 'logits' key with explicit error checking.

### 4. Shape Mismatch in Loss Computation
**Location**: `utils/losses.py:219-225`, `train.py:121-125`

**Issue**: BCE loss expects both logits and labels to have shape (B, 1), but sometimes received (B,).

**Fix**: Added explicit shape handling in both the loss function and fallback BCE.

### 5. Missing Debug Information
**Location**: `train.py:139-153`

**Issue**: No visibility into what the model was actually predicting during training.

**Fix**: Added diagnostic logging every 10 batches:
```
Logits: [min, max] | Preds: [min, max] | Labels: mean
```

This helps identify issues like:
- All logits are the same
- Model collapsed to one class
- Labels are imbalanced

## Files Modified

### 1. `models/dinov3_backbone.py`
- Line 8: Removed unused `AutoImageProcessor` import
- Line 43: Added `attn_implementation="eager"` to enable attention capture and eliminate warnings

### 2. `train.py`
- Line 93-100: Added error check for missing 'logits' key
- Line 112-125: Fixed BCE loss shape handling
- Line 146-153: Added diagnostic logging
- Line 158-163: Fixed loss logging key and added mu_ap, r_at_p90
- Line 195-200: Fixed validation to use 'logits' key with error checking

### 3. `utils/losses.py`
- Line 219-225: Added shape handling for BCE loss to prevent dimension mismatches

### 4. New Files Created

#### `diagnose_training_issue.py`
Diagnostic script that analyzes training metrics to identify:
- Loss tracking bugs
- Accuracy issues
- Suspicious behavior patterns

#### `test_training_fix.py`
Comprehensive test suite that verifies:
1. Model forward pass returns proper logits
2. Loss computation works correctly
3. Metrics tracker computes all metrics
4. Data loader produces balanced batches

#### `TRAINING_FIXES.md`
This document - complete explanation of issues and fixes.

#### `ATTENTION_FIX.md`
Documentation of the attention implementation fix to eliminate HuggingFace warnings.

## How to Verify Fixes

### Quick Test (5 minutes)
```bash
cd "/home/jowatson/Deep Learning/CEDetector_APT"
conda activate gpu_training
python3 test_training_fix.py
```

This will run all verification tests and confirm fixes work.

### Small Training Run (30 minutes)
```bash
# Single GPU test with tiny subset
python3 train.py \
    --epochs 3 \
    --ndec_epochs 0 \
    --batch_size 8 \
    --subset_fraction 0.001 \
    --save_dir ./test_fixes_checkpoints \
    --no_amp
```

Watch for:
- Loss should be non-zero (typically 0.5-2.0 initially)
- Accuracy should change over time (not stuck at 50%)
- Logits range should vary between batches
- Predictions should not all be 0.5

### Full Training
Once verified, submit normal training job:
```bash
sbatch run_training.slurm
```

## Expected Behavior After Fixes

### Initial Training (Epoch 0-2)
- **Loss**: 0.8-1.5 (decreasing)
- **Accuracy**: 55-65% (improving from random)
- **µAP**: 0.5-0.7
- **R@P90**: 0.0-0.3 (will improve as model learns)

### Mid Training (Epoch 10-15)
- **Loss**: 0.4-0.7
- **Accuracy**: 75-85%
- **µAP**: 0.75-0.85
- **R@P90**: 0.5-0.7

### Late Training (Epoch 20-25)
- **Loss**: 0.2-0.4
- **Accuracy**: 85-92%
- **µAP**: 0.85-0.90
- **R@P90**: 0.75-0.85

## Monitoring During Training

### Check Logs
```bash
tail -f logs/cedetector_apt_*.out
```

Look for:
```
Epoch 0 [0/XXX] Loss: 1.2345 | Logits: [-2.1, 3.4] | Preds: [0.1, 0.97] | Labels: 0.500
```

**Good signs**:
- Loss is decreasing over epochs
- Logits range is wide (e.g., -3 to +3)
- Predictions vary (not all 0.5)
- Labels around 0.5 (balanced dataset)

**Bad signs**:
- Loss stuck at same value
- Logits all near zero
- Predictions all similar (e.g., 0.48-0.52)
- Labels heavily skewed (e.g., 0.9 or 0.1)

### Check Metrics Log
```bash
tail -20 checkpoints_*/training_metrics.jsonl | python3 -m json.tool
```

Verify:
- `loss` is non-zero
- `mu_ap` is increasing
- `r_at_p90` starts improving after a few epochs
- `accuracy` trends upward

## Additional Improvements Made

1. **Enhanced Logging**: Added µAP and R@P90 to training logs
2. **Better Error Messages**: Clear errors when model output is malformed
3. **Diagnostic Tools**: Scripts to analyze and test training
4. **Shape Safety**: Explicit shape handling prevents silent failures

## Troubleshooting

### If accuracy still low after fixes:

1. **Check learning rate**
   ```python
   # In train.py:289, try reducing LR
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # was 2e-4
   ```

2. **Check if model is actually training**
   ```bash
   # Look at gradient norms
   grep "grad_norm" logs/*.out
   ```

3. **Verify labels are correct**
   ```python
   # Add to train.py after line 84:
   print(f"Label distribution: {labels.float().mean():.3f}")
   ```
   Should be around 0.5 for balanced data.

4. **Check for NaN losses**
   ```python
   # Add after line 120:
   if torch.isnan(loss):
       raise ValueError("NaN loss detected!")
   ```

### If R@P90 still zero after epoch 5:

This means the model cannot achieve 90% precision at any threshold. Possible causes:

1. **Model predicting only one class**: Check prediction range in logs
2. **Loss not backpropagating**: Verify gradients are flowing
3. **Dataset issues**: Verify labels are correct
4. **Model architecture issue**: Check classifier is being called

## Summary

The main issues were:
1. **Silent failure mode**: Defaulting to zeros masked the real problem
2. **Metric tracking bug**: Couldn't see loss values
3. **No diagnostics**: Couldn't tell what model was doing

The fixes:
1. **Fail fast**: Raise errors instead of defaulting to zeros
2. **Proper logging**: Track all important metrics correctly
3. **Enhanced diagnostics**: See what model predicts during training

These changes ensure that:
- Problems are detected immediately (not after hours of training)
- All metrics are tracked correctly
- You can monitor training progress effectively
