# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Adaptive-DINO-ICD is an Image Copy Detection system (Phase 1) combining:
- **DINOv3 Backbone**: Vision transformer from HuggingFace
- **APT (Adaptive Patch Tokenization)**: Entropy-based adaptive patch sizing with quadtree selection
- **ASL (Asymmetric Similarity Learning)**: Direction-aware loss with norm ratio enforcement
- **D2LV-inspired Inference**: Global-local matching interface

The system trains on two data streams with fixed 70/30 ratio: DISC (unlabeled self-supervised) and NDEC (labeled pairs with direction annotations).

## Environment Setup

Use the `gpu_training` conda environment:
```bash
conda activate gpu_training
cd "/home/jowatson/Deep Learning/Code"
pip install -e ".[dev]"
```

## Common Commands

```bash
# Generate dummy data for testing
python scripts/make_dummy_data.py --output_dir ./data

# Convert NDEC challenge format to training format
python scripts/convert_ndec_annotations.py --ndec_root "/home/jowatson/Deep Learning/NDEC"

# Run sanity check (verifies forward pass)
python scripts/sanity_check_forward.py

# Run all tests
pytest tests/ -v

# Run single test
pytest tests/test_asl_loss.py::test_positive_loss -v

# Train with offline stub (testing)
python scripts/train_phase1.py --config configs/phase1_offline_stub.yaml

# Train with production DINOv3 (requires HF access)
python scripts/train_phase1.py --config configs/phase1_default.yaml

# Resume from checkpoint
python scripts/train_phase1.py --config configs/phase1_offline_stub.yaml --resume ./checkpoints/latest/checkpoint_latest.pt

# Code formatting
black src/ tests/ scripts/ --line-length 100
isort src/ tests/ scripts/ --profile black
mypy src/ --ignore-missing-imports
```

## Architecture

### Model Pipeline (AdaptiveBackbone)
```
Input Images → EntropyScorer → PatchSelector → PatchAggregator → DINOv3 → Output
                (entropy maps)  (quadtree)     (embeddings)      (transformer)
```

Output structure:
- `global_feats`: (B, 768) - CLS token or mean pooled
- `local_feats`: (B, N_max, 768) - Adaptive patch tokens
- `local_coords`: (B, N_max, 2) - Patch coordinates

### Directory Structure
- `src/adaptive_dino_icd/backbone/` - DINOv3 wrapper and adaptive pipeline orchestration
- `src/adaptive_dino_icd/apt/` - Entropy scoring, patch selection, aggregation
- `src/adaptive_dino_icd/losses/` - ASL loss and metric losses
- `src/adaptive_dino_icd/data/` - DISC/NDEC datasets and BatchMixer (70/30 ratio)
- `src/adaptive_dino_icd/inference/` - D2LV matcher for copy detection
- `configs/` - YAML configs (`phase1_offline_stub.yaml` for testing, `phase1_default.yaml` for production)

### Data Formats

**DISC21 Dataset** (located at `/home/jowatson/Deep Learning/DISC21`):
- Images in subdirectories: `refs_50k_*/R*.jpg`, `dev_queries_50k_*/Q*.jpg`
- DISCDataset uses recursive glob to find all images

**NDEC Dataset** (located at `/home/jowatson/Deep Learning/NDEC`):
- Original format: `public_ground_truth_h5.csv` with `query_id,reference_id`
- Run `convert_ndec_annotations.py` to generate `pairs_converted.csv`
- Converted format columns: `img_a_path,img_b_path,label,direction,similar_pair_for_metric`
- Query images: `query_set/query_images_h5/Q*.jpg`
- Reference images: `reference_set_true match/R*.jpg`
- Negative pairs: `negative_pair/*/` subdirectories

### Configuration System

All configs are dataclass-based and loaded via `load_config()`. Key sections: `backbone`, `apt`, `loss`, `data`, `training`.

Use `offline_stub: true` in backbone config for testing without HuggingFace network access.

## Phase 1 Limitations

- Local features computed but not used in loss (Phase 2)
- RoPE coordinates computed but not injected into attention (Phase 2)
- No full retrieval infrastructure (Phase 2)
