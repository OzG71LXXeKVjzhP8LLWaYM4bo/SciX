# SciX Experiment Log

This document tracks experiments for the antibacterial polymer MIC prediction project.

---

## Experiment Configuration

### Models
- **Neural Network**: 64→32→16→1, ReLU, Dropout=0.3, Adam optimizer
- **XGBoost**: 200 estimators, max_depth=5, lr=0.05
- **Ridge Regression**: L2 regularization, alpha=1.0
- **Logistic Regression**: 3-class classification, C=1.0

### Feature Sets
| Set | Features | Count |
|-----|----------|-------|
| Composition | blocks, dpn, Dispersity, cLogP, Target, NMR, GPC + compositions | 11 |
| Entropy | blocks, dpn, Dispersity, cLogP, Target, NMR, GPC + entropy metrics | 11 |
| Combined | All features | 15 |

### Evaluation
- **Method**: 5-fold stratified cross-validation with 5 random seeds
- **Total evaluations**: 25 per configuration
- **Seeds**: 42, 123, 456, 789, 1000

---

## Results Summary

### Classification Results (Logistic Regression)

| Target | Features | Accuracy | Std | Entropy Benefit |
|--------|----------|----------|-----|-----------------|
| **MIC_SA** | composition | **93.9%** | 1.2% | baseline |
| MIC_SA | entropy | 93.7% | 0.5% | -0.2% |
| MIC_SA | combined | 93.7% | 0.6% | -0.2% |
| **MIC_PAO1** | **entropy** | **80.9%** | 2.0% | **+6.5%** |
| MIC_PAO1 | combined | 79.5% | 3.7% | +5.1% |
| MIC_PAO1 | composition | 74.4% | 1.2% | baseline |
| **MIC_PAO1_PA** | **entropy** | **80.0%** | 2.1% | **+4.5%** |
| MIC_PAO1_PA | combined | 78.9% | 1.2% | +3.4% |
| MIC_PAO1_PA | composition | 75.5% | 2.0% | baseline |

### Regression Results (Best Models)

| Target | Model | Features | R² | Std |
|--------|-------|----------|-----|-----|
| **MIC_SA** | Ridge | combined | **0.724** | 0.017 |
| MIC_SA | XGBoost | combined | 0.707 | 0.043 |
| MIC_SA | XGBoost | entropy | 0.703 | 0.038 |
| MIC_SA | Ridge | entropy | 0.702 | 0.013 |
| MIC_SA | Ridge | composition | 0.684 | 0.022 |
| MIC_PAO1_PA | XGBoost | entropy | 0.183 | 0.079 |
| MIC_PAO1_PA | XGBoost | combined | 0.131 | 0.045 |
| MIC_PAO1 | XGBoost | combined | 0.118 | 0.099 |
| MIC_PAO1 | XGBoost | entropy | 0.086 | 0.083 |

---

## Key Findings

### 1. Shannon Entropy Impact

**Classification** (entropy vs composition):
- MIC_PAO1: 74.4% → 80.9% **(+6.5%)**
- MIC_PAO1_PA: 75.5% → 80.0% **(+4.5%)**
- MIC_SA: 93.9% → 93.7% (no improvement)

**Regression** (MIC_SA only):
- Entropy R² = 0.702 vs Composition R² = 0.684 **(+0.018)**

### 2. Best Models by Target

| Target | Task | Best Model | Performance |
|--------|------|------------|-------------|
| MIC_SA | Regression | Ridge + combined | R² = 0.724 ± 0.017 |
| MIC_SA | Classification | Logistic + composition | 93.9% ± 1.2% |
| MIC_PAO1 | Classification | Logistic + entropy | 80.9% ± 2.0% |
| MIC_PAO1_PA | Classification | Logistic + entropy | 80.0% ± 2.1% |

### 3. Model Performance Ranking

**Regression** (MIC_SA):
1. Ridge (R² = 0.72)
2. XGBoost (R² = 0.71)
3. Neural Network (R² = 0.45)

**Classification** (all targets):
- Logistic regression provides stable performance across all targets
- Neural networks underperform due to small dataset size

---

## Class Distribution

| Target | Low (≤64) | Medium (64-128) | High (>128) |
|--------|-----------|-----------------|-------------|
| MIC_PAO1 | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |
| MIC_SA | 1 (0.9%) | 83 (74.8%) | 27 (24.3%) |
| MIC_PAO1_PA | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |

---

## Commands Reference

```bash
# Run all experiments with 5-fold CV
uv run python main.py --model nn xgboost ridge logistic --features composition entropy combined --target MIC_PAO1 MIC_SA MIC_PAO1_PA --cv 5

# Run with specific seed
uv run python main.py --model logistic ridge --features entropy --target MIC_PAO1_PA --cv 5 --seed 42

# Run without CV (single split)
uv run python main.py --model ridge --features entropy --target MIC_SA

# Quiet mode
uv run python main.py --cv 5 --quiet
```

---

## Conclusions

1. **Shannon entropy improves classification for Pseudomonas targets** (+4-6%)
2. **MIC_SA is highly predictable** (94% classification, R²=0.72 regression)
3. **Classification is more robust than regression** for this dataset
4. **Simple models (Ridge, Logistic) outperform complex models** (NN, XGBoost)
5. **Entropy has minimal impact on regression** (+0.018 R²)
