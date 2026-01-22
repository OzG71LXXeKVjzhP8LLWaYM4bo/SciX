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
- **Method**: 5-fold stratified cross-validation
- **Metrics**: RMSE, MAE, R² (regression); Accuracy, F1 macro/weighted (classification)

---

## Results Summary

### Classification Results (Logistic Regression)

| Target | Features | Accuracy | Std | Entropy Benefit |
|--------|----------|----------|-----|-----------------|
| **MIC_SA** | **composition** | **95.5%** | 4.1% | baseline |
| MIC_SA | entropy | 93.7% | 5.4% | -1.8% |
| MIC_SA | combined | 93.7% | 5.4% | -1.8% |
| **MIC_PAO1** | **entropy** | **80.2%** | 3.5% | **+5.4%** |
| MIC_PAO1 | combined | 78.4% | 3.4% | +3.6% |
| MIC_PAO1 | composition | 74.8% | 4.3% | baseline |
| **MIC_PAO1_PA** | **entropy** | **78.4%** | 5.1% | **+4.5%** |
| MIC_PAO1_PA | combined | 78.3% | 3.6% | +4.4% |
| MIC_PAO1_PA | composition | 73.9% | 5.0% | baseline |

### Regression Results (Best Models)

| Target | Model | Features | R² | Std |
|--------|-------|----------|-----|-----|
| **MIC_SA** | XGBoost | composition | **0.739** | 0.098 |
| MIC_SA | XGBoost | combined | 0.722 | 0.164 |
| MIC_SA | XGBoost | entropy | 0.713 | 0.206 |
| MIC_SA | Ridge | combined | 0.697 | 0.134 |
| MIC_SA | Ridge | composition | 0.689 | 0.110 |
| MIC_SA | Ridge | entropy | 0.684 | 0.155 |
| MIC_SA | NN | entropy | 0.544 | 0.158 |
| MIC_PAO1 | XGBoost | entropy | 0.065 | 0.404 |
| MIC_PAO1_PA | XGBoost | combined | 0.061 | 0.341 |

---

## Key Findings

### 1. Shannon Entropy Impact

**Classification** (entropy vs composition):
- MIC_PAO1: 74.8% → 80.2% **(+5.4%)**
- MIC_PAO1_PA: 73.9% → 78.4% **(+4.5%)**
- MIC_SA: 95.5% → 93.7% (no improvement, composition better)

**Regression** (MIC_SA only):
- Entropy R² = 0.713 vs Composition R² = 0.739 (composition better)

### 2. Best Models by Target

| Target | Task | Best Model | Performance |
|--------|------|------------|-------------|
| MIC_SA | Regression | XGBoost + composition | R² = 0.739 ± 0.098 |
| MIC_SA | Classification | Logistic + composition | 95.5% ± 4.1% |
| MIC_PAO1 | Classification | Logistic + entropy | 80.2% ± 3.5% |
| MIC_PAO1_PA | Classification | Logistic + entropy | 78.4% ± 5.1% |

### 3. Model Performance Ranking

**Regression** (MIC_SA):
1. XGBoost (R² = 0.74)
2. Ridge (R² = 0.69)
3. Neural Network (R² = 0.36-0.54)

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
# Run all experiments with 5-fold CV (default)
uv run python main.py --model ridge xgboost logistic --features composition entropy combined --target MIC_PAO1 MIC_SA MIC_PAO1_PA

# Run with specific seed
uv run python main.py --model logistic ridge --features entropy --target MIC_PAO1_PA --seed 42

# Quiet mode
uv run python main.py --quiet
```

---

## Conclusions

1. **Shannon entropy improves classification for Pseudomonas targets** (+4.5-5.4%)
2. **MIC_SA is highly predictable** (95.5% classification, R²=0.74 regression)
3. **Classification is more robust than regression** for this dataset
4. **XGBoost and Ridge outperform neural networks** due to small dataset
5. **Entropy does not improve regression** (composition features perform better)
