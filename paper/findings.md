# Shannon Entropy as a Predictor of Antibacterial Polymer Activity

## Abstract

**Hypothesis**: The degree of randomization in monomer sequencing, encoded via Shannon Entropy, demonstrates predictive value for the Minimum Inhibitory Concentration (MIC) of antibacterial polymers.

**Methods**: We compare neural networks, XGBoost, Ridge regression, and Logistic regression using three feature sets: composition-based, entropy-based, and combined features. Models are evaluated on MIC prediction for three bacterial strains (PAO1, SA, PAO1_PA) using both single train/test splits and **5-fold stratified cross-validation** to obtain reliable performance estimates.

**Key Findings**: Robust cross-validation (5 seeds × 5-fold CV) reveals that only MIC_SA shows genuine regression predictive power (R² = 0.72 ± 0.02). Initial single-split results for MIC_PAO1 (R²=0.40) and MIC_PAO1_PA (R²=0.48) were highly inflated—robust CV shows R² = 0.12 and 0.18 respectively. Classification is more robust: Logistic regression achieves 80.9% ± 2.0% for MIC_PAO1, 80.0% ± 2.1% for MIC_PAO1_PA, and 93.9% ± 1.2% for MIC_SA. **Shannon entropy features improve classification accuracy by +4.5% to +6.5% for PAO1 targets**, but have minimal impact on regression (+0.018 R²).

---

## 1. Introduction

### 1.1 Background on Antibacterial Polymers

Synthetic antimicrobial polymers represent a promising class of materials for combating antibiotic-resistant bacteria. Unlike traditional antibiotics, these polymers can be designed with tunable properties that affect their bactericidal activity.

### 1.2 Current Approaches to MIC Prediction

Traditional approaches to predicting MIC focus on:
- Molecular weight and dispersity
- Lipophilicity (cLogP)
- Monomer composition ratios
- Structural topology (block vs. random)

### 1.3 Research Gap: Role of Sequence Randomness

The relationship between monomer sequence randomness and antibacterial activity remains understudied. While block copolymers and random copolymers show different activity profiles, the quantitative effect of sequence entropy has not been systematically evaluated.

### 1.4 Hypothesis

We hypothesize that Shannon Entropy-based features capturing sequence randomness provide additional predictive power for MIC prediction beyond traditional composition features.

---

## 2. Methods

### 2.1 Dataset Description

- **Source**: Experimental polymer synthesis dataset
- **Samples**: ~111 unique polymers
- **Features**: 16 columns including compositions, structural parameters, and MIC values
- **Targets**: MIC for PAO1, SA, and PAO1_PA bacterial strains

### 2.2 Data Preprocessing

1. **MIC Value Handling**:
   - Ranges ("32-64"): Geometric mean → sqrt(32 × 64) = 45.25
   - Censored (">128"): Imputed as threshold value

2. **Feature Engineering**:
   - Computed Shannon Entropy metrics from block sequences
   - Normalized composition features

### 2.3 Shannon Entropy Feature Engineering

Four entropy-based features were computed:

1. **Composition Entropy**: H = -Σ(pᵢ × log₂(pᵢ)) for monomer fractions
2. **Block Entropy**: Entropy over block size distribution
3. **Sequence Entropy**: Combined metric accounting for intra-block mixing
4. **Randomness Score**: Weighted combination based on block count

### 2.4 Feature Sets

| Set | Name | Features |
|-----|------|----------|
| A | Composition | Structural + composition fractions |
| B | Entropy | Structural + entropy metrics |
| C | Combined | All features |

### 2.5 Model Architectures

**Neural Network**:
- Architecture: 64 → 32 → 16 → 1 (with ReLU, Dropout)
- Optimizer: Adam with weight decay
- Early stopping: Patience = 30 epochs

**XGBoost**:
- n_estimators: 200
- max_depth: 5
- learning_rate: 0.05
- Regularization: L1 + L2

**Ridge Regression (Baseline)**:
- L2 regularization
- StandardScaler preprocessing

### 2.6 Evaluation Protocol

**Initial Evaluation**:
- Train/Test split: 80/20 with stratification (~88 train, ~23 test samples)
- Metrics: RMSE, MAE, R² (regression); Accuracy, F1 macro/weighted (classification)

**Cross-Validation** (Added after observing high variance):
- 5-fold stratified cross-validation
- For regression: stratification on binned target values
- Reports mean ± standard deviation across folds
- Reveals true generalization performance and result stability

---

## 3. Results

### 3.1 Model Performance Comparison

#### MIC_PAO1 (Pseudomonas aeruginosa)

| Model | Feature Set | RMSE | MAE | R² |
|-------|-------------|------|-----|-----|
| Ridge | Combined | 41.50 | 25.96 | **0.338** |
| XGBoost | Composition | 43.76 | 21.66 | 0.264 |
| NN | Entropy | 43.89 | 29.15 | 0.260 |
| Ridge | Entropy | 44.05 | 25.88 | 0.254 |
| Ridge | Composition | 45.53 | 28.19 | 0.203 |
| NN | Combined | 49.69 | 34.17 | 0.051 |
| XGBoost | Entropy | 51.24 | 27.10 | -0.009 |
| NN | Composition | 52.32 | 38.29 | -0.052 |
| XGBoost | Combined | 55.88 | 20.79 | -0.200 |

#### MIC_SA (Staphylococcus aureus)

| Model | Feature Set | RMSE | MAE | R² |
|-------|-------------|------|-----|-----|
| Ridge | Combined | 29.36 | 20.46 | **0.755** |
| Ridge | Entropy | 31.37 | 20.12 | 0.720 |
| XGBoost | Combined | 35.15 | 23.47 | 0.648 |
| Ridge | Composition | 36.46 | 25.59 | 0.622 |
| XGBoost | Entropy | 38.66 | 26.25 | 0.575 |
| XGBoost | Composition | 40.35 | 25.42 | 0.537 |
| NN | Combined | 40.40 | 30.03 | 0.536 |
| NN | Composition | 43.16 | 32.07 | 0.470 |
| NN | Entropy | 56.15 | 42.77 | 0.103 |

#### MIC_PAO1_PA

| Model | Feature Set | RMSE | MAE | R² |
|-------|-------------|------|-----|-----|
| Ridge | Entropy | 39.82 | 25.44 | **0.391** |
| Ridge | Combined | 43.39 | 25.59 | 0.276 |
| NN | Entropy | 43.80 | 29.82 | 0.263 |
| XGBoost | Entropy | 47.28 | 20.97 | 0.141 |
| Ridge | Composition | 47.58 | 29.77 | 0.130 |
| NN | Composition | 48.80 | 35.38 | 0.085 |
| XGBoost | Composition | 51.30 | 25.63 | -0.012 |
| NN | Combined | 52.53 | 38.46 | -0.060 |
| XGBoost | Combined | 54.79 | 23.42 | -0.154 |

### 3.2 Feature Importance Analysis

XGBoost feature importance analysis reveals the following top predictors:

**For MIC_PAO1 (Composition features):**
- dpn (degree of polymerization) - highest importance
- composition_B1
- cLogP_predicted
- Dispersity

**For MIC_SA (Entropy features):**
- composition_entropy - highest importance
- randomness_score
- Number of blocks
- cLogP_predicted

**For MIC_PAO1_PA (Entropy features):**
- randomness_score - highest importance
- sequence_entropy
- block_entropy
- dpn

### 3.3 Learning Dynamics

Neural network training showed typical convergence patterns with early stopping triggered between epochs 30-80. Validation loss curves indicate:

- Models trained on entropy features generally converged faster
- Combined features showed higher variance in training dynamics
- Overfitting was observed in several NN experiments, particularly for MIC_PAO1

Learning curve plots are available in `outputs/` directory.

### 3.4 Entropy vs Composition Comparison

| Model | Avg RMSE (Composition) | Avg RMSE (Entropy) | Improvement |
|-------|------------------------|--------------------| ------------|
| Ridge | 43.19 | 38.41 | **11.1%** |
| NN | 48.09 | 47.95 | 0.3% |
| XGBoost | 45.14 | 45.73 | -1.3% |

**Key Finding**: Ridge regression benefits most significantly from entropy features, with an 11.1% reduction in average RMSE across all targets. This suggests that entropy features capture linear relationships with MIC that composition features miss.

### 3.5 Classification Results (Logistic Regression)

We also evaluated MIC prediction as a 3-class classification problem using logistic regression:
- **Class 0 (Low/Active)**: MIC ≤ 64
- **Class 1 (Medium/Moderate)**: 64 < MIC ≤ 128
- **Class 2 (High/Inactive)**: MIC > 128

#### Class Distribution

| Target | Low (≤64) | Medium (64-128) | High (>128) |
|--------|-----------|-----------------|-------------|
| MIC_PAO1 | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |
| MIC_SA | 1 (0.9%) | 83 (74.8%) | 27 (24.3%) |
| MIC_PAO1_PA | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |

#### Classification Performance

| Target | Feature Set | Accuracy | F1 (macro) | F1 (weighted) |
|--------|-------------|----------|------------|---------------|
| MIC_PAO1 | composition | **0.826** | **0.683** | 0.824 |
| MIC_PAO1 | entropy | 0.826 | 0.537 | 0.785 |
| MIC_PAO1 | combined | 0.826 | 0.683 | 0.824 |
| MIC_SA | composition | **0.913** | 0.620 | 0.893 |
| MIC_SA | entropy | 0.913 | 0.620 | 0.893 |
| MIC_SA | combined | 0.913 | 0.620 | 0.893 |
| MIC_PAO1_PA | composition | 0.652 | 0.294 | 0.652 |
| MIC_PAO1_PA | entropy | **0.783** | **0.527** | 0.763 |
| MIC_PAO1_PA | combined | 0.783 | 0.527 | 0.763 |

**Key Findings**:
1. High classification accuracy (82.6%-91.3%) demonstrates categorical MIC prediction is viable
2. MIC_SA shows artificially high accuracy due to extreme class imbalance (only 1 Low sample)
3. For MIC_PAO1_PA, entropy features improve accuracy by 20% relative (65.2% → 78.3%)
4. F1 macro scores (0.29-0.68) indicate difficulty distinguishing minority classes

### 3.6 Impact of Molecular Weight Features

After identifying that NMR, GPC, and Target molecular weight columns were unused despite having moderate correlations with MIC (0.27-0.30), we added them to all feature sets.

#### Updated Feature Sets

| Set | Features | Count |
|-----|----------|-------|
| Composition | blocks, dpn, Dispersity, cLogP, Target, NMR, GPC + compositions | 11 |
| Entropy | blocks, dpn, Dispersity, cLogP, Target, NMR, GPC + entropy metrics | 11 |
| Combined | All features | 15 |

#### Regression Results with MW Features

| Target | Best Model | Old R² | New R² | Improvement |
|--------|------------|--------|--------|-------------|
| MIC_PAO1 | Ridge+entropy | 0.254 | **0.402** | +58% |
| MIC_SA | Ridge+combined | 0.755 | **0.766** | +1.5% |
| MIC_PAO1_PA | Ridge+entropy | 0.391 | **0.484** | +24% |

#### Classification Results with MW Features

| Target | Best Model | Old Acc | New Acc | Improvement |
|--------|------------|---------|---------|-------------|
| MIC_PAO1 | Logistic+combined | 82.6% | **91.3%** | +8.7% |
| MIC_SA | Logistic+entropy | 91.3% | 91.3% | 0% |
| MIC_PAO1_PA | Logistic+entropy | 78.3% | **95.7%** | +17.4% |

**Key Finding**: Molecular weight features (Target, NMR, GPC) substantially improve prediction accuracy. The MIC_PAO1_PA target now achieves 95.7% classification accuracy with F1=0.886, demonstrating that polymer molecular weight is a critical predictor of antibacterial activity.

### 3.7 Cross-Validation Results (Critical Update)

Single train/test splits with only ~23 test samples showed high variance. We implemented **robust cross-validation**: 5-fold stratified CV repeated with 5 different random seeds (42, 123, 456, 789, 1000) for 25 total evaluations per configuration.

#### Single-Split vs Robust Cross-Validation Comparison

| Target | Task | Single-Split | Robust CV (5 seeds) | Status |
|--------|------|--------------|---------------------|--------|
| MIC_PAO1 | Regression | R²=0.402 | R²=0.12 ± 0.10 | **Highly Inflated** |
| MIC_SA | Regression | R²=0.766 | R²=0.72 ± 0.02 | ✓ Reliable |
| MIC_PAO1_PA | Regression | R²=0.484 | R²=0.18 ± 0.08 | **Highly Inflated** |
| MIC_PAO1 | Classification | 91.3% | 80.9% ± 2.0% | Inflated |
| MIC_SA | Classification | 91.3% | 93.9% ± 1.2% | ✓ Reliable |
| MIC_PAO1_PA | Classification | 95.7% | 80.0% ± 2.1% | **Highly Inflated** |

#### Cross-Validated Regression Results (Best Models)

| Target | Model | Features | Mean R² | Std |
|--------|-------|----------|---------|-----|
| **MIC_SA** | Ridge | combined | **0.724** | 0.017 |
| MIC_SA | XGBoost | combined | 0.707 | 0.043 |
| MIC_SA | Ridge | entropy | 0.702 | 0.013 |
| MIC_SA | Ridge | composition | 0.684 | 0.022 |
| MIC_PAO1_PA | XGBoost | entropy | 0.183 | 0.079 |
| MIC_PAO1 | XGBoost | combined | 0.118 | 0.099 |

#### Cross-Validated Classification Results (Logistic Regression)

| Target | Features | Mean Accuracy | Std | Entropy Benefit |
|--------|----------|---------------|-----|-----------------|
| MIC_SA | composition | **93.9%** | 1.2% | baseline |
| MIC_SA | entropy | 93.7% | 0.5% | -0.2% |
| **MIC_PAO1** | **entropy** | **80.9%** | 2.0% | **+6.5%** |
| MIC_PAO1 | composition | 74.4% | 1.2% | baseline |
| **MIC_PAO1_PA** | **entropy** | **80.0%** | 2.1% | **+4.5%** |
| MIC_PAO1_PA | composition | 75.5% | 2.0% | baseline |

**Key Findings from Robust Cross-Validation**:

1. **MIC_SA is the only reliable regression target** - R² = 0.72 ± 0.02 is stable across seeds and represents genuine predictive power.

2. **MIC_PAO1 and MIC_PAO1_PA regression is weak** - R² = 0.12-0.18 with high variance; these targets are difficult to predict quantitatively.

3. **Classification is robust for all targets** - Accuracy ranges from 80-94% with low variance (1-2% std).

4. **Shannon entropy improves PAO1 classification by +4-6%** - Entropy features provide consistent improvement for MIC_PAO1 (+6.5%) and MIC_PAO1_PA (+4.5%).

5. **Entropy has no impact on MIC_SA** - Already at 94% accuracy with composition features; no room for improvement.

6. **The 95.7% accuracy for MIC_PAO1_PA was a lucky split** - True robust accuracy is 80.0% ± 2.1%.

---

## 4. Discussion

### 4.1 Does Entropy Improve Predictions?

**Yes, with caveats.** The experimental results provide partial support for our hypothesis:

1. **Strong evidence for Ridge regression**: Entropy features improved Ridge regression by 11.1% on average, with the most dramatic improvement for MIC_PAO1_PA (R² improved from 0.130 to 0.391).

2. **Mixed evidence for other models**: Neural networks showed marginal improvement (0.3%), while XGBoost performed slightly worse with entropy features (-1.3%).

3. **Target-dependent effects**:
   - MIC_PAO1_PA benefited most from entropy features (best model uses entropy-only)
   - MIC_SA showed strong performance with combined features
   - MIC_PAO1 showed modest improvements

The entropy features appear to capture information about sequence randomness that correlates with antibacterial activity, particularly for linear models. The randomness_score and composition_entropy emerged as important predictors in feature importance analysis.

### 4.2 Model Comparison Insights

**Ridge regression outperformed complex models.** This counterintuitive result is explained by:

1. **Small dataset size**: With only 111 samples, neural networks and gradient boosting methods are prone to overfitting despite regularization.

2. **Feature quality over model complexity**: Well-engineered features (entropy metrics) combined with a simple linear model outperformed complex models with raw features.

3. **Linear relationships dominate**: The MIC-feature relationships appear to be largely linear, favoring Ridge regression.

**Neural networks underperformed expectations** due to:
- Insufficient data for learning complex non-linear patterns
- High variance in small-batch training
- Early stopping triggered before meaningful pattern learning

### 4.3 Classification vs Regression

The addition of logistic regression classification provides complementary insights:

1. **Practical utility**: Classification into Active/Moderate/Inactive categories may be more actionable for drug screening than exact MIC values.

2. **Consistent entropy benefit**: For MIC_PAO1_PA, entropy features improved both regression (R² 0.130→0.391) and classification (accuracy 65.2%→78.3%), reinforcing the importance of sequence randomness.

3. **Class imbalance challenge**: The extreme imbalance in MIC_SA (only 1 Low sample) suggests the binning thresholds may need adjustment for some targets, or alternative sampling strategies should be considered.

4. **F1 vs Accuracy gap**: The significant gap between accuracy and F1 macro scores (e.g., 91.3% vs 62.0% for MIC_SA) highlights that high accuracy can be misleading with imbalanced classes.

### 4.4 Single-Split vs Cross-Validation: A Cautionary Tale

The most important methodological finding from this study is the **dramatic difference between single-split and robustly cross-validated results**:

| Target | Single-Split R² | Robust CV R² (5 seeds) | Inflation Factor |
|--------|-----------------|------------------------|------------------|
| MIC_PAO1 | 0.402 | 0.12 ± 0.10 | 3× inflated |
| MIC_PAO1_PA | 0.484 | 0.18 ± 0.08 | 3× inflated |
| MIC_SA | 0.766 | 0.72 ± 0.02 | 1.06× (reliable) |

**Why did this happen?**

1. **Small test set**: With only ~23 test samples, a single "lucky" split can dramatically overestimate performance.

2. **High variance in weak signals**: MIC_PAO1 and MIC_PAO1_PA have weak underlying signals (R² = 0.12-0.18), making them susceptible to random fluctuations.

3. **Target-specific predictability**: MIC_SA has genuine learnable structure (R² = 0.72 ± 0.02 across 5 seeds), while PAO1 variants have weaker relationships.

**Implications**:

- **Always use robust CV for small datasets**: Multiple seeds reveal true variance; single-seed CV can still be misleading.
- **Classification is more stable**: Classification accuracy has ~2% std vs regression R² with ~10% std.
- **Trust but verify**: Initial 95.7% accuracy was a lucky split—true robust accuracy is 80.0% ± 2.1%.

### 4.5 Limitations

1. **Small Dataset**: 111 samples limits deep learning potential and causes high variance in cross-validation
2. **Censored Data**: Right-censored MIC values (">128") introduce uncertainty
3. **Class Imbalance**: Many samples have high MIC (low activity), with MIC_SA having only 1 sample in the Low class
4. **Fixed Binning Thresholds**: The 64/128 thresholds may not be optimal for all targets
5. **Target Heterogeneity**: Different bacterial strains show vastly different predictability, suggesting different underlying mechanisms

### 4.6 Future Directions

1. Expand dataset to improve cross-validation stability
2. Incorporate cytotoxicity data for selectivity analysis
3. Investigate why MIC_SA is predictable while PAO1 variants are not
4. Explore transfer learning from larger polymer datasets

---

## 5. Conclusions

### 5.1 Summary of Findings

This study evaluated the predictive value of Shannon Entropy features for antibacterial polymer MIC prediction across three bacterial strains. **Critical: Results are reported using robust cross-validation (5 seeds × 5-fold CV = 25 evaluations per configuration).**

Key findings include:

1. **Only MIC_SA shows reliable regression performance**: Ridge regression achieves R² = 0.72 ± 0.02 for MIC_SA. This is the only target with genuine predictive power for continuous MIC prediction.

2. **MIC_PAO1 and MIC_PAO1_PA regression is weak**: Robust CV shows R² = 0.12 ± 0.10 and R² = 0.18 ± 0.08 respectively—much weaker than single-split results suggested.

3. **Classification is robust for all targets**: Logistic regression achieves:
   - MIC_SA: **93.9% ± 1.2%** accuracy
   - MIC_PAO1: **80.9% ± 2.0%** accuracy
   - MIC_PAO1_PA: **80.0% ± 2.1%** accuracy

4. **Shannon entropy improves classification for PAO1 targets**:
   - MIC_PAO1: +6.5% (74.4% → 80.9%)
   - MIC_PAO1_PA: +4.5% (75.5% → 80.0%)
   - MIC_SA: no improvement (already at 94%)

5. **Entropy has minimal impact on regression**: For MIC_SA, entropy improves R² by only 0.018 (0.684 → 0.702).

6. **The 95.7% accuracy was a lucky split**: Robust CV shows true MIC_PAO1_PA accuracy is 80.0% ± 2.1%.

### 5.2 Support/Refutation of Hypothesis

**The hypothesis is partially supported for classification, not for regression.**

Shannon Entropy features demonstrate consistent predictive value for MIC **classification** on PAO1 targets:

- **Supported**: Entropy features improve classification accuracy by +4.5% to +6.5% for MIC_PAO1 and MIC_PAO1_PA (robust across 5 random seeds)
- **Not supported for regression**: Entropy provides only +0.018 R² improvement for MIC_SA (the only predictable target)
- **Not supported for MIC_SA classification**: Already saturated at 94% accuracy with composition features

### 5.3 Practical Implications

1. **For MIC_SA prediction**: Ridge regression with combined features provides reliable predictions (R² = 0.72 ± 0.02). This can guide polymer synthesis decisions for *S. aureus* activity.

2. **For MIC_PAO1/PAO1_PA prediction**: Use classification with entropy features. Expect ~80% accuracy for Active/Moderate/Inactive categorization.

3. **For drug screening pipelines**: Classification at 80-94% accuracy enables useful high-throughput filtering. Entropy features are recommended for PAO1 targets.

4. **For reporting ML results**: Always use robust CV (multiple seeds) for small datasets. Single-split and even single-seed CV can be misleading.

5. **Shannon entropy recommendation**: Include entropy features for classification tasks on Pseudomonas targets; they provide consistent +4-6% improvement.

---

## Appendix

### A. Entropy Calculation Details

```python
def composition_entropy(compositions):
    """Shannon entropy from monomer fractions."""
    probs = [c for c in compositions if c > 0]
    return -sum(p * log2(p) for p in probs)
```

### B. Block Sequence Parsing

Examples:
- "ABC" → Random copolymer (high entropy)
- "(A50C20)(B30)" → 2-block (medium entropy)
- "(A25C10)(B30)(A25C10)" → 3-block (lower entropy)

### C. Hyperparameter Search Space

*[To be documented]*

---

## References

*[To be added]*
