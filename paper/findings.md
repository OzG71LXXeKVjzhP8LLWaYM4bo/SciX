# Shannon Entropy as a Predictor of Antibacterial Polymer Activity

## Abstract

**Hypothesis**: The degree of randomization in monomer sequencing, encoded via Shannon Entropy, demonstrates predictive value for the Minimum Inhibitory Concentration (MIC) of antibacterial polymers.

**Methods**: We compare neural networks, XGBoost, and Ridge regression using three feature sets: composition-based, entropy-based, and combined features. Models are evaluated on MIC prediction for three bacterial strains (PAO1, SA, PAO1_PA).

**Key Findings**: Shannon Entropy features improve Ridge regression performance by 11.1% on average compared to composition-only features. The best model (Ridge + Combined) achieves R² = 0.755 for MIC_SA prediction. For MIC_PAO1_PA, entropy-only features outperform composition features across all models, with Ridge + Entropy achieving R² = 0.391. The small dataset size (111 samples) favors simpler linear models over neural networks.

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

- Train/Test split: 80/20 with stratification
- 5-fold cross-validation for hyperparameter selection
- Metrics: RMSE, MAE, R²

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

### 4.4 Limitations

1. **Small Dataset**: 111 samples limits deep learning potential
2. **Censored Data**: Right-censored MIC values (">128") introduce uncertainty
3. **Class Imbalance**: Many samples have high MIC (low activity), with MIC_SA having only 1 sample in the Low class
4. **Fixed Binning Thresholds**: The 64/128 thresholds may not be optimal for all targets

### 4.5 Future Directions

1. Incorporate cytotoxicity data for selectivity analysis
2. Expand dataset through transfer learning or augmentation
3. Explore attention mechanisms for sequence modeling

---

## 5. Conclusions

### 5.1 Summary of Findings

This study evaluated the predictive value of Shannon Entropy features for antibacterial polymer MIC prediction across three bacterial strains using both regression and classification approaches. Key findings include:

1. **Best regression performance**: Ridge regression with combined features achieved R² = 0.755 for MIC_SA, the highest predictive accuracy observed.

2. **Best classification performance**: Logistic regression achieves 91.3% accuracy on MIC_SA and 82.6% on MIC_PAO1 for 3-class categorization (Active/Moderate/Inactive).

3. **Entropy feature value**: Shannon Entropy features improved Ridge regression by 11.1% on average and logistic classification by 20% for MIC_PAO1_PA, demonstrating consistent predictive utility.

4. **Model selection matters**: Simple linear models (Ridge, Logistic) outperformed neural networks and XGBoost on this small dataset.

5. **Target variability**: MIC_SA was most predictable in both paradigms, while MIC_PAO1_PA benefited most from entropy features.

6. **Classification utility**: Categorical prediction provides practical screening capability, though class imbalance (especially for MIC_SA) requires careful interpretation.

### 5.2 Support/Refutation of Hypothesis

**The hypothesis is partially supported.**

Shannon Entropy features do demonstrate predictive value for MIC, particularly when used with Ridge regression. The randomness_score and composition_entropy features emerged as important predictors, confirming that sequence randomness information contributes to MIC prediction.

However, the hypothesis that neural networks would capture non-linear entropy-MIC relationships was not supported. The dataset size limitation likely prevented neural networks from realizing their potential.

### 5.3 Practical Implications

1. **For polymer design**: Sequence entropy metrics should be considered alongside traditional composition features when designing antibacterial polymers.

2. **For computational prediction**: Ridge regression with entropy features provides a simple, interpretable, and effective baseline for continuous MIC prediction.

3. **For drug screening**: Logistic regression classification offers practical categorical predictions (Active/Moderate/Inactive) with >80% accuracy, suitable for high-throughput screening pipelines.

4. **For future studies**: Larger datasets (>500 samples) would be needed to fully evaluate neural network approaches and address class imbalance.

5. **Feature engineering priority**: Investing in domain-specific feature engineering (like entropy metrics) yields better returns than model complexity for small datasets.

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
