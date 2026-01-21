# Shannon Entropy as a Predictor of Antibacterial Polymer Activity

## Abstract

**Hypothesis**: The degree of randomization in monomer sequencing, encoded via Shannon Entropy, demonstrates predictive value for the Minimum Inhibitory Concentration (MIC) of antibacterial polymers.

**Methods**: We compare neural networks, XGBoost, Ridge regression, and Logistic regression using three feature sets: composition-based, entropy-based, and combined features. Models are evaluated on MIC prediction for three bacterial strains (PAO1, SA, PAO1_PA) using robust 5-fold stratified cross-validation with 5 random seeds.

**Key Findings**: Shannon entropy features improve classification accuracy by +4.5% to +6.5% for Pseudomonas targets (MIC_PAO1: 74.4%→80.9%, MIC_PAO1_PA: 75.5%→80.0%). MIC_SA achieves the highest predictability with 93.9% classification accuracy and R²=0.72 for regression. Entropy has minimal impact on regression performance (+0.018 R²). Classification provides more robust predictions than regression for this small dataset (111 samples).

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
- **Samples**: 111 unique polymers
- **Features**: Structural parameters, compositions, molecular weights, and MIC values
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

| Set | Name | Features | Count |
|-----|------|----------|-------|
| A | Composition | Structural + composition fractions + MW | 11 |
| B | Entropy | Structural + entropy metrics + MW | 11 |
| C | Combined | All features | 15 |

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

**Ridge Regression**:
- L2 regularization
- StandardScaler preprocessing

**Logistic Regression**:
- 3-class classification (Low/Medium/High MIC)
- StandardScaler preprocessing

### 2.6 Evaluation Protocol

- **Robust Cross-Validation**: 5-fold stratified CV with 5 random seeds (25 total evaluations per configuration)
- **Regression Metrics**: RMSE, MAE, R²
- **Classification Metrics**: Accuracy, F1 macro, F1 weighted
- **MIC Classes**: Low (≤64), Medium (64-128), High (>128)

---

## 3. Results

### 3.1 Classification Performance

| Target | Features | Accuracy | Std | F1 Macro |
|--------|----------|----------|-----|----------|
| **MIC_SA** | composition | **93.9%** | 1.2% | 0.87 |
| MIC_SA | entropy | 93.7% | 0.5% | 0.84 |
| MIC_SA | combined | 93.7% | 0.6% | 0.85 |
| **MIC_PAO1** | **entropy** | **80.9%** | 2.0% | 0.63 |
| MIC_PAO1 | combined | 79.5% | 3.7% | 0.59 |
| MIC_PAO1 | composition | 74.4% | 1.2% | 0.52 |
| **MIC_PAO1_PA** | **entropy** | **80.0%** | 2.1% | 0.61 |
| MIC_PAO1_PA | combined | 78.9% | 1.2% | 0.60 |
| MIC_PAO1_PA | composition | 75.5% | 2.0% | 0.54 |

### 3.2 Regression Performance

| Target | Model | Features | R² | Std |
|--------|-------|----------|-----|-----|
| **MIC_SA** | Ridge | combined | **0.724** | 0.017 |
| MIC_SA | XGBoost | combined | 0.707 | 0.043 |
| MIC_SA | Ridge | entropy | 0.702 | 0.013 |
| MIC_SA | Ridge | composition | 0.684 | 0.022 |
| MIC_PAO1_PA | XGBoost | entropy | 0.183 | 0.079 |
| MIC_PAO1_PA | XGBoost | combined | 0.131 | 0.045 |
| MIC_PAO1 | XGBoost | combined | 0.118 | 0.099 |
| MIC_PAO1 | XGBoost | entropy | 0.086 | 0.083 |

### 3.3 Shannon Entropy Impact

#### Classification

| Target | Composition | Entropy | Improvement |
|--------|-------------|---------|-------------|
| MIC_PAO1 | 74.4% | 80.9% | **+6.5%** |
| MIC_PAO1_PA | 75.5% | 80.0% | **+4.5%** |
| MIC_SA | 93.9% | 93.7% | -0.2% |

#### Regression (MIC_SA only)

| Features | R² |
|----------|-----|
| Composition | 0.684 |
| Entropy | 0.702 |
| **Improvement** | **+0.018** |

### 3.4 Model Comparison

**Best Models by Target**:

| Target | Task | Best Model | Performance |
|--------|------|------------|-------------|
| MIC_SA | Regression | Ridge + combined | R² = 0.724 ± 0.017 |
| MIC_SA | Classification | Logistic + composition | 93.9% ± 1.2% |
| MIC_PAO1 | Classification | Logistic + entropy | 80.9% ± 2.0% |
| MIC_PAO1_PA | Classification | Logistic + entropy | 80.0% ± 2.1% |

**Key Observations**:
- Ridge and XGBoost outperform neural networks on this small dataset
- Classification provides more stable predictions than regression
- MIC_SA is the most predictable target for both tasks

---

## 4. Discussion

### 4.1 Shannon Entropy Improves Classification for Pseudomonas Targets

Entropy features provide consistent improvement for classification on PAO1 targets:
- **MIC_PAO1**: +6.5% accuracy improvement (74.4% → 80.9%)
- **MIC_PAO1_PA**: +4.5% accuracy improvement (75.5% → 80.0%)

This suggests that sequence randomness information helps distinguish active from inactive polymers against Pseudomonas aeruginosa strains.

### 4.2 MIC_SA is Highly Predictable

MIC_SA shows the strongest predictability:
- **Classification**: 93.9% accuracy with low variance (±1.2%)
- **Regression**: R² = 0.724 with low variance (±0.017)

Entropy features do not improve MIC_SA predictions, likely because composition features already capture the relevant information for Staphylococcus aureus activity.

### 4.3 Regression vs Classification

Classification outperforms regression for practical utility:
- All targets achieve 80-94% classification accuracy
- Regression only works well for MIC_SA (R² = 0.72)
- MIC_PAO1 and MIC_PAO1_PA regression shows weak predictability (R² = 0.1-0.2)

For drug screening applications, classification into Active/Moderate/Inactive categories is sufficient and more reliable.

### 4.4 Model Selection

Simple linear models (Ridge, Logistic) perform comparably or better than complex models (NN, XGBoost) on this dataset:
- Small sample size (111) limits deep learning potential
- Ridge regression achieves the best regression performance
- Logistic regression provides stable classification across all targets

### 4.5 Limitations

1. **Small Dataset**: 111 samples limits model complexity and generalization
2. **Class Imbalance**: MIC_SA has only 1 sample in the Low class
3. **Target Heterogeneity**: Different bacterial strains show different predictability
4. **Censored Data**: Right-censored MIC values (">128") introduce uncertainty

---

## 5. Conclusions

### 5.1 Summary of Findings

1. **Shannon entropy improves classification for Pseudomonas targets**: +6.5% for MIC_PAO1 and +4.5% for MIC_PAO1_PA

2. **MIC_SA is the most predictable target**: 93.9% classification accuracy and R² = 0.72 for regression

3. **Classification is more robust than regression**: All targets achieve 80-94% classification accuracy

4. **Entropy has minimal impact on regression**: Only +0.018 R² improvement for MIC_SA

5. **Simple models perform best**: Ridge and Logistic regression outperform neural networks

### 5.2 Hypothesis Evaluation

**The hypothesis is supported for classification on Pseudomonas targets.**

Shannon Entropy features demonstrate consistent predictive value for MIC classification:
- +4.5% to +6.5% accuracy improvement for PAO1 targets
- Robust across 5 random seeds

Entropy features do not significantly improve regression or MIC_SA classification (already saturated at 94%).

### 5.3 Practical Recommendations

1. **For MIC_SA prediction**: Use Ridge regression with combined features (R² = 0.72)

2. **For MIC_PAO1/PAO1_PA prediction**: Use Logistic classification with entropy features (~80% accuracy)

3. **For drug screening**: Classification at 80-94% accuracy enables effective high-throughput filtering

4. **Feature recommendation**: Include entropy features for Pseudomonas targets; they provide consistent +4-6% improvement

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

### C. Class Distribution

| Target | Low (≤64) | Medium (64-128) | High (>128) |
|--------|-----------|-----------------|-------------|
| MIC_PAO1 | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |
| MIC_SA | 1 (0.9%) | 83 (74.8%) | 27 (24.3%) |
| MIC_PAO1_PA | 21 (18.9%) | 75 (67.6%) | 15 (13.5%) |

---

## References

*[To be added]*
