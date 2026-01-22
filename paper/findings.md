# Shannon Entropy as a Predictor of Antibacterial Polymer Activity

## Abstract

**Hypothesis**: The degree of randomization in monomer sequencing, encoded via Shannon Entropy, demonstrates predictive value for the Minimum Inhibitory Concentration (MIC) of antibacterial polymers.

**Methods**: We compare Neural Networks, XGBoost, Ridge regression, and Logistic regression using three feature sets: composition-based, entropy-based, and combined features. Models are evaluated on MIC prediction for three bacterial strains (PAO1, SA, PAO1_PA) using 5-fold stratified cross-validation across 5 independent sessions.

**Key Findings**: Shannon entropy features improve classification accuracy by +4.5% to +6.5% for Pseudomonas targets (MIC_PAO1: 74.4%→80.9%, MIC_PAO1_PA: 75.5%→80.0%). MIC_SA achieves the highest predictability with 93.9% classification accuracy and R²=0.72 for regression. Combined features (composition + entropy) improve regression performance. Classification provides more robust predictions than regression for this small dataset (111 samples).

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

- **Cross-Validation**: 5-fold stratified CV
- **Sessions**: 5 independent runs with different random seeds
- **Regression Metrics**: RMSE, MAE, R²
- **Classification Metrics**: Accuracy, F1 macro, F1 weighted
- **MIC Classes**: Low (≤64), Medium (64-128), High (>128)

---

## 3. Results

### 3.1 Classification Performance

Results aggregated across 5 independent sessions (seeds: 42, 123, 456, 789, 1000):

| Target | Features | Accuracy | Std | F1 Macro |
|--------|----------|----------|-----|----------|
| **MIC_SA** | **composition** | **93.9%** | 4.2% | 0.85 |
| MIC_SA | entropy | 93.7% | 4.3% | 0.84 |
| MIC_SA | combined | 93.7% | 4.0% | 0.84 |
| **MIC_PAO1** | **entropy** | **80.9%** | 3.6% | 0.63 |
| MIC_PAO1 | combined | 79.5% | 4.4% | 0.61 |
| MIC_PAO1 | composition | 74.4% | 6.0% | 0.53 |
| **MIC_PAO1_PA** | **entropy** | **80.0%** | 4.9% | 0.62 |
| MIC_PAO1_PA | combined | 78.9% | 4.3% | 0.60 |
| MIC_PAO1_PA | composition | 75.5% | 5.1% | 0.55 |

### 3.2 Regression Performance

| Target | Model | Features | R² | Std | RMSE | Std |
|--------|-------|----------|-----|-----|------|-----|
| **MIC_SA** | Ridge | combined | **0.724** | 0.144 | 28.0 | 7.5 |
| MIC_SA | XGBoost | combined | 0.707 | 0.161 | 28.6 | 9.0 |
| MIC_SA | XGBoost | entropy | 0.703 | 0.149 | 29.1 | 8.1 |
| MIC_SA | Ridge | entropy | 0.702 | 0.150 | 29.2 | 7.8 |
| MIC_SA | XGBoost | composition | 0.687 | 0.158 | 29.5 | 8.5 |
| MIC_SA | Ridge | composition | 0.684 | 0.110 | 30.6 | 6.0 |
| MIC_SA | NN | combined | 0.612 | 0.177 | 33.5 | 8.1 |
| MIC_SA | NN | entropy | 0.589 | 0.198 | 34.4 | 8.5 |
| MIC_SA | NN | composition | 0.567 | 0.158 | 35.8 | 7.0 |
| MIC_PAO1_PA | XGBoost | entropy | 0.183 | 0.261 | 48.7 | 10.2 |
| MIC_PAO1 | XGBoost | combined | 0.118 | 0.209 | 50.6 | 8.6 |

### 3.3 Shannon Entropy Impact

#### Classification

| Target | Composition | Entropy | Improvement |
|--------|-------------|---------|-------------|
| MIC_PAO1 | 74.4% | 80.9% | **+6.5%** |
| MIC_PAO1_PA | 75.5% | 80.0% | **+4.5%** |
| MIC_SA | 93.9% | 93.7% | -0.2% |

#### Regression (MIC_SA)

| Model | Composition R² | Combined R² | Improvement |
|-------|----------------|-------------|-------------|
| Ridge | 0.684 | 0.724 | +0.040 |
| XGBoost | 0.687 | 0.707 | +0.020 |

### 3.4 Model Comparison

**Best Models by Target**:

| Target | Task | Best Model | Performance |
|--------|------|------------|-------------|
| MIC_SA | Regression | Ridge + combined | R² = 0.724 ± 0.144 |
| MIC_SA | Classification | Logistic + composition | 93.9% ± 4.2% |
| MIC_PAO1 | Classification | Logistic + entropy | 80.9% ± 3.6% |
| MIC_PAO1_PA | Classification | Logistic + entropy | 80.0% ± 4.9% |

**Key Observations**:
- Ridge and XGBoost outperform neural networks on this small dataset
- Neural networks show R² = 0.57-0.61 vs Ridge/XGBoost R² = 0.68-0.72 for MIC_SA
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
- **Classification**: 93.9% accuracy (composition features)
- **Regression**: R² = 0.724 with Ridge + combined features

Entropy features do not improve MIC_SA classification (-0.2%), likely because composition features already capture the relevant information for Staphylococcus aureus activity.

### 4.3 Combined Features Improve Regression

Unlike the previous single-session results, the 5-session aggregated analysis reveals that combined features (composition + entropy) improve regression performance:
- Ridge: R² = 0.684 (composition) → 0.724 (combined), +0.040
- XGBoost: R² = 0.687 (composition) → 0.707 (combined), +0.020

This indicates entropy features provide complementary information for continuous MIC prediction.

### 4.4 Regression vs Classification

Classification outperforms regression for practical utility:
- All targets achieve 75-94% classification accuracy
- Regression only works well for MIC_SA (R² = 0.72)
- MIC_PAO1 and MIC_PAO1_PA regression shows weak predictability (R² < 0.2)

For drug screening applications, classification into Active/Moderate/Inactive categories is sufficient and more reliable.

### 4.5 Model Selection

Ridge and XGBoost perform comparably and outperform neural networks on this dataset:
- Small sample size (111) limits deep learning potential
- Neural networks achieve R² = 0.57-0.61 vs Ridge/XGBoost R² = 0.68-0.72 for MIC_SA
- Ridge achieves the best regression performance with combined features
- Logistic regression provides stable classification across all targets

### 4.6 Reproducibility

Results are highly reproducible across 5 independent sessions:
- Classification accuracy: cross-session std = 1.3-2.3%
- Regression R²: cross-session std = 0.05-0.11
- Low variability confirms the robustness of findings

### 4.7 Limitations

1. **Small Dataset**: 111 samples limits model complexity and generalization
2. **Class Imbalance**: MIC_SA has only 1 sample in the Low class
3. **Target Heterogeneity**: Different bacterial strains show different predictability
4. **Censored Data**: Right-censored MIC values (">128") introduce uncertainty

---

## 5. Conclusions

### 5.1 Summary of Findings

1. **Shannon entropy improves classification for Pseudomonas targets**: +6.5% for MIC_PAO1 and +4.5% for MIC_PAO1_PA

2. **MIC_SA is the most predictable target**: 93.9% classification accuracy and R² = 0.72 for regression

3. **Classification is more robust than regression**: All targets achieve 75-94% classification accuracy

4. **Combined features improve regression**: R² = 0.72 vs 0.68 for composition alone

5. **Ridge and XGBoost outperform neural networks**: Small dataset size limits deep learning potential

6. **Results are reproducible**: Low cross-session variability confirms robustness

### 5.2 Hypothesis Evaluation

**The hypothesis is supported for classification on Pseudomonas targets.**

Shannon Entropy features demonstrate consistent predictive value for MIC classification:
- +4.5% to +6.5% accuracy improvement for PAO1 targets

Combined features (composition + entropy) also improve regression for MIC_SA (R² +0.04).

### 5.3 Practical Recommendations

1. **For MIC_SA prediction**: Use Ridge with combined features (R² = 0.72) or Logistic classification (93.9%)

2. **For MIC_PAO1/PAO1_PA prediction**: Use Logistic classification with entropy features (~80% accuracy)

3. **For drug screening**: Classification at 75-94% accuracy enables effective high-throughput filtering

4. **Feature recommendation**: Include entropy features for all targets; they improve Pseudomonas classification and SA regression

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

### D. Cross-Session Variability

| Metric | Cross-Session Std |
|--------|-------------------|
| Classification accuracy | 1.3-2.3% |
| Regression R² | 0.05-0.11 |
| Regression RMSE | 1.0-3.6 |

---

## References

*[To be added]*
