# Shannon Entropy as a Predictor of Antibacterial Polymer Activity

## Abstract

**Hypothesis**: The degree of randomization in monomer sequencing, encoded via Shannon Entropy, demonstrates predictive value for the Minimum Inhibitory Concentration (MIC) of antibacterial polymers.

**Methods**: We compare Neural Networks, XGBoost, Ridge regression, and Logistic regression using three feature sets: composition-based, entropy-based, and combined features. Models are evaluated on MIC prediction for three bacterial strains (PAO1, SA, PAO1_PA) using 5-fold stratified cross-validation.

**Key Findings**: Shannon entropy features improve classification accuracy by +4.5% to +5.4% for Pseudomonas targets (MIC_PAO1: 74.8%→80.2%, MIC_PAO1_PA: 73.9%→78.4%). MIC_SA achieves the highest predictability with 95.5% classification accuracy and R²=0.74 for regression. Entropy does not improve regression performance. Classification provides more robust predictions than regression for this small dataset (111 samples). Neural networks underperform traditional ML models due to limited data.

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
- **Regression Metrics**: RMSE, MAE, R²
- **Classification Metrics**: Accuracy, F1 macro, F1 weighted
- **MIC Classes**: Low (≤64), Medium (64-128), High (>128)

---

## 3. Results

### 3.1 Classification Performance

| Target | Features | Accuracy | Std | F1 Macro |
|--------|----------|----------|-----|----------|
| **MIC_SA** | **composition** | **95.5%** | 4.1% | 0.88 |
| MIC_SA | entropy | 93.7% | 5.4% | 0.81 |
| MIC_SA | combined | 93.7% | 5.4% | 0.85 |
| **MIC_PAO1** | **entropy** | **80.2%** | 3.5% | 0.62 |
| MIC_PAO1 | combined | 78.4% | 3.4% | 0.59 |
| MIC_PAO1 | composition | 74.8% | 4.3% | 0.55 |
| **MIC_PAO1_PA** | **entropy** | **78.4%** | 5.1% | 0.57 |
| MIC_PAO1_PA | combined | 78.3% | 3.6% | 0.60 |
| MIC_PAO1_PA | composition | 73.9% | 5.0% | 0.52 |

### 3.2 Regression Performance

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

### 3.3 Shannon Entropy Impact

#### Classification

| Target | Composition | Entropy | Improvement |
|--------|-------------|---------|-------------|
| MIC_PAO1 | 74.8% | 80.2% | **+5.4%** |
| MIC_PAO1_PA | 73.9% | 78.4% | **+4.5%** |
| MIC_SA | 95.5% | 93.7% | -1.8% |

#### Regression (MIC_SA only)

| Model | Composition R² | Entropy R² | Improvement |
|-------|----------------|------------|-------------|
| XGBoost | 0.739 | 0.713 | -0.026 |
| Ridge | 0.689 | 0.684 | -0.005 |

### 3.4 Model Comparison

**Best Models by Target**:

| Target | Task | Best Model | Performance |
|--------|------|------------|-------------|
| MIC_SA | Regression | XGBoost + composition | R² = 0.739 ± 0.098 |
| MIC_SA | Classification | Logistic + composition | 95.5% ± 4.1% |
| MIC_PAO1 | Classification | Logistic + entropy | 80.2% ± 3.5% |
| MIC_PAO1_PA | Classification | Logistic + entropy | 78.4% ± 5.1% |

**Key Observations**:
- XGBoost and Ridge outperform neural networks on this small dataset
- Neural networks show R² = 0.36-0.54 vs XGBoost/Ridge R² = 0.68-0.74 for MIC_SA
- Classification provides more stable predictions than regression
- MIC_SA is the most predictable target for both tasks

---

## 4. Discussion

### 4.1 Shannon Entropy Improves Classification for Pseudomonas Targets

Entropy features provide consistent improvement for classification on PAO1 targets:
- **MIC_PAO1**: +5.4% accuracy improvement (74.8% → 80.2%)
- **MIC_PAO1_PA**: +4.5% accuracy improvement (73.9% → 78.4%)

This suggests that sequence randomness information helps distinguish active from inactive polymers against Pseudomonas aeruginosa strains.

### 4.2 MIC_SA is Highly Predictable

MIC_SA shows the strongest predictability:
- **Classification**: 95.5% accuracy (composition features)
- **Regression**: R² = 0.739 with XGBoost

Entropy features do not improve MIC_SA predictions (-1.8% for classification), likely because composition features already capture the relevant information for Staphylococcus aureus activity.

### 4.3 Regression vs Classification

Classification outperforms regression for practical utility:
- All targets achieve 78-96% classification accuracy
- Regression only works well for MIC_SA (R² = 0.74)
- MIC_PAO1 and MIC_PAO1_PA regression shows weak predictability (R² < 0.1)

For drug screening applications, classification into Active/Moderate/Inactive categories is sufficient and more reliable.

### 4.4 Model Selection

XGBoost and Ridge perform comparably and outperform neural networks on this dataset:
- Small sample size (111) limits deep learning potential
- Neural networks achieve R² = 0.36-0.54 vs XGBoost/Ridge R² = 0.68-0.74 for MIC_SA
- XGBoost achieves the best regression performance
- Logistic regression provides stable classification across all targets

### 4.5 Limitations

1. **Small Dataset**: 111 samples limits model complexity and generalization
2. **Class Imbalance**: MIC_SA has only 1 sample in the Low class
3. **Target Heterogeneity**: Different bacterial strains show different predictability
4. **Censored Data**: Right-censored MIC values (">128") introduce uncertainty

---

## 5. Conclusions

### 5.1 Summary of Findings

1. **Shannon entropy improves classification for Pseudomonas targets**: +5.4% for MIC_PAO1 and +4.5% for MIC_PAO1_PA

2. **MIC_SA is the most predictable target**: 95.5% classification accuracy and R² = 0.74 for regression

3. **Classification is more robust than regression**: All targets achieve 78-96% classification accuracy

4. **Entropy does not improve regression**: Composition features perform equally or better

5. **XGBoost and Ridge outperform neural networks**: Small dataset size limits deep learning potential

### 5.2 Hypothesis Evaluation

**The hypothesis is supported for classification on Pseudomonas targets.**

Shannon Entropy features demonstrate consistent predictive value for MIC classification:
- +4.5% to +5.4% accuracy improvement for PAO1 targets

Entropy features do not improve regression or MIC_SA classification (composition already achieves 95.5%).

### 5.3 Practical Recommendations

1. **For MIC_SA prediction**: Use XGBoost with composition features (R² = 0.74) or Logistic classification (95.5%)

2. **For MIC_PAO1/PAO1_PA prediction**: Use Logistic classification with entropy features (~78-80% accuracy)

3. **For drug screening**: Classification at 78-96% accuracy enables effective high-throughput filtering

4. **Feature recommendation**: Include entropy features for Pseudomonas targets; they provide consistent +4-5% improvement

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
