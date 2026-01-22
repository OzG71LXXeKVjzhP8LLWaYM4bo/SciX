# SciX: Antibacterial Polymer MIC Prediction

## Installation

```bash
uv sync
```

## Usage

```bash
# Run all models with 5-fold cross-validation
uv run python main.py

# Run specific models
uv run python main.py --model nn xgboost ridge logistic

# Run with specific feature sets
uv run python main.py --features entropy combined

# Run with specific targets
uv run python main.py --target MIC_SA MIC_PAO1

# Run with different seed
uv run python main.py --seed 123

# Quiet mode
uv run python main.py --quiet
```

## Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data` | Path to dataset CSV | `Dataset final scix.xlsx - Dataset_Complete_modified.csv` |
| `--target` | Target columns | All MIC targets |
| `--model` | Model types: `nn`, `xgboost`, `ridge`, `logistic` | All models |
| `--features` | Feature sets: `composition`, `entropy`, `combined` | All sets |
| `--output` | Output directory | `outputs` |
| `--seed` | Random seed | 42 |
| `--cv` | Number of CV folds | 5 |
| `--quiet` | Suppress verbose output | False |
