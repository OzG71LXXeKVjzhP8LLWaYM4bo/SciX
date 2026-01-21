"""ML models for MIC prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


@dataclass
class TrainingHistory:
    """Store training history for neural networks."""

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float


class MICPredictor(nn.Module):
    """Neural network for MIC prediction."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = None, dropout: float = 0.3):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32, 16]

        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # Decreasing dropout for deeper layers
            drop_rate = dropout * (1 - i * 0.2)
            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class NeuralNetworkRegressor:
    """Wrapper class for training and inference with MICPredictor."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        lr: float = 0.001,
        weight_decay: float = 0.01,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MICPredictor(input_dim, hidden_dims, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )
        self.scaler = StandardScaler()
        self.history: Optional[TrainingHistory] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 200,
        batch_size: int = 16,
        patience: int = 20,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train the neural network with early stopping."""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_t = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            X_val_t = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_t = torch.FloatTensor(y_val).to(self.device)

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        criterion = nn.MSELoss()

        for epoch in range(epochs):
            # Training
            self.model.train()
            indices = torch.randperm(len(X_train_t))
            epoch_losses = []

            for i in range(0, len(X_train_t), batch_size):
                batch_idx = indices[i : i + batch_size]
                X_batch = X_train_t[batch_idx]
                y_batch = y_train_t[batch_idx]

                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())

            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)

            # Validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_t)
                    val_loss = criterion(val_pred, y_val_t).item()
                val_losses.append(val_loss)

                self.scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                if verbose and epoch % 20 == 0:
                    print(
                        f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )
            else:
                if verbose and epoch % 20 == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.history = TrainingHistory(
            train_losses=train_losses,
            val_losses=val_losses,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
        )

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy()

        return predictions

    def save(self, path: str | Path) -> None:
        """Save model weights."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "scaler_mean": self.scaler.mean_,
                "scaler_scale": self.scaler.scale_,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.scaler.mean_ = checkpoint["scaler_mean"]
        self.scaler.scale_ = checkpoint["scaler_scale"]


class XGBoostRegressor:
    """XGBoost wrapper for MIC prediction."""

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            early_stopping_rounds=20,
        )
        self.feature_names: list[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list[str] = None,
        verbose: bool = True,
    ) -> None:
        """Train the XGBoost model."""
        self.feature_names = feature_names or []

        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        importance = self.model.feature_importances_
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        return {f"feature_{i}": v for i, v in enumerate(importance)}

    def save(self, path: str | Path) -> None:
        """Save model to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))

    def load(self, path: str | Path) -> None:
        """Load model from JSON."""
        self.model.load_model(str(path))


class RidgeRegressor:
    """Ridge regression baseline for MIC prediction."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Train the Ridge regression model."""
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def get_coefficients(self, feature_names: list[str] = None) -> dict[str, float]:
        """Get model coefficients."""
        coefs = self.model.coef_
        if feature_names:
            return dict(zip(feature_names, coefs))
        return {f"feature_{i}": v for i, v in enumerate(coefs)}


def get_model(model_type: str, input_dim: int, **kwargs) -> Any:
    """Factory function to create models."""
    if model_type == "nn":
        return NeuralNetworkRegressor(input_dim=input_dim, **kwargs)
    elif model_type == "xgboost":
        return XGBoostRegressor(**kwargs)
    elif model_type == "ridge":
        return RidgeRegressor(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models with synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 8)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1

    X_train, X_val = X[:80], X[80:]
    y_train, y_val = y[:80], y[80:]

    print("Testing Neural Network:")
    nn_model = NeuralNetworkRegressor(input_dim=8, hidden_dims=[32, 16])
    history = nn_model.fit(X_train, y_train, X_val, y_val, epochs=50, verbose=False)
    print(f"  Best epoch: {history.best_epoch}, Best val loss: {history.best_val_loss:.4f}")

    print("\nTesting XGBoost:")
    xgb_model = XGBoostRegressor(n_estimators=50)
    xgb_model.fit(X_train, y_train, X_val, y_val, verbose=False)
    pred = xgb_model.predict(X_val)
    print(f"  Val predictions range: [{pred.min():.2f}, {pred.max():.2f}]")

    print("\nTesting Ridge:")
    ridge_model = RidgeRegressor()
    ridge_model.fit(X_train, y_train)
    pred = ridge_model.predict(X_val)
    print(f"  Val predictions range: [{pred.min():.2f}, {pred.max():.2f}]")
