"""ActiveLearningAgent: entropy / margin / random query, кривая обучения vs baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _prepare_xy(df: pd.DataFrame, y_col: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    d = df.dropna(subset=[y_col]).copy()
    y = d[y_col].astype(str).values
    feature_cols = [c for c in ["area_m2", "rooms", "geo_lat", "geo_lon"] if c in d.columns]
    cat_cols = ["source"] if "source" in d.columns else []
    X = d[feature_cols + cat_cols].copy()
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y, feature_cols + cat_cols


def _make_pipeline(feature_cols: list[str], cat_cols: list[str], random_state: int) -> ColumnTransformer:
    num_cols = [c for c in feature_cols if c != "source"]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if num_cols:
        transformers.append(("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols))
    if "source" in cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                ["source"],
            )
        )
    return ColumnTransformer(transformers)


class ActiveLearningAgent:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.model_: Pipeline | None = None
        self._prep_: ColumnTransformer | None = None
        self._feature_cols_: list[str] = []
        self._cat_cols_: list[str] = []

    def _build_model(self, X: pd.DataFrame) -> Pipeline:
        num_cols = [c for c in X.columns if c != "source"]
        cat_cols = ["source"] if "source" in X.columns else []
        prep = _make_pipeline(num_cols + cat_cols, cat_cols, self.random_state)
        clf = RandomForestClassifier(
            n_estimators=80,
            max_depth=14,
            random_state=self.random_state,
            class_weight="balanced",
        )
        return Pipeline([("prep", prep), ("clf", clf)])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
        self.model_ = self._build_model(X)
        self.model_.fit(X, y)
        return self.model_

    def query(self, X_pool: pd.DataFrame, strategy: str, batch_size: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Сначала fit()")
        if len(X_pool) == 0:
            return np.array([], dtype=int)
        bs = min(batch_size, len(X_pool))
        proba = self.model_.predict_proba(X_pool)
        rng = np.random.default_rng(self.random_state)

        if strategy == "random":
            idx = rng.choice(len(X_pool), size=bs, replace=False)
            return np.sort(idx)

        if strategy == "entropy":
            ent = -np.sum(proba * np.log(proba + 1e-12), axis=1)
            idx = np.argsort(-ent)[:bs]
            return np.sort(idx)

        if strategy == "margin":
            sp = np.sort(proba, axis=1)[:, ::-1]
            if sp.shape[1] < 2:
                margin = np.zeros(len(sp))
            else:
                margin = sp[:, 0] - sp[:, 1]
            idx = np.argsort(margin)[:bs]
            return np.sort(idx)

        raise ValueError(f"Неизвестная стратегия: {strategy}")

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
        if self.model_ is None:
            raise RuntimeError("Сначала fit()")
        pred = self.model_.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        }

    def run_cycle(
        self,
        df: pd.DataFrame,
        *,
        y_col: str = "label_segment_ref",
        test_size: float = 0.2,
        seed_size: int = 500,
        strategy: str = "entropy",
        n_iterations: int = 5,
        batch_size: int = 150,
    ) -> list[dict[str, Any]]:
        """Симуляция AL: метки пула берутся из y_col (oracle)."""
        X_raw, y, cols = _prepare_xy(df, y_col)
        if len(X_raw) < seed_size + batch_size + 10:
            seed_size = max(50, len(X_raw) // 10)

        idx_all = np.arange(len(X_raw))
        rng = np.random.default_rng(self.random_state)
        idx_train, idx_test = train_test_split(
            idx_all,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y,
        )
        X_test = X_raw.iloc[idx_test].reset_index(drop=True)
        y_test = y[idx_test]

        X_tr = X_raw.iloc[idx_train].reset_index(drop=True)
        y_tr = y[idx_train]

        if seed_size >= len(X_tr):
            seed_size = max(30, len(X_tr) // 4)

        idx_seed, idx_pool = train_test_split(
            np.arange(len(X_tr)),
            train_size=seed_size,
            random_state=self.random_state,
            stratify=y_tr,
        )

        labeled_idx = list(idx_seed)
        pool_idx = list(idx_pool)

        history: list[dict[str, Any]] = []

        def snapshot(it: int, phase: str) -> None:
            X_lab = X_tr.iloc[labeled_idx]
            y_lab = y_tr[labeled_idx]
            self.fit(X_lab, y_lab)
            met = self.evaluate(X_test, y_test)
            history.append(
                {
                    "iteration": it,
                    "phase": phase,
                    "n_labeled": len(labeled_idx),
                    "n_pool": len(pool_idx),
                    **met,
                }
            )

        snapshot(0, "after_seed")

        for it in range(1, n_iterations + 1):
            if len(pool_idx) == 0:
                break
            X_pool = X_tr.iloc[pool_idx]
            y_pool = y_tr[pool_idx]
            self.fit(X_tr.iloc[labeled_idx], y_tr[labeled_idx])
            local_idx = self.query(X_pool, strategy, batch_size)
            picked = [pool_idx[i] for i in local_idx]
            for p in picked:
                pool_idx.remove(p)
                labeled_idx.append(p)
            snapshot(it, strategy)

        return history

    def report(
        self,
        history_main: list[dict[str, Any]],
        history_random: list[dict[str, Any]],
        out_plot: str | Path = "reports/learning_curve.png",
        out_json: str | Path = "reports/al_history.json",
    ) -> Path:
        out_plot = Path(out_plot)
        out_json = Path(out_json)
        out_plot.parent.mkdir(parents=True, exist_ok=True)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"main": history_main, "random_baseline": history_random}, f, ensure_ascii=False, indent=2)

        plt.figure(figsize=(8, 5))
        for name, hist, style in [
            ("entropy (query)", history_main, "-"),
            ("random baseline", history_random, "--"),
        ]:
            xs = [h["n_labeled"] for h in hist]
            ys = [h["f1_macro"] for h in hist]
            plt.plot(xs, ys, style, label=name)
        plt.xlabel("Число размеченных примеров (симуляция)")
        plt.ylabel("F1 macro (hold-out test)")
        plt.title("Active Learning: информативный отбор vs случайный")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_plot, dpi=120)
        plt.close()
        return out_plot
