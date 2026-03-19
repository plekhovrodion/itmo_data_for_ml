"""
ActiveLearningAgent — отбор примеров для разметки (entropy, margin, random) и кривые обучения.
"""

from __future__ import annotations

from collections import Counter
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore


Strategy = Literal["entropy", "margin", "random"]


def prepare_al_data(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "category",
    min_class_count: int = 10,
    test_size: float = 0.2,
    initial_labeled: int = 50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Фильтрует редкие классы, делит на test / initial labeled / pool (стратификация).
    """
    from sklearn.model_selection import train_test_split

    work = df.dropna(subset=[label_col, text_col]).copy()
    vc = work[label_col].value_counts()
    keep = vc[vc >= min_class_count].index
    work = work[work[label_col].isin(keep)].reset_index(drop=True)
    if len(work) < initial_labeled + 10:
        raise ValueError(
            f"Слишком мало строк после фильтра (нужно > {initial_labeled + 10}), получено {len(work)}"
        )

    train_all, test_df = train_test_split(
        work,
        test_size=test_size,
        stratify=work[label_col],
        random_state=random_state,
    )
    labeled_df, pool_df = train_test_split(
        train_all,
        train_size=initial_labeled,
        stratify=train_all[label_col],
        random_state=random_state,
    )
    return labeled_df.reset_index(drop=True), pool_df.reset_index(drop=True), test_df.reset_index(drop=True)


class ActiveLearningAgent:
    """
    Скиллы: fit, query, evaluate, report, run_cycle.
    """

    def __init__(
        self,
        model: str = "logreg",
        *,
        text_col: str = "text",
        label_col: str = "category",
        random_state: int = 42,
        max_features: int = 30_000,
        ngram_range: Tuple[int, int] = (1, 2),
        logreg_C: float = 1.0,
        max_iter: int = 500,
        class_weight: Optional[str] = "balanced",
        vectorizer_min_df: Optional[int] = None,
        sublinear_tf: bool = True,
    ):
        if model != "logreg":
            raise ValueError(f"Поддерживается только model='logreg', передано: {model!r}")
        self.model = model
        self.text_col = text_col
        self.label_col = label_col
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.logreg_C = logreg_C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.vectorizer_min_df = vectorizer_min_df
        self.sublinear_tf = sublinear_tf
        self.vectorizer_: Optional[TfidfVectorizer] = None
        self.clf_: Optional[LogisticRegression] = None

    def fit(
        self,
        labeled_df: pd.DataFrame,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "ActiveLearningAgent":
        texts = labeled_df[self.text_col].astype(str).tolist()
        y = pd.Series(labeled_df[self.label_col]).astype(str).str.strip().values
        n = len(texts)
        min_df = self.vectorizer_min_df if self.vectorizer_min_df is not None else (1 if n < 800 else 2)
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=min_df,
            sublinear_tf=self.sublinear_tf,
        )
        X = self.vectorizer_.fit_transform(texts)
        self.clf_ = LogisticRegression(
            C=self.logreg_C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
            class_weight=self.class_weight,
        )
        self.clf_.fit(X, y, sample_weight=sample_weight)
        return self

    def _predict_proba_pool(self, pool_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.vectorizer_ is None or self.clf_ is None:
            raise RuntimeError("Сначала вызовите fit(labeled_df).")
        if pool_df.empty:
            return np.zeros((0, len(self.clf_.classes_))), np.array([])
        Xp = self.vectorizer_.transform(pool_df[self.text_col].astype(str).tolist())
        proba = self.clf_.predict_proba(Xp)
        return proba, pool_df.index.values

    def query(
        self,
        pool_df: pd.DataFrame,
        strategy: Strategy,
        batch_size: int,
    ) -> List[Any]:
        if pool_df.empty or batch_size <= 0:
            return []
        n = min(batch_size, len(pool_df))
        if strategy == "random":
            pos = self.rng.choice(len(pool_df), size=n, replace=False)
            return pool_df.index[pos].tolist()

        proba, idx_array = self._predict_proba_pool(pool_df)
        if proba.shape[0] == 0:
            return []

        if strategy == "entropy":
            ent = -np.sum(proba * np.log(proba + 1e-12), axis=1)
            order = np.argsort(-ent)
        elif strategy == "margin":
            if proba.shape[1] < 2:
                order = np.arange(len(proba))
            else:
                sorted_p = np.sort(proba, axis=1)
                margin = sorted_p[:, -1] - sorted_p[:, -2]
                order = np.argsort(margin)
        else:
            raise ValueError(f"Неизвестная стратегия: {strategy!r}")

        pick_pos = order[:n]
        return idx_array[pick_pos].tolist()

    def evaluate(self, labeled_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        if self.vectorizer_ is None or self.clf_ is None:
            raise RuntimeError("Сначала вызовите fit(labeled_df).")
        X_test = self.vectorizer_.transform(test_df[self.text_col].astype(str).tolist())
        y_test = pd.Series(test_df[self.label_col]).astype(str).str.strip().values
        y_pred = self.clf_.predict(X_test)
        cnt = Counter(y_test.tolist())
        maj_n = max(cnt.values()) if cnt else 0
        maj_baseline = float(maj_n / len(y_test)) if len(y_test) else 0.0
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "majority_class_baseline_accuracy": maj_baseline,
        }

    def report(
        self,
        history: Sequence[Dict[str, Any]],
        path: str = "learning_curve.png",
        *,
        second_history: Optional[Sequence[Dict[str, Any]]] = None,
        second_label: str = "random",
    ) -> str:
        if plt is None:
            raise RuntimeError("Для report установите matplotlib.")
        xs = [h["n_labeled"] for h in history]
        acc = [h["accuracy"] for h in history]
        f1s = [h["f1"] for h in history]

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(xs, acc, "o-", label="entropy" if second_history else "run")
        axes[1].plot(xs, f1s, "o-", label="entropy" if second_history else "run")
        if second_history:
            xs2 = [h["n_labeled"] for h in second_history]
            axes[0].plot(xs2, [h["accuracy"] for h in second_history], "s--", label=second_label)
            axes[1].plot(xs2, [h["f1"] for h in second_history], "s--", label=second_label)
        axes[0].set_xlabel("n_labeled")
        axes[0].set_ylabel("accuracy")
        axes[0].set_title("Accuracy vs labeled size")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("n_labeled")
        axes[1].set_ylabel("F1 macro")
        axes[1].set_title("F1 vs labeled size")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def run_cycle(
        self,
        labeled_df: pd.DataFrame,
        pool_df: pd.DataFrame,
        test_df: pd.DataFrame,
        strategy: Strategy,
        n_iterations: int,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        labeled = labeled_df.copy()
        pool = pool_df.copy()
        history: List[Dict[str, Any]] = []

        self.fit(labeled)
        m0 = self.evaluate(labeled, test_df)
        history.append({"iteration": 0, "n_labeled": int(len(labeled)), **m0})

        for it in range(1, n_iterations + 1):
            if pool.empty:
                break
            idx = self.query(pool, strategy, batch_size)
            if not idx:
                break
            to_add = pool.loc[idx]
            pool = pool.drop(index=idx)
            labeled = pd.concat([labeled, to_add], axis=0, ignore_index=True)
            self.fit(labeled)
            m = self.evaluate(labeled, test_df)
            history.append({"iteration": it, "n_labeled": int(len(labeled)), **m})

        return history

    def explain_selection(
        self,
        pool_df: pd.DataFrame,
        indices: Sequence[Any],
        strategy: Strategy,
        max_text_chars: int = 400,
    ) -> str:
        """
        LLM-скилл (бонус): кратко объясняет выбор индексов. Нужен ANTHROPIC_API_KEY.
        """
        try:
            import anthropic
        except ImportError:
            return "Установите anthropic: pip install anthropic"

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "ANTHROPIC_API_KEY не задан."

        rows = pool_df.loc[list(indices)] if len(indices) else pool_df.iloc[:0]
        snippets = []
        for i, (_, row) in enumerate(rows.iterrows()):
            t = str(row.get(self.text_col, ""))[:max_text_chars]
            snippets.append(f"[{i}] {t!r}")

        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Ты эксперт по active learning. Стратегия отбора: {strategy}.
Индексы в пуле (значения индекса датафрейма): {list(indices)}.
Фрагменты текста выбранных новостей:
{chr(10).join(snippets)}

Кратко (3–5 предложений на русском) объясни, зачем при {strategy} могли быть выбраны эти примеры для ручной разметки."""

        try:
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            return f"Ошибка API: {e}"
