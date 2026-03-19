"""
Точка входа для воспроизводимого пайплайна: active learning на unified_news.
Запуск: python run.py
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd

from agents.al_agent import ActiveLearningAgent, prepare_al_data

DEFAULT_CSV = os.path.join("data", "raw", "unified_news.csv")
DEFAULT_CURVE = os.path.join("data", "processed", "learning_curve.png")


def active_learning_op(
    csv_path: str = DEFAULT_CSV,
    output_curve_path: str = DEFAULT_CURVE,
    *,
    strategy: str = "entropy",
    n_iterations: int = 5,
    batch_size: int = 20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Операция финального пайплайна: подготовка данных, AL-цикл, сохранение learning_curve.png.

    Returns:
        dict с полями history, curve_path, n_labeled_final
    """
    df = pd.read_csv(csv_path)
    labeled_df, pool_df, test_df = prepare_al_data(
        df,
        text_col="text",
        label_col="category",
        min_class_count=10,
        test_size=0.2,
        initial_labeled=50,
        random_state=random_state,
    )
    agent = ActiveLearningAgent(model="logreg", random_state=random_state)
    history: List[Dict[str, Any]] = agent.run_cycle(
        labeled_df=labeled_df,
        pool_df=pool_df,
        test_df=test_df,
        strategy=strategy,  # type: ignore[arg-type]
        n_iterations=n_iterations,
        batch_size=batch_size,
    )
    curve_path = agent.report(history, path=output_curve_path)
    return {
        "history": history,
        "curve_path": curve_path,
        "n_labeled_final": history[-1]["n_labeled"] if history else 0,
    }


def main() -> None:
    out = active_learning_op()
    print("Active learning завершён.")
    print("Кривая:", out["curve_path"])
    print("Финальный размер размеченного пула:", out["n_labeled_final"])
    for row in out["history"]:
        print(row)


if __name__ == "__main__":
    main()
