#!/usr/bin/env python3
"""Этап 4: Active Learning — entropy vs random, learning_curve.png, отчёт."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from agents.al_agent import ActiveLearningAgent


def main() -> None:
    root = Path(__file__).resolve().parent
    cfg_path = root / "al_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    al = cfg.get("al", {})

    df = pd.read_csv(root / "data/labeled/labeled.csv", low_memory=False)
    rs = int(al.get("random_state", 42))

    agent_e = ActiveLearningAgent(random_state=rs)
    hist_e = agent_e.run_cycle(
        df,
        y_col=al.get("target_column", "label_segment_ref"),
        test_size=float(al.get("test_size", 0.2)),
        seed_size=int(al.get("seed_size", 500)),
        strategy="entropy",
        n_iterations=int(al.get("n_iterations", 5)),
        batch_size=int(al.get("batch_size", 150)),
    )

    agent_r = ActiveLearningAgent(random_state=rs)
    hist_r = agent_r.run_cycle(
        df,
        y_col=al.get("target_column", "label_segment_ref"),
        test_size=float(al.get("test_size", 0.2)),
        seed_size=int(al.get("seed_size", 500)),
        strategy="random",
        n_iterations=int(al.get("n_iterations", 5)),
        batch_size=int(al.get("batch_size", 150)),
    )

    plot_path = agent_e.report(
        hist_e,
        hist_r,
        out_plot=root / "reports/learning_curve.png",
        out_json=root / "reports/al_history.json",
    )

    # Краткий markdown
    last_e = hist_e[-1] if hist_e else {}
    last_r = hist_r[-1] if hist_r else {}
    md = root / "reports/al_report.md"
    md.write_text(
        "\n".join(
            [
                "# Active Learning (этап 4)",
                "",
                f"- **Стратегия:** entropy vs random (см. `al_config.yaml`).",
                f"- **Последняя F1 macro (entropy):** {last_e.get('f1_macro', 'n/a')}",
                f"- **Последняя F1 macro (random):** {last_r.get('f1_macro', 'n/a')}",
                f"- **Размеченных (финал):** {last_e.get('n_labeled', 'n/a')}",
                "",
                "Симуляция: метки из пула взяты из `label_segment_ref` (oracle).",
                "На коротких циклах и одном hold-out случайный отбор иногда не хуже entropy — смотрите кривую и повторите с другим `random_state` при необходимости.",
                "",
                f"![learning curve](learning_curve.png)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Кривая: {plot_path}")
    print(f"История: {root / 'reports/al_history.json'}")
    print(f"Отчёт: {md}")


if __name__ == "__main__":
    main()
