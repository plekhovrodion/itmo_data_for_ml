---
name: active-learning-agent
description: Active learning for text classification (entropy, margin, random query strategies, learning curves, run.py active_learning_op). Use when implementing AL, comparing sampling strategies, or working with ActiveLearningAgent.
---

# ActiveLearningAgent

Отбор примеров для разметки по неопределённости модели (TF-IDF + логистическая регрессия).

## Quick Start

```python
from agents.al_agent import ActiveLearningAgent, prepare_al_data
import pandas as pd

df = pd.read_csv("data/raw/unified_news.csv")
labeled_df, pool_df, test_df = prepare_al_data(
    df, text_col="text", label_col="category",
    min_class_count=10, test_size=0.2, initial_labeled=50, random_state=42,
)
agent = ActiveLearningAgent(model="logreg", random_state=42)
history = agent.run_cycle(
    labeled_df, pool_df, test_df,
    strategy="entropy", n_iterations=5, batch_size=20,
)
agent.report(history, path="data/processed/learning_curve.png")
```

Корневой импорт по контракту: `from al_agent import ActiveLearningAgent`.

## Workflow

1. **Подготовка данных**: `prepare_al_data(df)` — фильтр редких классов, стратифицированный test и стартовые 50 размеченных.
2. **Цикл**: `run_cycle(..., test_df=...)` — на каждом шаге `fit` → метрики на `test_df` → `query` → перенос батча из пула.
3. **Пайплайн**: `python run.py` вызывает `active_learning_op()` и пишет `data/processed/learning_curve.png`.
4. **Сравнение стратегий**: ноутбук `notebooks/al_experiment.ipynb` (entropy vs random).

## Методы

| Метод | Назначение |
|--------|------------|
| `fit(labeled_df)` | Обучение векторизатора и `LogisticRegression` |
| `query(pool_df, strategy, batch_size)` | Индексы строк пула: `entropy`, `margin`, `random` |
| `evaluate(labeled_df, test_df)` | `accuracy`, `f1` (macro) |
| `report(history, path)` | Графики accuracy и F1 vs `n_labeled` |
| `explain_selection(pool_df, indices, strategy)` | Опционально Claude (`ANTHROPIC_API_KEY`) |

## Параметры по умолчанию

- Задача: классификация `category` по `text` (строки с непустой категорией).
- Старт: 50 примеров; 5 итераций × 20 примеров → 150 размеченных к концу цикла.

См. также: сквозной пайплайн с `label_final` и аппрувами — [data-pipeline-hitl](../data-pipeline-hitl/SKILL.md).
