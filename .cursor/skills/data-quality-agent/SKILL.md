---
name: data-quality-agent
description: Detects and fixes data quality issues (missing values, duplicates, outliers, class imbalance). Use when cleaning data, validating datasets, or working with DataQualityAgent.
---

# DataQualityAgent

Агент-детектив для выявления и устранения проблем качества данных.

## Quick Start

```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent()
report = agent.detect_issues(df)
df_clean = agent.fix(df, strategy={'missing': 'fill_unknown', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
comparison = agent.compare(df, df_clean)
```

## Skills

| Method | Returns |
|-------|---------|
| `detect_issues(df)` | `{'missing': {...}, 'duplicates': {...}, 'outliers': {...}, 'imbalance': {...}}` |
| `fix(df, strategy)` | Очищенный DataFrame |
| `compare(df_before, df_after)` | DataFrame с метриками (metric, before, after, change) |
| `explain_and_recommend(report, task_description)` | LLM-рекомендация (требует ANTHROPIC_API_KEY) |

## Strategy Options

| Problem | Strategies |
|---------|------------|
| missing | `drop`, `mode`, `fill_unknown`, `median`, `mean` |
| duplicates | `drop` |
| outliers | `clip_iqr`, `clip_zscore`, `drop` |

## Outliers

Для текстовых данных агент создаёт производные признаки `text_len` и `word_count`. Выбросы определяются по IQR (|x| < Q1-1.5*IQR или > Q3+1.5*IQR) или z-score (|z| > 3). Для производных колонок `clip_*` удаляет строки с выбросами (текст нельзя clip'нуть).

## Example

```python
strategy_a = {'missing': 'fill_unknown', 'duplicates': 'drop', 'outliers': 'clip_iqr'}
strategy_b = {'missing': 'drop', 'duplicates': 'drop', 'outliers': 'drop'}
df_a = agent.fix(df.copy(), strategy=strategy_a)
df_b = agent.fix(df.copy(), strategy=strategy_b)
agent.compare(df, df_a)
```
