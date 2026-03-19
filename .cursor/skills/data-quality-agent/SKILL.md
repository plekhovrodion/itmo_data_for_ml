---
name: data-quality-agent
description: Detects and fixes data quality issues (missing values, duplicates, outliers, class imbalance). Use when cleaning data, validating datasets, fixing data quality, or working with DataQualityAgent.
---

# DataQualityAgent

Агент для выявления и устранения проблем качества данных.

## Quick Start

```python
from agents.data_quality_agent import DataQualityAgent

agent = DataQualityAgent()
report = agent.detect_issues(df)
df_clean = agent.fix(df, strategy={'missing': 'fill_unknown', 'duplicates': 'drop', 'outliers': 'clip_iqr'})
comparison = agent.compare(df, df_clean)
```

## Workflow

1. **Detect**: `report = agent.detect_issues(df)` — получить отчёт о проблемах
2. **Fix**: `df_clean = agent.fix(df, strategy={...})` — применить стратегии
3. **Compare**: `agent.compare(df, df_clean)` — сравнить метрики до/после
4. **Optional**: `agent.explain_and_recommend(report, task)` — LLM-рекомендации (ANTHROPIC_API_KEY)

## Methods

| Method | Returns |
|--------|---------|
| `detect_issues(df)` | `{missing, duplicates, outliers, imbalance}` |
| `fix(df, strategy)` | Очищенный DataFrame |
| `compare(df_before, df_after)` | DataFrame (metric, before, after, change) |
| `explain_and_recommend(report, task)` | LLM-рекомендация |

## Strategy Options

| Problem | Strategies |
|---------|------------|
| missing | `drop`, `mode`, `fill_unknown`, `median`, `mean` |
| duplicates | `drop` |
| outliers | `clip_iqr`, `clip_zscore`, `drop` |

## Outliers

Для текстовых колонок создаются `text_len` и `word_count`. IQR: Q1−1.5×IQR, Q3+1.5×IQR. Z-score: |z| > 3. Для производных колонок `clip_*` удаляет строки (текст нельзя clip).

## Example

```python
strategy = {'missing': 'fill_unknown', 'duplicates': 'drop', 'outliers': 'clip_iqr'}
df_clean = agent.fix(df.copy(), strategy=strategy)
agent.compare(df, df_clean)
```

См. также: оркестрация и точки аппрува — [data-pipeline-hitl](../data-pipeline-hitl/SKILL.md).
