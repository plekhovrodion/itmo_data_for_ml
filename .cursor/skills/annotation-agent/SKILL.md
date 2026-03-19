---
name: annotation-agent
description: Auto-labeling with zero-shot NLI, annotation spec, Label Studio export, quality checks. Use when labeling news text, exporting to Label Studio, or working with AnnotationAgent.
---

# AnnotationAgent

Авторазметка текстов новостей (многоязычный zero-shot через transformers) и экспорт в Label Studio.

## Quick Start

```python
from agents.annotation_agent import AnnotationAgent
import pandas as pd

agent = AnnotationAgent(text_col="text", confidence_threshold=0.5)
df = pd.read_csv("data/raw/unified_news.csv").head(100)
out = agent.auto_label(df, task="sentiment_classification")
agent.export_to_labelstudio(out, "labelstudio_import.json")
```

Корневой импорт: `from annotation_agent import AnnotationAgent`.

## Workflow

1. **Спецификация задачи**: `agent.generate_spec(task, path="annotation_spec.md")` — классы и описание для разметчиков.
2. **Авторазметка**: `agent.auto_label(df, task=...)` — колонки `pred_label`, `confidence`.
3. **Качество**: `agent.check_quality(df, gold_col="...")` при наличии эталона; иначе эвристики по confidence.
4. **Label Studio**: `export_to_labelstudio` / `export_low_confidence_for_review` — JSON для импорта.

## Задачи (TASK_LABELS)

- `sentiment_classification` — positive / negative / neutral.
- `topic_news` — politics, economy, society, tech, sports, culture.

Модель по умолчанию: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (нужны `torch`, `transformers`).

## Label Studio

Поля в конфиге LS должны совпадать с константами в модуле: ключ данных `text`, choices `sentiment` → `text` (см. docstring в `agents/annotation_agent.py`).
