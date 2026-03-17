---
name: data-collection-agent
description: Collects and unifies news and media from multiple sources (HuggingFace, Kaggle, RSS, HTML). Use when gathering news data, building unified datasets, or working with NewsCollectionAgent.
---

# DataCollectionAgent

Агент для сбора и унификации новостей и медиа из различных источников.

## Quick Start

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config_path='config.yaml')
df = agent.run()
agent.save(df, 'data/raw/unified_news.csv')
```

## Sources

| Type | Method | Cache |
|------|--------|-------|
| HuggingFace | `_load_hf_dataset` | `data/raw/hf_<name>/data.csv` |
| Kaggle | `_load_kaggle_dataset` | `data/raw/kaggle_<name>/` |
| RSS | `_load_rss_data` | `data/raw/parsed_rss/<feed>/` |
| HTML | `_load_html_data` | `data/raw/parsed_html/<site>/` |

## Output Schema

| Column | Type | Description |
|--------|------|--------------|
| title | str | Заголовок |
| text | str | Текст статьи |
| summary | str | Краткое содержание |
| url | str | Ссылка |
| published_at | datetime | Дата публикации |
| category | str | Категория |
| source | str | Источник (hf:*, kaggle:*, parsed_rss:*, parsed_html:*) |
| collected_at | datetime | Время сбора |

## Config (config.yaml)

- `limit` — лимит записей на источник (HF, Kaggle)
- `limit_per_feed` — лимит на RSS-фид
- `target_size` — целевой размер итогового датасета

## CLI

```bash
python agents/data_collection_agent.py --config config.yaml --output data/raw/unified_news.csv
```

## Kaggle

Kaggle API требует `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
