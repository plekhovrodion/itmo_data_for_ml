---
name: data-collection-agent
description: Collects and unifies news and media from multiple sources (HuggingFace, Kaggle, RSS, HTML). Use when gathering news data, building unified datasets, adding data sources, or working with DataCollectionAgent.
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

## Workflow

1. **Run full pipeline**: `agent.run()` → `agent.save(df, path)`
2. **Add source**: Add entry to `sources` in config.yaml (see Config schema below)
3. **Run subset**: `agent.run(sources=[...])` with filtered sources list

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

```yaml
output_schema: { title, text, summary, url, source, published_at, category, collected_at }
target_size: 10000
sources:
  # HuggingFace
  - type: hf_dataset
    name: IlyaGusev/gazeta
    limit: 3000
  # Kaggle
  - type: kaggle_dataset
    name: owner/dataset-name
    limit: 2000
  # RSS
  - type: rss_parser
    feeds: [https://...]
    limit_per_feed: 400
  # HTML
  - type: html_parser
    url: https://...
    limit: 300
```

## CLI

```bash
python agents/data_collection_agent.py --config config.yaml --output data/raw/unified_news.csv
```

## Kaggle

Requires `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
cp path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
