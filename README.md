# ITMO: датасет новостей и агенты данных

Репозиторий курса по данным для ML: сбор и унификация новостей, контроль качества, **active learning** для отбора примеров к разметке и **авторазметка** (zero-shot + экспорт в Label Studio).

## Результаты

- **Всего записей** в `unified_news.csv`: порядка десяти тысяч (зависит от `config.yaml` и прогонов сбора)
- **Источники**: HuggingFace (Gazeta), Kaggle (несколько датасетов), RSS, HTML-скрапинг
- **Структура**: каждый источник кэшируется в своей папке под `data/raw/`

## Описание задачи

Создание унифицированного датасета новостей для ML-задач:

- Суммаризация текстов
- Классификация по категориям
- Анализ тональности
- Обучение языковых моделей

## Архитектура агентов

**DataCollectionAgent** — сбор данных:
- `load_hf_dataset` / `load_kaggle_dataset` — HuggingFace, Kaggle
- `load_rss_data` / `load_html_data` — RSS, HTML-скрапинг
- `merge(sources)` — объединение в единый датасет

**DataQualityAgent** — проверка качества:
- `detect_issues(df)` — пропуски, дубликаты, выбросы, дисбаланс классов
- `fix(df, strategy)` — устранение проблем
- `compare(df_before, df_after)` — сравнение до/после

**ActiveLearningAgent** — умный отбор разметки ([agents/al_agent.py](agents/al_agent.py)):
- `fit(labeled_df)` — TF-IDF + логистическая регрессия
- `query(pool_df, strategy, batch_size)` — стратегии `entropy`, `margin`, `random`
- `evaluate(labeled_df, test_df)` — accuracy и F1 macro на отложенной выборке
- `report(history, path)` — график качества vs. `n_labeled`
- `run_cycle(...)` — цикл: старт с 50 размеченных, 5 итераций по 20 примеров из пула
- `prepare_al_data(df, ...)` — фильтр редких категорий (`category`, минимум 10 примеров), стратифицированный test 20% и начальный пул
- `explain_selection(...)` — опционально Claude API (`ANTHROPIC_API_KEY`) для пояснения выбранных индексов

Задача по умолчанию: многоклассовая классификация **темы** (`category`) по тексту `text` на строках, где категория заполнена.

Импорт по контракту задания: `from al_agent import ActiveLearningAgent` (тонкий модуль в корне репозитория); альтернатива — `from agents.al_agent import ActiveLearningAgent`.

**AnnotationAgent** — разметка и экспорт ([agents/annotation_agent.py](agents/annotation_agent.py)):
- Zero-shot классификация (тональность, темы) через transformers
- Экспорт в формат Label Studio, спецификация задачи
- В корне: [annotation_agent.py](annotation_agent.py) для импорта `from annotation_agent import AnnotationAgent`

## Структура репозитория

```
.
├── agents/
│   ├── data_collection_agent.py
│   ├── data_quality_agent.py
│   ├── annotation_agent.py
│   └── al_agent.py
├── run.py
├── al_agent.py              # реэкспорт ActiveLearningAgent
├── annotation_agent.py      # реэкспорт AnnotationAgent
├── annotation_spec.md
├── labelstudio_import.json
├── labelstudio_review.json
├── config.yaml
├── data/
│   ├── raw/
│   │   ├── hf_IlyaGusev_gazeta/       # HuggingFace (кэш)
│   │   │   └── data.csv
│   │   ├── kaggle_<dataset>/          # Kaggle (Lenta, Russian News 2020, Large Russian News)
│   │   │   └── *.csv
│   │   ├── parsed_rss/                # RSS
│   │   │   ├── lenta/data.csv
│   │   │   ├── ria/data.csv
│   │   │   ├── tass/data.csv
│   │   │   └── kommersant/data.csv
│   │   ├── parsed_html/               # HTML-скрапинг
│   │   │   ├── lenta/data.csv
│   │   │   ├── ria/data.csv
│   │   │   └── tass/data.csv
│   │   └── unified_news.csv           # Итоговый датасет
│   └── processed/           # кривые AL, выборки после разметки (png, csv)
├── notebooks/
│   ├── eda.ipynb                 # EDA датасета
│   ├── data_quality.ipynb        # Проверка качества (Detective, Surgeon, Argument)
│   └── al_experiment.ipynb       # AL: entropy vs random, экономия разметки
├── requirements.txt
└── README.md
```

## Использование

### Установка

```bash
pip install -r requirements.txt
```

### Active Learning (воспроизводимый запуск)

После установки зависимостей и наличия `data/raw/unified_news.csv`:

```bash
python run.py
```

Скрипт вызывает `active_learning_op()`: стратифицированный сплит, цикл со стратегией `entropy`, сохранение кривой в `data/processed/learning_curve.png`.

Сравнение **entropy** и **random** и оценка «сколько примеров сэкономлено» — в [notebooks/al_experiment.ipynb](notebooks/al_experiment.ipynb).

После `pip install nbconvert` можно выполнить ноутбук из корня репозитория без открытия Jupyter:

```bash
python3 -m nbconvert --execute --to notebook --ExecutePreprocessor.timeout=600 \
  --output al_experiment_full.ipynb notebooks/al_experiment.ipynb
```

Файл `al_experiment_full.ipynb` в списке выше не коммитится (см. `.gitignore`): достаточно исходного `al_experiment.ipynb`. График сравнения стратегий по умолчанию сохраняется в `data/processed/al_entropy_vs_random.png`.

### Kaggle API

Для автоматической загрузки Kaggle-датасетов:

```bash
# Скопировать kaggle.json в ~/.kaggle/
mkdir -p ~/.kaggle
cp path/to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Запуск сбора

```python
from agents.data_collection_agent import DataCollectionAgent

agent = DataCollectionAgent(config_path='config.yaml')
df = agent.run()
agent.save(df, 'data/raw/unified_news.csv')
```

### CLI

```bash
python agents/data_collection_agent.py --config config.yaml --output data/raw/unified_news.csv
```

### Проверка качества данных

```python
from agents.data_quality_agent import DataQualityAgent
import pandas as pd

df = pd.read_csv('data/raw/unified_news.csv')
agent = DataQualityAgent()
report = agent.detect_issues(df)
df_fixed = agent.fix(df, strategy='drop_duplicates')
```

## Источники данных


| Источник                                          | Тип         | Папка                | Описание                                 |
| ------------------------------------------------- | ----------- | -------------------- | ---------------------------------------- |
| IlyaGusev/gazeta                                  | HuggingFace | hf_IlyaGusev_gazeta/ | Новости Gazeta.ru с суммаризацией        |
| yutkin/corpus-of-russian-news-articles-from-lenta | Kaggle      | kaggle_yutkin_*/     | Новости Lenta.ru                         |
| vfomenko/russian-news-2020                        | Kaggle      | kaggle_vfomenko_*/   | Русские новости 2020                     |
| vyhuholl/large-russian-news-dataset               | Kaggle      | kaggle_vyhuholl_*/   | Большой датасет русских новостей         |
| Lenta, RIA, TASS, Kommersant                      | RSS         | parsed_rss//         | RSS-ленты (feedparser)                   |
| Lenta, RIA, TASS                                  | HTML        | parsed_html//        | Скрапинг главных страниц (BeautifulSoup) |


## Схема выходных данных


| Колонка      | Тип      | Описание                                               |
| ------------ | -------- | ------------------------------------------------------ |
| title        | str      | Заголовок                                              |
| text         | str      | Текст статьи                                           |
| summary      | str      | Краткое содержание                                     |
| url          | str      | Ссылка на источник                                     |
| published_at | datetime | Дата публикации                                        |
| category     | str      | Категория                                              |
| source       | str      | Источник (hf:*, kaggle:*, parsed_rss:*, parsed_html:*) |
| collected_at | datetime | Время сбора                                            |


## Конфигурация

Параметры в `config.yaml`:

- `limit` — лимит записей на источник (HF, Kaggle)
- `limit_per_feed` — лимит на RSS-фид
- `target_size` — целевой размер итогового датасета (10000)

## Лицензия

MIT License