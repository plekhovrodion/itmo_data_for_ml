# NewsCollectionAgent: Новости и медиа

Агент для сбора и унификации новостей и медиа из множества источников: HuggingFace, Kaggle, RSS, HTML-парсинг.

## Результаты

- **Всего записей**: ~10 000 (целевой размер)
- **Источники**: HuggingFace (Gazeta), Kaggle (3 датасета), RSS (6 фидов), HTML (3 сайта)
- **Структура**: каждый источник сохраняется в отдельную папку с кэшированием

## Описание задачи

Создание унифицированного датасета новостей для ML-задач:

- Суммаризация текстов
- Классификация по категориям
- Анализ тональности
- Обучение языковых моделей

## Архитектура агента

```
NewsCollectionAgent/
├── load_hf_dataset(name) -> DataFrame      # HuggingFace, кэш в hf_*/
├── load_kaggle_dataset(name) -> DataFrame  # Kaggle, кэш в kaggle_*/
├── load_rss_data(feeds) -> DataFrame       # RSS, кэш в parsed_rss/<feed>/
├── load_html_data(url) -> DataFrame        # HTML-скрапинг, кэш в parsed_html/<site>/
└── merge(sources) -> DataFrame
```

## Структура репозитория

```
.
├── agents/
│   └── data_collection_agent.py
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
│   └── processed/
├── notebooks/
│   └── eda.ipynb
├── requirements.txt
└── README.md
```

## Использование

### Установка

```bash
pip install -r requirements.txt
```

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