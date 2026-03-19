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

Скрипт **`run_pipeline.py`** строит сквозной пайплайн на **тональности** (`sentiment_classification`): после авторазметки и HITL целевая колонка — `label_final` (см. раздел «Финальный пайплайн» ниже).

Отдельно скрипт **`run.py`** оставлен для быстрого эксперимента: многоклассовая классификация **темы** (`category`) по `text` на уже готовом `unified_news.csv`.

Импорт по контракту задания: `from al_agent import ActiveLearningAgent` (тонкий модуль в корне репозитория); альтернатива — `from agents.al_agent import ActiveLearningAgent`.

**AnnotationAgent** — разметка и экспорт ([agents/annotation_agent.py](agents/annotation_agent.py)):
- Zero-shot классификация (тональность, темы) через transformers
- Экспорт в формат Label Studio, спецификация задачи
- В корне: [annotation_agent.py](annotation_agent.py) для импорта `from annotation_agent import AnnotationAgent`

## Скиллы Cursor

В каталоге [.cursor/skills/](.cursor/skills/) лежат **Agent Skills** для Cursor: краткие сценарии и API по каждому агенту.

| Скилл | Файл | Когда подключается |
|--------|------|---------------------|
| **Пайплайн + HITL** | [data-pipeline-hitl/SKILL.md](.cursor/skills/data-pipeline-hitl/SKILL.md) | Сквозной сценарий, `run_pipeline.py`, аппрувы между этапами |
| Сбор данных | [data-collection-agent/SKILL.md](.cursor/skills/data-collection-agent/SKILL.md) | Сбор, источники, `config.yaml` |
| Качество данных | [data-quality-agent/SKILL.md](.cursor/skills/data-quality-agent/SKILL.md) | Очистка, дубликаты, выбросы |
| Active Learning | [active-learning-agent/SKILL.md](.cursor/skills/active-learning-agent/SKILL.md) | AL, `run.py`, сравнение стратегий |
| Разметка | [annotation-agent/SKILL.md](.cursor/skills/annotation-agent/SKILL.md) | Zero-shot, Label Studio |

## Структура репозитория

```
.
├── .cursor/skills/          # Agent Skills (Cursor)
├── agents/
│   ├── data_collection_agent.py
│   ├── data_quality_agent.py
│   ├── annotation_agent.py
│   └── al_agent.py
├── run.py
├── run_pipeline.py          # финальный пайплайн: 4 агента + HITL + AL + модель
├── pipeline_config.yaml     # параметры пайплайна (стратегия чистки, AL, порог HITL)
├── pipeline_config.fast.yaml # меньше строк для ускоренного прогона
├── pipeline_config.smoke.yaml # мало строк + задуман под --mock-annotation
├── pipeline_config.max20.yaml # max_rows: 20 + согласованные малые параметры AL
├── al_agent.py              # реэкспорт ActiveLearningAgent
├── annotation_agent.py      # реэкспорт AnnotationAgent
├── review_queue.csv         # HITL: очередь на проверку (генерируется пайплайном)
├── review_labels_corrected.csv # HITL: узкая таблица row_id + метки (предпочтительно для Excel)
├── review_queue_corrected.csv # HITL: полный CSV с text (Excel часто портит последнюю колонку)
├── models/                  # сохранённая модель после run_pipeline (joblib)
├── reports/                 # quality_report.md, annotation_spec.md, al_*, model_metrics.json
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
│   ├── labeled/             # итог после пайплайна: final.parquet
│   └── processed/           # cleaned, annotated, merged_after_hitl, learning_curve.png
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

Нужны **PyTorch + transformers** (для `AnnotationAgent`), **pyarrow** (parquet), **joblib** (сохранение модели) — всё перечислено в `requirements.txt`.

**Воспроизводимость для сдачи финального проекта:** основная команда — **`python run_pipeline.py`** (полный пайплайн: 4 агента, HITL, AL, модель). Скрипт **`run.py`** — отдельный демо-запуск active learning по колонке `category` на уже готовом `unified_news.csv`, без авторазметки и HITL; для критерия «единый пайплайн» используйте `run_pipeline.py`.

### Финальный пайплайн (`run_pipeline.py`)

Один сценарий оркестрации на чистом Python: **сбор → чистка → zero-shot тональность → HITL → active learning → обучение TF-IDF + LogReg → артефакты**.

```bash
# Полный прогон с нуля (нужны сеть, при необходимости Kaggle API)
python run_pipeline.py

# Уже есть data/raw/unified_news.csv — пропустить сбор
python run_pipeline.py --skip-collect
```

**Human-in-the-loop (обязательная правка данных):** после авторазметки создаются **`review_queue.csv`** (полный текст) и **`review_labels_corrected.csv`** (только `row_id`, `pred_label`, `confidence`, `corrected_label`). **Удобнее править узкий файл** — Excel при сохранении широкого CSV с переносами строк в тексте часто обрезает последнюю колонку. Заполните **`corrected_label`**: `positive` / `negative` / `neutral` для каждой строки (UTF-8 CSV). Альтернатива — **`review_queue_corrected.csv`**. Затем:

```bash
python run_pipeline.py --from-step merge-hitl --skip-collect
```

Если низкоуверенных строк нет, пайплайн идёт дальше без остановки.

Параметры стратегии чистки, AL и обучения — в [`pipeline_config.yaml`](pipeline_config.yaml). Для быстрой проверки без долгой разметки можно использовать [`pipeline_config.fast.yaml`](pipeline_config.fast.yaml):

```bash
python run_pipeline.py --skip-collect --config pipeline_config.fast.yaml
```

**Проверка, что цепочка не сломана (без zero-shot, ~секунды):** синтетические метки вместо `transformers`; у всех строк **`confidence` 0.99** — при пороге 0.7 очередь HITL обычно пуста. Для отработки ручной проверки используйте реальную авторазметку без `--mock-annotation` или поднимите **`annotation.confidence_threshold`** (например 0.999).

```bash
python run_pipeline.py --skip-collect --mock-annotation --config pipeline_config.smoke.yaml
```

Реальная авторазметка без `--mock-annotation` на десятках тысяч строк на CPU может занимать **часы**; в консоли печатается прогресс (частота зависит от объёма). Лимит строк: `annotation.max_rows` в yaml или флаг **`--annotate-max-rows N`**. Если `N` маленький, снизьте **`al.min_class_count`** и **`initial_labeled`** (см. готовый профиль **`pipeline_config.max20.yaml`**), иначе `prepare_al_data` упадёт: после фильтра редких классов не останется строк.

**Артефакты:**

| Путь | Содержимое |
|------|------------|
| `data/raw/unified_news.csv` | сырой объединённый датасет |
| `data/processed/cleaned.parquet` | после `DataQualityAgent.fix` |
| `data/processed/annotated.parquet` | + `pred_label`, `confidence`, `row_id` |
| `data/processed/merged_after_hitl.parquet` | + `label_final` |
| `data/labeled/final.parquet` | подмножество строк после AL (размер = финальный labeled pool) |
| `reports/quality_report.md` | `detect_issues`, `compare`, опционально Claude (`ANTHROPIC_API_KEY`) |
| `reports/annotation_spec.md`, `annotation_report.md` | спецификация и метрики авторазметки |
| `reports/al_report.md`, `al_history.json`, `data/processed/learning_curve.png` | AL |
| `reports/model_metrics.json` | accuracy и F1 macro на отложенном тесте |
| `models/sentiment_tfidf_logreg.joblib` | векторизатор + классификатор |

**Data card (итоговый размеченный датасет, `data/labeled/final.*`):**

- **Модальность:** текст (русскоязычные новости).
- **Задача:** трёхклассовая **тональность** (`label_final`: positive, negative, neutral).
- **Источники:** см. `config.yaml` и таблицу «Источники данных» ниже; итог сбора — `unified_news.csv`.
- **Разметка:** zero-shot (`MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) + слияние с ручными правками из HITL для низкой уверенности.
- **Объём и классы:** после прогона уточните по `final.parquet` (число строк, `value_counts` по `label_final`); изначальный объём сырья задаётся `target_size` в `config.yaml`.
- **Ограничения:** в шаге AL метки в пуле известны симулятору (как прокси стоимости разметки); в проде здесь был бы второй раунд ручной разметки отобранных индексов.

**Бонус (+3):** при заданном `ANTHROPIC_API_KEY` в отчёт качества добавляется `explain_and_recommend`; в `al_report.md` — `explain_selection` для первого батча.

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

Параметры **`run_pipeline.py`** — в `pipeline_config.yaml` (стратегия `DataQualityAgent`, порог HITL, гиперпараметры AL и LogReg).

## Финальный отчёт (5 разделов для сдачи)

Заполните по результатам своего прогона (числа — из `reports/*`, `final.parquet`, ноутбуков).

1. **Задача и датасет** — модальность (текст), ориентировочный объём после сбора и после чистки, классы тональности, откуда взяты данные (2+ источника через `DataCollectionAgent`).
2. **Роль агентов** — какие решения приняты: источники и `target_size` (сбор), стратегия `fix` и что удалили/сохранили (качество), задача и модель zero-shot (разметка), стратегия AL и размеры пула (отбор), архитектура классификатора (обучение).
3. **HITL** — порог уверенности; сколько строк в `review_queue.csv`; сколько меток исправлено и типичные ошибки авторазметки; при необходимости — как подтверждали стратегию чистки по `quality_report.md`.
4. **Метрики по этапам** — кратко: что показал `detect_issues` / `compare`; средняя уверенность и распределение `pred_label` до правок; динамика accuracy/F1 в `al_history.json` и на графике; **итоговые** accuracy и F1 macro из `model_metrics.json` на отложенном тесте.
5. **Ретроспектива** — что сработало (например, entropy vs random, эффект HITL), что нет, что бы изменили (другая модель, больше ручной разметки, другой порог, убрать симуляцию оракула в AL).

## Лицензия

MIT License