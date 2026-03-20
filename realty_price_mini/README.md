# Мини-проект: цена за м² (Kaggle + etagi)

Два источника по контракту пайплайна сбора данных:

1. **Kaggle** — [Russia Real Estate 2021](https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-2021) (`mrdaniilak/russia-real-estate-2021`).  
   Поля `price` и `area` → **`price_per_m2`** = `price / area` (руб./м²).

2. **etagi.com** — публичная выдача квартир (по умолчанию Санкт-Петербург, `spb.etagi.com/realty/flats/`).  
   Данные из встроенного JSON в HTML (`var data=...` → `lists.flats`), до **100 строк**, пагинация `/page/N/`.

## Запуск

```bash
cd realty_price_mini
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python collect.py
```

### Этап 2 — качество данных (`DataQualityAgent`)

```bash
python quality_step.py
# или: python quality_step.py --strategy strategy_conservative.yaml --out data/processed/merged_clean_conservative.csv
```

- `detect_issues` → `reports/quality_detect.json`, текстовый разбор — `reports/quality_report.md`.
- `fix` по **`strategy.yaml`** (balanced: median по числам, `drop_duplicates` по подмножеству колонок, **clip IQR** по выбросам).
- Альтернатива: **`strategy_conservative.yaml`** — удаление строк с выбросами (`drop_iqr`).
- Сравнение до/после: `reports/comparison.csv`.
- Ноутбук: `notebooks/data_quality.ipynb` (детект, два прогона стратегий, обоснование).

Итог очистки по умолчанию: `data/processed/merged_clean.csv`.

**Ключи `strategy`:** `missing.numeric` (`median` | `mean` | `drop_rows`), `duplicates.mode` (`drop`), `outliers.mode` (`clip_iqr` | `drop_iqr`), `iqr_multiplier`, `coerce_city_or_region_str`.

### Этап 3 — авторазметка (`AnnotationAgent`)

```bash
python annotation_step.py
```

- **`auto_label`:** сегмент цены `budget` / `mid` / `premium` (терции `price_per_m2`), признаки без утечки: площадь, комнаты, гео, источник.
- **`annotation_spec.md`** — инструкция для разметчиков.
- **`labelstudio_import.json`** — импорт в Label Studio (текст + предсказанный класс).
- **`review_queue.csv`** — низкая уверенность (`confidence` < порога в `annotation_config.yaml`); после правок сохраните **`review_queue_corrected.csv`** с колонкой `label_human`.
- Итог: `data/labeled/labeled.csv`, метрики: `reports/annotation_metrics.json`, отчёт: `reports/annotation_report.md`.

### Этап 4 — Active Learning (`ActiveLearningAgent`)

```bash
python al_step.py
```

- Симуляция: стартовая размеченная выборка → итерации отбора из пула; **истинные метки** пула из `label_segment_ref` (oracle).
- Сравнение **`entropy`** (неопределённость по `predict_proba`) и **`random`** на одном графике.
- Артефакты: `reports/learning_curve.png`, `reports/al_history.json`, `reports/al_report.md`.
- Параметры: `al_config.yaml` (`seed_size`, `batch_size`, `n_iterations`, …).

Артефакты (этап 1):

- `data/raw/kaggle_russia_realestate_m2.csv` — подвыборка из Kaggle (см. `max_rows` в `config.yaml`).
- `data/raw/etagi_spb_flats_m2.csv` — 100 объявлений с сайта.
- `data/raw/merged_price_m2.csv` — объединение двух таблиц.

## Конфигурация

`config.yaml`: лимиты строк, URL базы etagi, пауза между запросами (не дёргайте сайт без задержки).

## Загрузка Kaggle

Используется **`kagglehub`** (публичный датасет кэшируется в `~/.cache/kagglehub/`).  
При ошибках авторизации см. [документацию Kaggle API](https://github.com/Kaggle/kaggle-api) (переменные `KAGGLE_USERNAME` / `KAGGLE_KEY`).

## Юридическое

Скрейпинг **только в учебных объёмах**, с соблюдением robots.txt и условий сайта. Используйте полученные данные ответственно и не нарушайте ToS etagi.com.

## Схема колонок

| Колонка | Смысл |
|---------|--------|
| `price_per_m2` | Цена за м², ₽ |
| `total_price_rub` | Общая цена объекта |
| `area_m2` | Площадь |
| `city_or_region` | Город (etagi) или код региона (Kaggle) |
| `source` | `kaggle:...` или `scrape:...` |
| `listing_url` | Ссылка на карточку (etagi) или пусто (Kaggle) |
| `text`, `image`, `label` | Совместимость с контрактом `DataCollectionAgent` (описание, превью, `label` ≈ цена/м²) |
