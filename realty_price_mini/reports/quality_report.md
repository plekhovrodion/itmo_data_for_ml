# Отчёт качества данных (этап 2)

Источник: `data/raw/merged_price_m2.csv` после объединения Kaggle + etagi.

## Пропуски (missing)

- Сводка по колонкам: см. `reports/quality_detect.json` → `missing.per_column`.
- Риски для ML: пропуски в `geo_*`, `image`, колонках только Kaggle (`date`, `street_id`, …) ожидаемы для строк etagi и наоборот.

## Дубликаты

- Дедупликация по подмножеству: `price_per_m2`, `area_m2`, `total_price_rub`, `source` (см. `strategy.yaml`).
- Семантика: `keep=first`, остальные копии удаляются.

## Выбросы

- Метод: **IQR** с множителем **1.5** (по умолчанию).
- Колонки: `price_per_m2`, `total_price_rub`, `area_m2`.
- Режим **balanced**: `clip_iqr` (хвосты подрезаются к границам IQR).
- Режим **conservative**: `drop_iqr` — см. `strategy_conservative.yaml` и ноутбук.

## Дисбаланс / распределение целевого признака

- `label` совпадает с ценой за м² (регрессия). В отчёте `detect_issues` — **энтропия по квантильным бинам** `label` (не классификация).

## Решение по стратегии `fix`

- **Основная для пайплайна:** `strategy.yaml` → **balanced** (median + drop duplicates + clip IQR).
- **Альтернатива:** `strategy_conservative.yaml` — агрессивнее по выбросам (удаление строк).

Сравнение метрик: `reports/comparison.csv`.
