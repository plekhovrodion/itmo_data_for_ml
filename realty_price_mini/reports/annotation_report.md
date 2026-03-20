# Отчёт этапа 3 (авторазметка)

- **Задача:** классификация ценового сегмента `budget` / `mid` / `premium` (терции `price_per_m2`).
- **Модель:** `RandomForestClassifier` на признаках `area_m2`, `rooms`, `geo_lat`, `geo_lon`, `source` (без `total_price_rub`, чтобы избежать утечки).
- **Колонки в `data/labeled/labeled.csv`:** `label_auto`, `confidence`, эталон `label_segment_ref` для контроля.
- **Метрики:** см. `reports/annotation_metrics.json`.
- **Label Studio:** `labelstudio_import.json` (до `ls_export_max` задач из `annotation_config.yaml`).
- **HITL:** строки с `confidence` ниже порога — в `review_queue.csv`; после правки колонки `label_human` сохраните как `review_queue_corrected.csv` и при необходимости смержите в пайплайн обучения.

Проверка импорта LS: создайте проект **Text classification** с полем текста `text` и choices `from_name=segment` → `to_name=text` (см. [Label Studio import](https://labelstud.io/guide/import.html)).
