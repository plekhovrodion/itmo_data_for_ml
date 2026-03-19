---
name: data-pipeline-hitl
description: >-
  Orchestrates the four project data agents (collection, quality, annotation, active learning) in order
  with mandatory human-in-the-loop approval gates before continuing. Use when running the end-to-end ML data
  pipeline, final course project, run_pipeline.py, HITL review queues, or when the user asks for a skill
  pipeline, phased workflow, or approvals between agents.
---

# Пайплайн четырёх агентов + Human-in-the-loop

Этот скилл задаёт **порядок работы** и **где обязательно остановиться** и дождаться явного решения пользователя. На каждом gate ассистент кратко резюмирует артефакты и задаёт вопрос на подтверждение; **следующую фазу не начинать**, пока пользователь не ответил (или не скорректировал план).

Кодовая точка входа репозитория: [`run_pipeline.py`](../../../run_pipeline.py), конфиг: [`pipeline_config.yaml`](../../../pipeline_config.yaml).

## Карта фаз и проектных скиллов

| Фаза | Скилл (детали API) | Основной артефакт |
|------|-------------------|-------------------|
| 1. Сбор | [data-collection-agent](../data-collection-agent/SKILL.md) | `data/raw/unified_news.csv` |
| 2. Чистка | [data-quality-agent](../data-quality-agent/SKILL.md) | `reports/quality_report.md`, `data/processed/cleaned.parquet` |
| 3. Авторазметка | [annotation-agent](../annotation-agent/SKILL.md) | `data/processed/annotated.parquet`, `reports/annotation_spec.md` |
| 4. HITL | *(человек)* | `review_queue.csv` → `review_queue_corrected.csv` |
| 5. Active Learning | [active-learning-agent](../active-learning-agent/SKILL.md) | `reports/al_report.md`, `data/processed/learning_curve.png` |
| 6. Обучение + итог | код в `run_pipeline.py` | `models/sentiment_tfidf_logreg.joblib`, `reports/model_metrics.json`, `data/labeled/final.parquet` |

---

## Gate 1 — После сбора (опциональный аппрув)

**Когда:** есть `unified_news.csv` (или эквивалент после `DataCollectionAgent.save`).

**Показать пользователю:** число строк, список типов источников из лога/`collection_stats`, путь к файлу.

**Вопрос на аппрув:**  
«Данные собраны. Устраивает объём и источники? Продолжить чистку (DataQualityAgent)?»

Если пользователь просит изменить источники — правки только в `config.yaml` и повтор фазы 1.

---

## Gate 2 — После чистки (рекомендуемый аппрув)

**Когда:** записан `reports/quality_report.md`, применён `fix()` согласно `pipeline_config.yaml` → `cleaned.parquet`.

**Показать пользователю:** краткое резюме `detect_issues` (пропуски, дубли, выбросы, дисбаланс) и таблица `compare` (строки до/после).

**Вопрос на аппрув:**  
«Стратегия `quality.strategy` в `pipeline_config.yaml` приемлема? Подтвердить и перейти к авторазметке?»

Если нет — скорректировать `strategy`, перезапустить фазу 2 (не идти к AnnotationAgent).

---

## Gate 3 — Перед тяжёлой авторазметкой (рекомендуемый аппрув)

**Когда:** перед запуском zero-shot на всём датафрейме.

**Показать пользователю:** задачу (`sentiment_classification` / `topic_news`), порог `confidence_threshold`, опционально `annotation.max_rows` для отладки.

**Вопрос на аппрув:**  
«Запуск zero-shot на N строк может занять долго и нагрузить RAM. Подтвердить параметры или сначала прогон на подвыборке?»

---

## Gate 4 — HITL после авторазметки (обязательный аппрув)

**Когда:** создан `review_queue.csv` с колонкой `corrected_label` для строк с низкой уверенностью.

**Показать пользователю:** сколько строк в очереди, путь к файлам, допустимые значения меток (`positive` / `negative` / `neutral` для тональности).

**Стоп:** не запускать `merge-hitl`, AL и train, пока пользователь явно не сообщил, что **`review_queue_corrected.csv` готов** (или что очередь пуста и продолжать можно).

**Вопрос на аппрув:**  
«Очередь на ручную проверку: M строк. Заполните `corrected_label` и сохраните как `review_queue_corrected.csv`. Напишите «готово», когда можно продолжить с `--from-step merge-hitl`.»

---

## Gate 5 — После Active Learning (опциональный аппрув)

**Когда:** есть `al_history.json` / `al_report.md` и кривая обучения.

**Показать пользователю:** финальный `n_labeled`, последние accuracy / F1 на тесте; напомнить, что в коде пул симулирует «оракул» (метки уже в данных).

**Вопрос на аппрув:**  
«Результаты AL устраивают? Переходим к финальному обучению и сохранению модели?»

---

## Gate 6 — Перед сдачей (опциональный аппрув)

**Когда:** есть `model_metrics.json`, `data/labeled/final.parquet`.

**Показать пользователю:** accuracy, F1 macro, путь к `joblib`.

**Вопрос на аппрув:**  
«Метрики и итоговый датасет готовы к фиксации в отчёте (README, 5 разделов)?»

---

## Правила для ассистента (Claude Code)

1. **Один gate — одно явное решение пользователя.** Формулировки вроде «продолжай», «да», «gate 4 готов» считаются подтверждением.
2. **Не смешивать фазы:** не предлагать править AL, пока не закрыт HITL (Gate 4), если пайплайн идёт через `run_pipeline.py`.
3. **При ошибках окружения** (`transformers`, OOM, обрыв лога) — сначала диагностика, затем предложить `pipeline_config.fast.yaml` или `annotation.max_rows`, не «тихо» пропускать Gate 3.
4. **Команды для пользователя** (копипаст):
   - полный прогон: `python run_pipeline.py` или `python run_pipeline.py --skip-collect`
   - после HITL: `python run_pipeline.py --from-step merge-hitl --skip-collect`
5. Четыре узких скилла читать по ссылкам в таблице выше; этот скилл отвечает только за **порядок и аппрувы**.

## Быстрая блок-схема

```text
collect → [Gate1] → clean → [Gate2] → annotate → [Gate3] → review_queue → [Gate4 обяз.] →
merge → AL → [Gate5] → train → finalize → [Gate6]
```
