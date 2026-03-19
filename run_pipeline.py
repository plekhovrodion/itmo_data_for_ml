"""
Финальный дата-пайплайн: сбор → чистка → авторазметка → HITL → AL → обучение → артефакты.

Запуск:
  python run_pipeline.py              # полный прогон (остановится, если нужен HITL)
  python run_pipeline.py --from-step merge-hitl   # после правки review_queue_corrected.csv
  python run_pipeline.py --skip-collect           # если уже есть data/raw/unified_news.csv
  python run_pipeline.py --skip-collect --mock-annotation  # без zero-shot: проверка цепочки за секунды

Двухпроходный HITL:
  1) Прогон создаёт review_queue.csv — заполните corrected_label и сохраните как review_queue_corrected.csv
  2) Снова: python run_pipeline.py --from-step merge-hitl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from agents.annotation_agent import AnnotationAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.data_quality_agent import DataQualityAgent
from agents.al_agent import ActiveLearningAgent, prepare_al_data

ROOT = Path(__file__).resolve().parent
DEFAULT_PIPELINE_CONFIG = ROOT / "pipeline_config.yaml"
RAW_CSV = ROOT / "data" / "raw" / "unified_news.csv"
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.parquet"
CLEANED_CSV = ROOT / "data" / "processed" / "cleaned.csv"
ANNOTATED_PATH = ROOT / "data" / "processed" / "annotated.parquet"
MERGED_PATH = ROOT / "data" / "processed" / "merged_after_hitl.parquet"
FINAL_LABELED_PATH = ROOT / "data" / "labeled" / "final.parquet"
REVIEW_QUEUE = ROOT / "review_queue.csv"
REVIEW_CORRECTED = ROOT / "review_queue_corrected.csv"
# Узкая таблица без полного text — Excel не ломает CSV; предпочтительно для HITL.
REVIEW_LABELS_CORRECTED = ROOT / "review_labels_corrected.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
QUALITY_REPORT = REPORTS_DIR / "quality_report.md"
ANNOTATION_SPEC = REPORTS_DIR / "annotation_spec.md"
ANNOTATION_REPORT = REPORTS_DIR / "annotation_report.md"
AL_REPORT = REPORTS_DIR / "al_report.md"
AL_HISTORY_JSON = REPORTS_DIR / "al_history.json"
MODEL_METRICS = REPORTS_DIR / "model_metrics.json"
LEARNING_CURVE = ROOT / "data" / "processed" / "learning_curve.png"
MODEL_BUNDLE = MODELS_DIR / "sentiment_tfidf_logreg.joblib"


def _normalized_hitl_label(val: Any) -> Optional[str]:
    """Метка из review_queue_corrected.csv: pandas может отдать float/NaN — не вызываем .lower() на нём."""
    if val is None:
        return None
    try:
        if val is not None and pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat"):
        return None
    return s


def load_pipeline_config(path: Path = DEFAULT_PIPELINE_CONFIG) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_dirs() -> None:
    for d in (
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "data" / "labeled",
        REPORTS_DIR,
        MODELS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def step_collect(cfg: Dict[str, Any], skip_collect: bool) -> None:
    if skip_collect:
        if not RAW_CSV.is_file():
            raise FileNotFoundError(
                f"--skip-collect: нет файла {RAW_CSV}. Сначала соберите данные без флага."
            )
        print(f"[collect] пропуск, используется {RAW_CSV}")
        return
    agent = DataCollectionAgent(config_path=str(ROOT / "config.yaml"))
    df = agent.run()
    agent.save(df, str(RAW_CSV))


def step_clean(cfg: Dict[str, Any]) -> pd.DataFrame:
    ensure_dirs()
    df_raw = pd.read_csv(RAW_CSV)
    q_agent = DataQualityAgent()
    report = q_agent.detect_issues(df_raw)
    strategy = (cfg.get("quality") or {}).get("strategy") or {
        "missing": "drop",
        "duplicates": "drop",
        "outliers": "drop",
    }
    df_clean = q_agent.fix(df_raw, strategy=strategy)
    cmp_df = q_agent.compare(df_raw, df_clean)

    lines = [
        "# Отчёт DataQualityAgent",
        "",
        f"Сгенерировано: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Стратегия fix()",
        "",
        f"```\n{json.dumps(strategy, ensure_ascii=False, indent=2)}\n```",
        "",
        "## detect_issues (кратко)",
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2, default=str)[:12000],
        "```",
        "",
        "## compare(df_before, df_after)",
        "",
    ]
    try:
        cmp_table = cmp_df.to_markdown(index=False)
    except Exception:
        cmp_table = cmp_df.to_string()
    lines.extend([cmp_table, ""])
    task_desc = (
        "Классификация тональности русскоязычных новостей после авторазметки и HITL."
    )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        llm_block = q_agent.explain_and_recommend(report, task_desc)
        lines.extend(["## Рекомендации (Claude API)", "", llm_block, ""])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[clean] отчёт: {QUALITY_REPORT}")

    try:
        df_clean.to_parquet(CLEANED_PATH, index=False)
        print(f"[clean] сохранено: {CLEANED_PATH}")
    except Exception as e:
        print(f"[clean] parquet недоступен ({e}), пишем csv")
        df_clean.to_csv(CLEANED_CSV, index=False)
        print(f"[clean] сохранено: {CLEANED_CSV}")
    return df_clean


def _load_cleaned_df() -> pd.DataFrame:
    if CLEANED_PATH.is_file():
        return pd.read_parquet(CLEANED_PATH)
    if CLEANED_CSV.is_file():
        return pd.read_csv(CLEANED_CSV)
    raise FileNotFoundError("Нет cleaned.parquet/csv — выполните этап clean.")


def step_annotate(
    cfg: Dict[str, Any],
    *,
    mock_annotation: bool = False,
    annotate_max_rows: Optional[int] = None,
) -> pd.DataFrame:
    df = _load_cleaned_df().reset_index(drop=True)
    df["row_id"] = range(len(df))

    ann_cfg = cfg.get("annotation") or {}
    max_rows = annotate_max_rows if annotate_max_rows is not None else ann_cfg.get("max_rows")
    print(
        f"[annotate] annotation.max_rows из настроек = {max_rows!r} "
        f"(None или 0 = все строки после clean, сейчас до обрезки {len(df)} строк)",
        flush=True,
    )
    if max_rows is not None and int(max_rows) > 0:
        df = df.head(int(max_rows)).copy()
        df["row_id"] = range(len(df))

    task = ann_cfg.get("task") or "sentiment_classification"
    thr = float(ann_cfg.get("confidence_threshold") or 0.7)

    agent = AnnotationAgent(
        modality="text",
        text_col="text",
        confidence_threshold=thr,
    )

    if mock_annotation:
        print(
            f"[annotate] MOCK: без transformers/zero-shot, метки по кругу (positive/negative/neutral), "
            f"confidence=0.99 для всех {len(df)} строк.",
            flush=True,
        )
        labeled = df.copy()
        labs = ["positive", "negative", "neutral"]
        labeled[AnnotationAgent.PRED_COL] = [labs[i % 3] for i in range(len(labeled))]
        labeled[AnnotationAgent.CONF_COL] = 0.99
        labeled["needs_review"] = labeled[AnnotationAgent.CONF_COL] < thr
    else:
        print(
            f"[annotate] zero-shot по {len(df)} строкам (модель тяжёлая; на CPU это долго, при OOM уменьшите "
            f"annotation.max_rows в pipeline_config.yaml, --mock-annotation для смоук-теста).",
            flush=True,
        )
        labeled = agent.auto_label(df, task=task)
    agent.generate_spec(labeled, task=task, output_path=str(ANNOTATION_SPEC))
    metrics = agent.check_quality(labeled)

    ann_lines = [
        "# Отчёт AnnotationAgent",
        "",
        f"Задача: `{task}`, порог confidence: {thr}",
        "",
        "## check_quality",
        "",
        "```json",
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
    ]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION_REPORT.write_text("\n".join(ann_lines), encoding="utf-8")

    try:
        labeled.to_parquet(ANNOTATED_PATH, index=False)
    except Exception:
        labeled.to_csv(ANNOTATED_PATH.with_suffix(".csv"), index=False)
    else:
        print(f"[annotate] сохранено: {ANNOTATED_PATH}")
    print(f"[annotate] спецификация: {ANNOTATION_SPEC}")
    return labeled


def _load_annotated_df() -> pd.DataFrame:
    if ANNOTATED_PATH.is_file():
        return pd.read_parquet(ANNOTATED_PATH)
    p = ANNOTATED_PATH.with_suffix(".csv")
    if p.is_file():
        return pd.read_csv(p)
    raise FileNotFoundError("Нет annotated — выполните annotate.")


def export_review_queue(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    conf_col = AnnotationAgent.CONF_COL
    pred_col = AnnotationAgent.PRED_COL
    low = df[pd.to_numeric(df[conf_col], errors="coerce") < threshold].copy()
    cols = ["row_id", "text", pred_col, conf_col, "corrected_label"]
    for c in cols:
        if c not in low.columns and c != "corrected_label":
            raise KeyError(f"В данных нет колонки {c}")
    out = low[[c for c in cols if c in low.columns]].copy()
    if "corrected_label" not in out.columns:
        out["corrected_label"] = ""
    else:
        out["corrected_label"] = ""
    out.to_csv(REVIEW_QUEUE, index=False, encoding="utf-8")
    print(f"[HITL] очередь на проверку: {REVIEW_QUEUE} ({len(out)} строк)")
    # Без колонки text — удобно править в Excel; полный текст остаётся в review_queue.csv
    narrow = pd.DataFrame(
        {
            "row_id": low["row_id"].astype(int),
            "pred_label": low[pred_col].astype(str),
            "confidence": low[conf_col],
            "corrected_label": "",
        }
    )
    narrow.to_csv(REVIEW_LABELS_CORRECTED, index=False, encoding="utf-8")
    print(
        f"[HITL] для правок (рекомендуется): {REVIEW_LABELS_CORRECTED} — заполните corrected_label, "
        f"сохраните UTF-8 CSV. Полный {REVIEW_QUEUE} с длинным text Excel часто портит при сохранении."
    )
    return out


def _hitl_labels_dict_for_needed(needed: set[int]) -> Optional[Dict[int, str]]:
    """Словарь row_id -> метка для всех needed; сначала узкий файл, иначе review_queue_corrected.csv."""

    def _from_two_col_df(corr: pd.DataFrame) -> Dict[int, str]:
        corr = corr.dropna(subset=["row_id"], how="any").copy()
        corr["row_id"] = pd.to_numeric(corr["row_id"], errors="coerce")
        corr = corr.dropna(subset=["row_id"])
        corr["row_id"] = corr["row_id"].astype(int)
        out: Dict[int, str] = {}
        for _, row in corr.iterrows():
            rid = int(row["row_id"])
            if rid not in needed:
                continue
            lab = _normalized_hitl_label(row.get("corrected_label"))
            if lab:
                out[rid] = lab
        return out

    if REVIEW_LABELS_CORRECTED.is_file():
        mini = pd.read_csv(REVIEW_LABELS_CORRECTED)
        if "row_id" in mini.columns and "corrected_label" in mini.columns:
            d = _from_two_col_df(mini)
            if set(d.keys()) == needed:
                return d

    if REVIEW_CORRECTED.is_file():
        wide = pd.read_csv(REVIEW_CORRECTED)
        if "row_id" in wide.columns and "corrected_label" in wide.columns:
            d = _from_two_col_df(wide)
            if set(d.keys()) == needed:
                return d

    return None


def hitl_corrections_ready(queue_df: pd.DataFrame) -> bool:
    if queue_df.empty:
        return True
    needed = set(queue_df["row_id"].astype(int).tolist())
    return _hitl_labels_dict_for_needed(needed) is not None


def _hitl_diagnose_missing_labels(queue_df: pd.DataFrame) -> str:
    needed = sorted(set(queue_df["row_id"].astype(int).tolist()))
    lines = [f"Ожидается метка для {len(needed)} row_id."]
    if REVIEW_LABELS_CORRECTED.is_file():
        mini = pd.read_csv(REVIEW_LABELS_CORRECTED)
        if "row_id" in mini.columns and "corrected_label" in mini.columns:
            mini = mini.dropna(subset=["row_id"], how="any")
            mini["row_id"] = pd.to_numeric(mini["row_id"], errors="coerce")
            mini = mini.dropna(subset=["row_id"])
            mini["row_id"] = mini["row_id"].astype(int)
            ok = set(
                int(r["row_id"])
                for _, r in mini.iterrows()
                if _normalized_hitl_label(r.get("corrected_label"))
            )
            miss = [r for r in needed if r not in ok]
            lines.append(f"Файл {REVIEW_LABELS_CORRECTED}: заполнено {len(ok & set(needed))}/{len(needed)}.")
            if miss:
                lines.append(f"Пустые/некорректные corrected_label, примеры row_id: {miss[:25]}{'…' if len(miss) > 25 else ''}")
    if REVIEW_CORRECTED.is_file():
        lines.append(
            f"Если правили только {REVIEW_CORRECTED}: откройте в редакторе хвост строки — "
            f"после confidence часто пропадает последняя колонка после сохранения из Excel."
        )
    return "\n".join(lines)


def merge_hitl_labels(annotated: pd.DataFrame, threshold: float) -> pd.DataFrame:
    pred_col = AnnotationAgent.PRED_COL
    out = annotated.copy()
    out["label_final"] = out[pred_col].astype(str)

    low_mask = pd.to_numeric(out[AnnotationAgent.CONF_COL], errors="coerce") < threshold
    low_ids = set(out.loc[low_mask, "row_id"].astype(int).tolist())

    labels = _hitl_labels_dict_for_needed(low_ids) if low_ids else {}
    if low_ids and labels:
        for rid, cl in labels.items():
            if rid in low_ids:
                out.loc[out["row_id"] == rid, "label_final"] = cl
    return out


def step_hitl_gate(annotated: pd.DataFrame, threshold: float) -> bool:
    """Пишет review_queue. Возвращает True, если можно сразу мержить (нет очереди или файл правок готов)."""
    q = export_review_queue(annotated, threshold)
    if q.empty:
        print("[HITL] низкой уверенности нет — corrected файл не нужен.")
        return True
    if hitl_corrections_ready(q):
        print("[HITL] правки полные — можно merge-hitl.")
        return True
    print(
        "\n--- HITL: требуется человек ---\n"
        f"1) Предпочтительно: откройте {REVIEW_LABELS_CORRECTED} (без длинного текста — Excel не ломает файл).\n"
        "2) Заполните corrected_label: positive / negative / neutral для каждой строки.\n"
        f"3) Сохраните как UTF-8 CSV. Полный текст при необходимости смотрите в {REVIEW_QUEUE}.\n"
        f"   Альтернатива: широкий {REVIEW_CORRECTED} (в Excel последняя колонка часто обрезается).\n"
        "4) Запустите: python run_pipeline.py --from-step merge-hitl\n"
    )
    return False


def step_merge_hitl(cfg: Dict[str, Any]) -> pd.DataFrame:
    annotated = _load_annotated_df()
    thr = float((cfg.get("annotation") or {}).get("confidence_threshold") or 0.7)
    q = pd.read_csv(REVIEW_QUEUE) if REVIEW_QUEUE.is_file() else pd.DataFrame()
    if not q.empty and not hitl_corrections_ready(q):
        raise SystemExit(
            "Неполные правки HITL. Нужна метка corrected_label для каждого row_id из очереди.\n"
            f"{_hitl_diagnose_missing_labels(q)}"
        )
    merged = merge_hitl_labels(annotated, thr)
    try:
        merged.to_parquet(MERGED_PATH, index=False)
    except Exception:
        merged.to_csv(MERGED_PATH.with_suffix(".csv"), index=False)
    print(f"[merge-hitl] сохранено: {MERGED_PATH}")
    return merged


def _load_merged_df() -> pd.DataFrame:
    if MERGED_PATH.is_file():
        return pd.read_parquet(MERGED_PATH)
    p = MERGED_PATH.with_suffix(".csv")
    if p.is_file():
        return pd.read_csv(p)
    raise FileNotFoundError("Нет merged_after_hitl — выполните merge-hitl.")


def run_al_cycle(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, ActiveLearningAgent]:
    al_cfg = cfg.get("al") or {}
    strategy = str(al_cfg.get("strategy") or "entropy")
    n_iterations = int(al_cfg.get("n_iterations") or 5)
    batch_size = int(al_cfg.get("batch_size") or 20)
    min_class_count = int(al_cfg.get("min_class_count") or 10)
    test_size = float(al_cfg.get("test_size") or 0.2)
    initial_labeled = int(al_cfg.get("initial_labeled") or 50)
    random_state = int(al_cfg.get("random_state") or 42)

    labeled_df, pool_df, test_df = prepare_al_data(
        df,
        text_col="text",
        label_col="label_final",
        min_class_count=min_class_count,
        test_size=test_size,
        initial_labeled=initial_labeled,
        random_state=random_state,
    )
    agent = ActiveLearningAgent(random_state=random_state)
    labeled = labeled_df.copy()
    pool = pool_df.copy()
    history: List[Dict[str, Any]] = []

    agent.fit(labeled)
    m0 = agent.evaluate(labeled, test_df)
    history.append(
        {
            "iteration": 0,
            "n_labeled": int(len(labeled)),
            "accuracy": m0["accuracy"],
            "f1": m0["f1"],
        }
    )

    for it in range(1, n_iterations + 1):
        if pool.empty:
            break
        idx = agent.query(pool, strategy, batch_size)  # type: ignore[arg-type]
        if not idx:
            break
        to_add = pool.loc[idx]
        pool = pool.drop(index=idx)
        labeled = pd.concat([labeled, to_add], axis=0, ignore_index=True)
        agent.fit(labeled)
        m = agent.evaluate(labeled, test_df)
        history.append(
            {
                "iteration": it,
                "n_labeled": int(len(labeled)),
                "accuracy": m["accuracy"],
                "f1": m["f1"],
            }
        )

    return history, labeled, test_df, agent


def step_al(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], pd.DataFrame, pd.DataFrame, ActiveLearningAgent]:
    df = _load_merged_df()
    history, labeled, test_df, agent = run_al_cycle(df, cfg)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    AL_HISTORY_JSON.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    curve_path = agent.report(history, path=str(LEARNING_CURVE))

    al_cfg = cfg.get("al") or {}
    strategy = str(al_cfg.get("strategy") or "entropy")
    extra = ""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            labeled_df, pool_df, _ = prepare_al_data(
                df,
                text_col="text",
                label_col="label_final",
                min_class_count=int(al_cfg.get("min_class_count") or 10),
                test_size=float(al_cfg.get("test_size") or 0.2),
                initial_labeled=int(al_cfg.get("initial_labeled") or 50),
                random_state=int(al_cfg.get("random_state") or 42),
            )
            prep = ActiveLearningAgent(random_state=int(al_cfg.get("random_state") or 42))
            prep.fit(labeled_df)
            batch_size = int(al_cfg.get("batch_size") or 20)
            idx = prep.query(pool_df, strategy, batch_size)  # type: ignore[arg-type]
            expl = prep.explain_selection(pool_df, idx, strategy)  # type: ignore[arg-type]
            extra = f"\n## Пояснение выбора (Claude API), первая итерация\n\n{expl}\n"
        except Exception as e:
            extra = f"\n## Пояснение выбора (Claude API)\n\nОшибка: {e}\n"

    al_md = [
        "# Отчёт ActiveLearningAgent",
        "",
        f"Стратегия: `{strategy}`, label_col: `label_final`",
        "",
        f"Кривая обучения: `{curve_path}`",
        "",
        "## История (accuracy / F1 macro vs n_labeled)",
        "",
        "```json",
        json.dumps(history, ensure_ascii=False, indent=2),
        "```",
        "",
        "**Примечание:** в `run_cycle` метки в пуле известны симулятору (оракул); в проде здесь был бы второй HITL.",
        extra,
    ]
    AL_REPORT.write_text("\n".join(al_md), encoding="utf-8")
    print(f"[al] отчёт: {AL_REPORT}, кривая: {curve_path}")
    return history, labeled, test_df, agent


def step_train(cfg: Dict[str, Any], labeled: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_cfg = cfg.get("train") or {}
    random_state = int(train_cfg.get("random_state") or 42)
    agent = ActiveLearningAgent(
        random_state=random_state,
        max_features=int(train_cfg.get("max_features") or 30_000),
        ngram_range=tuple(train_cfg.get("ngram_range") or (1, 2)),  # type: ignore[arg-type]
        logreg_C=float(train_cfg.get("logreg_C") or 1.0),
        max_iter=int(train_cfg.get("max_iter") or 500),
    )
    agent.fit(labeled)
    metrics = agent.evaluate(labeled, test_df)
    metrics.update(
        {
            "n_train": int(len(labeled)),
            "n_test": int(len(test_df)),
            "label_col": "label_final",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
    )

    import joblib

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": agent.vectorizer_,
            "clf": agent.clf_,
            "text_col": "text",
            "label_col": "label_final",
        },
        MODEL_BUNDLE,
    )
    MODEL_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train] модель: {MODEL_BUNDLE}, метрики: {MODEL_METRICS}")


def step_finalize(labeled: pd.DataFrame) -> None:
    ROOT.joinpath("data", "labeled").mkdir(parents=True, exist_ok=True)
    try:
        labeled.to_parquet(FINAL_LABELED_PATH, index=False)
    except Exception:
        labeled.to_csv(FINAL_LABELED_PATH.with_suffix(".csv"), index=False)
    print(f"[finalize] итоговый датасет: {FINAL_LABELED_PATH}")


def run_full_pipeline(
    cfg: Dict[str, Any],
    skip_collect: bool,
    *,
    mock_annotation: bool = False,
    annotate_max_rows: Optional[int] = None,
) -> None:
    step_collect(cfg, skip_collect=skip_collect)
    step_clean(cfg)
    annotated = step_annotate(
        cfg,
        mock_annotation=mock_annotation,
        annotate_max_rows=annotate_max_rows,
    )
    thr = float((cfg.get("annotation") or {}).get("confidence_threshold") or 0.7)
    if not step_hitl_gate(annotated, thr):
        sys.exit(0)
    merged = step_merge_hitl(cfg)
    _history, labeled, test_df, _ = step_al(cfg)
    step_train(cfg, labeled, test_df)
    step_finalize(labeled)


def main() -> None:
    parser = argparse.ArgumentParser(description="Финальный дата-пайплайн (4 агента + HITL)")
    parser.add_argument(
        "--from-step",
        choices=(
            "collect",
            "clean",
            "annotate",
            "merge-hitl",
            "al",
            "train",
            "all",
        ),
        default="all",
        help="С какого этапа продолжить (all = полный прогон с HITL-воротами).",
    )
    parser.add_argument("--skip-collect", action="store_true", help="Не вызывать сбор, нужен raw CSV")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_PIPELINE_CONFIG),
        help="Путь к pipeline_config.yaml",
    )
    parser.add_argument(
        "--mock-annotation",
        action="store_true",
        help="Не вызывать zero-shot: синтетические pred_label/confidence (проверка HITL→AL→train без GPU/CPU-часов).",
    )
    parser.add_argument(
        "--annotate-max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Переопределить annotation.max_rows из yaml (удобно, если в файле null, а нужен лимит).",
    )
    args = parser.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_pipeline_config(cfg_path)
    print(f"[pipeline] конфиг: {cfg_path}", flush=True)
    ensure_dirs()

    if args.from_step == "all":
        run_full_pipeline(
            cfg,
            skip_collect=args.skip_collect,
            mock_annotation=args.mock_annotation,
            annotate_max_rows=args.annotate_max_rows,
        )
        return

    if args.from_step == "collect":
        step_collect(cfg, skip_collect=args.skip_collect)
        return
    if args.from_step == "clean":
        if not args.skip_collect:
            step_collect(cfg, skip_collect=False)
        step_clean(cfg)
        return
    if args.from_step == "annotate":
        step_annotate(
            cfg,
            mock_annotation=args.mock_annotation,
            annotate_max_rows=args.annotate_max_rows,
        )
        return
    if args.from_step == "merge-hitl":
        merged = step_merge_hitl(cfg)
        _history, labeled, test_df, _ = step_al(cfg)
        step_train(cfg, labeled, test_df)
        step_finalize(labeled)
        return
    if args.from_step == "al":
        _history, labeled, test_df, _ = step_al(cfg)
        step_train(cfg, labeled, test_df)
        step_finalize(labeled)
        return
    if args.from_step == "train":
        raise SystemExit(
            "Используйте --from-step merge-hitl или al (train ожидает свежий цикл AL в этом процессе)."
        )


if __name__ == "__main__":
    main()
