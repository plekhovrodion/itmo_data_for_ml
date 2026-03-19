"""
AnnotationAgent — авторазметка, спецификация разметки, метрики качества, экспорт в Label Studio.

Labeling config (XML для импорта предсказаний должен совпадать с именами полей ниже):

    <View>
      <Text name="text" value="$text"/>
      <Choices name="sentiment" toName="text">
        <Choice value="positive"/>
        <Choice value="negative"/>
        <Choice value="neutral"/>
      </Choices>
    </View>

Константы экспорта (совместимость JSON ↔ XML):
  - data-ключ: "text"  →  <Text name="text" value="$text"/>
  - choices: from_name="sentiment", to_name="text"  →  <Choices name="sentiment" toName="text">
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

# Согласованы с примером Labeling config в docstring модуля
LS_DATA_TEXT_KEY = "text"
LS_CHOICES_FROM_NAME = "sentiment"
LS_CHOICES_TO_NAME = "text"

# Мультиязычная zero-shot NLI (подходит для русскоязычных новостей)
DEFAULT_ZS_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

TASK_LABELS: Dict[str, Dict[str, Any]] = {
    "sentiment_classification": {
        "description": (
            "Классификация тональности текста новости: определить, выражает ли фрагмент "
            "позитивное, негативное или нейтральное отношение к описываемым событиям."
        ),
        "classes": {
            "positive": "Текст в целом выражает одобрение, надежду или позитивную оценку ситуации.",
            "negative": "Текст выражает критику, тревогу, сожаление или негативную оценку.",
            "neutral": "Текст преимущественно информативен, без явной эмоциональной окраски.",
        },
        "hypothesis_template": "The sentiment of this text is {}.",
        "candidate_labels": ["positive", "negative", "neutral"],
    },
    "topic_news": {
        "description": "Отнесение новости к одной из тематических категорий.",
        "classes": {
            "politics": "Государство, власть, выборы, международные отношения.",
            "economy": "Рынки, бизнес, финансы, занятость.",
            "society": "Социальная сфера, здравоохранение, образование, быт.",
            "tech": "Наука, технологии, IT, космос.",
            "sports": "Спортивные события и персоны.",
            "culture": "Культура, искусство, кино, литература.",
        },
        "hypothesis_template": "This news article is mainly about {}.",
        "candidate_labels": ["politics", "economy", "society", "tech", "sports", "culture"],
    },
}


@dataclass
class AnnotationSpec:
    """Результат generate_spec: путь к Markdown и метаданные задачи."""

    path: str
    task: str
    classes: List[str]
    lines: List[str] = field(default_factory=list)


class AnnotationAgent:
    """
    Скиллы: auto_label, generate_spec, check_quality, export_to_labelstudio,
    export_low_confidence_for_review (бонус).
    """

    PRED_COL = "pred_label"
    CONF_COL = "confidence"

    def __init__(
        self,
        modality: str = "text",
        *,
        text_col: str = "text",
        confidence_threshold: float = 0.5,
        zero_shot_model: Optional[str] = None,
        device: Optional[int] = None,
    ):
        self.modality = modality
        self.text_col = text_col
        self.confidence_threshold = confidence_threshold
        self._zs_model_name = zero_shot_model or DEFAULT_ZS_MODEL
        self._device = device if device is not None else self._pick_device()
        self._pipeline = None

    @staticmethod
    def _pick_device() -> int:
        try:
            import torch

            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    def _get_zero_shot_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "Для auto_label установите: pip install transformers torch"
            ) from e
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self._zs_model_name,
            device=self._device,
        )
        return self._pipeline

    def auto_label(
        self,
        df: pd.DataFrame,
        task: str = "sentiment_classification",
    ) -> pd.DataFrame:
        """
        Авторазметка. modality='text': zero-shot по колонке text_col.

        Добавляет колонки: pred_label, confidence.
        """
        if self.modality != "text":
            raise NotImplementedError(
                f"modality={self.modality!r} не реализована; используйте modality='text'."
            )
        if self.text_col not in df.columns:
            raise ValueError(f"В DataFrame нет колонки {self.text_col!r}.")

        if task not in TASK_LABELS:
            raise ValueError(f"Неизвестная задача {task!r}. Доступны: {list(TASK_LABELS)}.")

        cfg = TASK_LABELS[task]
        labels: Sequence[str] = cfg["candidate_labels"]
        hyp = cfg["hypothesis_template"]

        clf = self._get_zero_shot_pipeline()
        out = df.copy()
        texts = out[self.text_col].fillna("").astype(str).tolist()

        pred_labels: List[str] = []
        confidences: List[float] = []

        # pipeline принимает список последовательностей
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for text in chunk:
                if not text.strip():
                    pred_labels.append(labels[0])
                    confidences.append(0.0)
                    continue
                res = clf(
                    text,
                    candidate_labels=list(labels),
                    hypothesis_template=hyp,
                    multi_label=False,
                )
                pred_labels.append(res["labels"][0])
                confidences.append(float(res["scores"][0]))

        out[self.PRED_COL] = pred_labels
        out[self.CONF_COL] = confidences
        out["needs_review"] = out[self.CONF_COL] < self.confidence_threshold
        return out

    def generate_spec(
        self,
        df: pd.DataFrame,
        task: str = "sentiment_classification",
        output_path: str = "annotation_spec.md",
        pred_col: Optional[str] = None,
    ) -> AnnotationSpec:
        """
        Пишет Markdown: задача, определения классов, ≥3 примера на класс, граничные случаи.
        """
        if task not in TASK_LABELS:
            raise ValueError(f"Неизвестная задача {task!r}.")

        cfg = TASK_LABELS[task]
        class_defs: Dict[str, str] = cfg["classes"]
        classes = list(class_defs.keys())
        pred_col = pred_col or self.PRED_COL

        lines: List[str] = [
            f"# Спецификация разметки: {task}",
            "",
            "## Задача",
            "",
            cfg["description"],
            "",
            "## Классы и определения",
            "",
        ]
        for c, definition in class_defs.items():
            lines.append(f"- **{c}**: {definition}")
        lines.extend(["", "## Примеры по классам", ""])

        text_col = self.text_col
        for cls in classes:
            lines.append(f"### {cls}")
            samples: List[str] = []
            if pred_col in df.columns and text_col in df.columns:
                cls_df = df[df[pred_col].astype(str) == cls]
                for t in cls_df[text_col].tolist():
                    s = str(t) if not pd.isna(t) else ""
                    s = s.strip()[:500]
                    if s:
                        samples.append(s)
                    if len(samples) >= 3:
                        break
            idx = 0
            while len(samples) < 3 and text_col in df.columns and len(df) > 0:
                row = df.iloc[idx % len(df)]
                idx += 1
                t = row.get(text_col, "")
                s = str(t) if not pd.isna(t) else ""
                s = s.strip()[:500]
                if s and s not in samples:
                    samples.append(s)
                if idx > len(df) * 3:
                    break
            while len(samples) < 3:
                samples.append(
                    "*(пример для иллюстрации — замените фрагментом из вашего датасета)*"
                )
            for j, s in enumerate(samples[:3], 1):
                lines.append(f"{j}. {s}")
            lines.append("")

        lines.extend(
            [
                "## Граничные случаи",
                "",
                "- **Ирония и сарказм**: формально позитивные слова при отрицательной оценке ситуации — "
                "ориентируйтесь на общий смысл, а не на отдельные лексемы.",
                "- **Смешанная тональность**: если позитив и негатив сопоставимы, выбирайте **neutral**, "
                "если в задании не сказано иное.",
                "- **Заголовок vs текст**: при расхождении приоритет у полного текста (колонка текста новости).",
                "",
            ]
        )

        # 1–2 коротких реальных примера «спорных» (короткий текст / пустой контекст)
        if text_col in df.columns:
            short = df[df[text_col].astype(str).str.len() < 40]
            if len(short) > 0:
                lines.append("Примеры коротких текстов из данных (могут быть неоднозначны):")
                for _, row in short.head(2).iterrows():
                    lines.append(f"- «{str(row[text_col])[:200]}»")
                lines.append("")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        body = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(body)

        return AnnotationSpec(path=output_path, task=task, classes=classes, lines=lines)

    def check_quality(
        self,
        df_labeled: pd.DataFrame,
        *,
        pred_col: Optional[str] = None,
        human_col: Optional[str] = None,
        auto_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Метрики: label_dist, confidence_mean, kappa (если есть две колонки разметчиков),
        agreement_pct при наличии пары human/auto.
        """
        pred_col = pred_col or self.PRED_COL
        auto_col = auto_col or pred_col

        metrics: Dict[str, Any] = {
            "label_dist": {},
            "confidence_mean": None,
            "kappa": None,
            "agreement_pct": None,
        }

        if pred_col in df_labeled.columns:
            vc = df_labeled[pred_col].value_counts(dropna=False)
            metrics["label_dist"] = {str(k): int(v) for k, v in vc.items()}

        if self.CONF_COL in df_labeled.columns:
            s = pd.to_numeric(df_labeled[self.CONF_COL], errors="coerce")
            if s.notna().any():
                metrics["confidence_mean"] = round(float(s.mean()), 4)

        hc = human_col or self._detect_human_column(df_labeled)
        if hc and auto_col in df_labeled.columns and hc in df_labeled.columns:
            sub = df_labeled[[hc, auto_col]].dropna()
            a = sub[hc].astype(str)
            b = sub[auto_col].astype(str)
            if len(a) > 0:
                metrics["agreement_pct"] = round(float((a == b).mean() * 100), 2)
                try:
                    metrics["kappa"] = round(
                        float(cohen_kappa_score(a, b)),
                        4,
                    )
                except ValueError:
                    metrics["kappa"] = None

        return metrics

    @staticmethod
    def _detect_human_column(df: pd.DataFrame) -> Optional[str]:
        for name in ("label_human", "label", "human_label", "gold", "y_true"):
            if name in df.columns:
                return name
        return None

    def export_to_labelstudio(
        self,
        df: pd.DataFrame,
        path: str = "labelstudio_import.json",
        *,
        pred_col: Optional[str] = None,
        text_col: Optional[str] = None,
    ) -> str:
        """JSON в формате импорта Label Studio (список задач с predictions)."""
        pred_col = pred_col or self.PRED_COL
        text_col = text_col or self.text_col
        tasks: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            text_val = row.get(text_col, "")
            if pd.isna(text_val):
                text_val = ""
            text_val = str(text_val)

            pred = row.get(pred_col, "")
            if pd.isna(pred):
                pred = ""
            pred = str(pred)

            score = 1.0
            if self.CONF_COL in row.index:
                try:
                    score = float(row[self.CONF_COL])
                except (TypeError, ValueError):
                    score = 0.0

            task: Dict[str, Any] = {
                "data": {LS_DATA_TEXT_KEY: text_val},
                "predictions": [
                    {
                        "model_version": "AnnotationAgent-zero-shot",
                        "score": score,
                        "result": [
                            {
                                "id": "auto",
                                "from_name": LS_CHOICES_FROM_NAME,
                                "to_name": LS_CHOICES_TO_NAME,
                                "type": "choices",
                                "value": {"choices": [pred]},
                            }
                        ],
                    }
                ],
            }
            tasks.append(task)

        out_dir = os.path.dirname(path) or "."
        os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)

        return path

    def export_low_confidence_for_review(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
        path: str = "labelstudio_review.json",
        **kwargs: Any,
    ) -> str:
        """
        Бонус: примеры с confidence < threshold в отдельном JSON для ручной доразметки.
        """
        thr = self.confidence_threshold if threshold is None else threshold
        if self.CONF_COL not in df.columns:
            raise ValueError(f"Нет колонки {self.CONF_COL}. Сначала вызовите auto_label.")
        sub = df[pd.to_numeric(df[self.CONF_COL], errors="coerce") < thr].copy()
        return self.export_to_labelstudio(sub, path=path, **kwargs)


def auto_label_op(df: pd.DataFrame, modality: str = "text") -> pd.DataFrame:
    """Оператор пайплайна: авторазметка с настройками по умолчанию."""
    return AnnotationAgent(modality=modality).auto_label(df)
