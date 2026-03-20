"""AnnotationAgent: авторазметка сегмента цены, спецификация, Label Studio, очередь проверки."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def _segment_from_price(series: pd.Series, labels: list[str]) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    cats = pd.qcut(s, q=3, labels=labels, duplicates="drop")
    return cats.astype(str)


class AnnotationAgent:
    """
    Табличная модальность: предсказание ценового сегмента (budget/mid/premium)
    по признакам без `price_per_m2` (площадь, комнаты, гео, источник).
    """

    def __init__(
        self,
        modality: str = "tabular",
        config_path: str | Path | None = None,
    ) -> None:
        self.modality = modality
        self.config_path = Path(config_path or "annotation_config.yaml")
        self.cfg = self._load_cfg()

    def _load_cfg(self) -> dict[str, Any]:
        import yaml

        if not self.config_path.exists():
            return {"annotation": {}}
        with open(self.config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return raw if isinstance(raw, dict) else {}

    def _ann(self) -> dict[str, Any]:
        return self.cfg.get("annotation", {})

    def auto_label(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.modality not in ("tabular", "text"):
            raise ValueError("В этом проекте поддержаны modality=tabular|text (text — заглушка).")

        df = df.copy()
        ann = self._ann()
        labels = list(ann.get("segment_labels", ["budget", "mid", "premium"]))
        rs = int(ann.get("random_state", 42))

        if "price_per_m2" not in df.columns:
            raise ValueError("Нужна колонка price_per_m2 для эталонных сегментов.")

        y = _segment_from_price(df["price_per_m2"], labels)
        mask = y.notna()
        df = df.loc[mask].reset_index(drop=True)
        y = y.loc[mask]

        # Без total_price_rub: иначе утечка (цена ≈ ₽/м² × площадь) в сторону эталона по price_per_m2.
        feature_cols = [c for c in ["area_m2", "rooms", "geo_lat", "geo_lon"] if c in df.columns]
        if "source" in df.columns:
            feature_cols.append("source")

        X = df[feature_cols].copy()
        for c in X.columns:
            if c != "source" and X[c].dtype == object:
                X[c] = pd.to_numeric(X[c], errors="coerce")

        num_cols = [c for c in X.columns if c != "source"]
        cat_cols = ["source"] if "source" in X.columns else []

        transformers = []
        if num_cols:
            transformers.append(
                ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_cols)
            )
        if cat_cols:
            transformers.append(
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
            )

        prep = ColumnTransformer(transformers)
        clf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=rs, class_weight="balanced")
        pipe = Pipeline([("prep", prep), ("clf", clf)])

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        pipe.fit(X, y_enc)
        proba = pipe.predict_proba(X)
        pred_enc = np.argmax(proba, axis=1)
        conf = proba.max(axis=1)

        out = df.copy()
        out["label_auto"] = le.inverse_transform(pred_enc)
        out["confidence"] = conf
        out["label_segment_ref"] = y.values
        return out

    def generate_spec(self, df: pd.DataFrame, task: str = "price_segment") -> Path:
        ann = self._ann()
        labels = ann.get("segment_labels", ["budget", "mid", "premium"])
        out = Path("annotation_spec.md")

        examples: dict[str, list[str]] = {lab: [] for lab in labels}
        if "label_segment_ref" in df.columns and "text" in df.columns:
            for lab in labels:
                sub = df[df["label_segment_ref"].astype(str) == lab]
                for _, r in sub.head(3).iterrows():
                    t = str(r.get("text", ""))[:200]
                    examples[lab].append(t)

        lines = [
            "# Спецификация разметки: сегмент цены за м² (budget / mid / premium)",
            "",
            "## Задача",
            "",
            "- **Что размечаем:** принадлежность объявления к одному из трёх ценовых сегментов по рынку (терции распределения `price_per_m2` на обучающей выборке).",
            "- **Зачем в ML:** прокси-классификация для стратификации, бизнес-правил и последующей регрессии/ранжирования.",
            "",
            "## Классы (с определениями)",
            "",
            "| Класс | Определение | Не путать с |",
            "|-------|-------------|-------------|",
            "| **budget** | Нижняя треть цен за м² среди объявлений в выборке | «дёшево» в абсолютных ₽ без контекста площади |",
            "| **mid** | Средняя треть | «средний чек» по общей цене без учёта м² |",
            "| **premium** | Верхняя треть по ₽/м² | дорогой объект при низкой ₽/м² из-за большой площади |",
            "",
            "## Примеры (≥3 на класс)",
            "",
        ]
        for lab in labels:
            lines.append(f"### {lab}")
            lines.append("")
            for i, ex in enumerate(examples.get(lab, []) or ["(нет примера)"], start=1):
                lines.append(f"{i}. {ex}")
            lines.append("")

        lines.extend(
            [
                "## Граничные случаи",
                "",
                "- Объект на границе терций: допускается согласование с эталоном `label_segment_ref` или эскалация.",
                "- Пустой `text`: ориентироваться на числовые поля и источник (`source`).",
                "- Разные регионы в одном классе: сегмент **относительный** внутри текущей выборки, не федеральная норма.",
                "",
                "## Эскалация",
                "",
                "- При систематическом расхождении авторазметки и эталона — пересмотр признаков или порогов.",
                "",
            ]
        )

        out.write_text("\n".join(lines), encoding="utf-8")
        return out

    def check_quality(self, df_labeled: pd.DataFrame) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if "label_auto" not in df_labeled.columns:
            metrics["note"] = "нет label_auto"
            return metrics

        vc = df_labeled["label_auto"].value_counts(normalize=True)
        metrics["label_dist"] = vc.round(4).to_dict()

        if "confidence" in df_labeled.columns:
            metrics["confidence_mean"] = float(df_labeled["confidence"].mean())
        else:
            metrics["confidence_mean"] = None

        if "label_human" in df_labeled.columns:
            both = df_labeled["label_human"].notna() & df_labeled["label_auto"].notna()
            if both.sum() > 0:
                agree = (
                    df_labeled.loc[both, "label_human"].astype(str) == df_labeled.loc[both, "label_auto"].astype(str)
                ).mean()
                metrics["agreement_pct"] = float(agree)
                metrics["kappa"] = None
                metrics["kappa_note"] = "для κ нужна разметка двух асессоров; используйте agreement_pct"
            else:
                metrics["agreement_pct"] = None
        else:
            metrics["kappa"] = None
            metrics["note"] = "колонка label_human отсутствует — эталон только label_segment_ref"

        if "label_segment_ref" in df_labeled.columns and "label_auto" in df_labeled.columns:
            m = (
                df_labeled["label_auto"].astype(str) == df_labeled["label_segment_ref"].astype(str)
            ).mean()
            metrics["agreement_with_ref_pct"] = float(m)

        return metrics

    def export_to_labelstudio(self, df: pd.DataFrame, out_path: str | Path = "labelstudio_import.json") -> Path:
        """Импорт в Label Studio: Text + Choices (проверено на схеме с from_name=segment)."""
        ann = self._ann()
        nmax = int(ann.get("ls_export_max", 50))
        sub = df.head(nmax)
        tasks: list[dict[str, Any]] = []
        for i, row in sub.iterrows():
            tid = f"row_{i}"
            text = str(row.get("text", ""))[:4000]
            lab = str(row.get("label_auto", ""))
            score = float(row.get("confidence", 0.0) or 0.0)
            tasks.append(
                {
                    "data": {"text": text, "id": tid},
                    "predictions": [
                        {
                            "result": [
                                {
                                    "from_name": "segment",
                                    "to_name": "text",
                                    "type": "choices",
                                    "value": {"choices": [lab]},
                                }
                            ],
                            "score": score,
                        }
                    ],
                }
            )
        path = Path(out_path)
        path.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def export_low_confidence(
        self,
        df: pd.DataFrame,
        out_csv: str | Path = "review_queue.csv",
    ) -> Path | None:
        ann = self._ann()
        thr = float(ann.get("low_confidence_threshold", 0.55))
        if "confidence" not in df.columns:
            return None
        low = df[df["confidence"] < thr].copy()
        if low.empty:
            p = Path(out_csv)
            p.write_text("id,text,label_auto,confidence,label_human\n", encoding="utf-8")
            return p
        low["id"] = low.index.astype(str)
        cols = ["id", "text", "label_auto", "confidence"]
        for c in cols:
            if c not in low.columns and c != "label_human":
                low[c] = ""
        low["label_human"] = ""
        out = low[["id", "text", "label_auto", "confidence", "label_human"]]
        p = Path(out_csv)
        out.to_csv(p, index=False, encoding="utf-8")
        return p
