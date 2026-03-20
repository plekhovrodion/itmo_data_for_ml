"""DataQualityAgent: detect_issues, fix, compare."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _entropy_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-np.sum(p * np.log(p + 1e-30)))


class DataQualityAgent:
    """Аудит и очистка табличных данных."""

    def __init__(
        self,
        label_column: str = "label",
        outlier_columns: list[str] | None = None,
        duplicate_subset: list[str] | None = None,
        iqr_multiplier: float = 1.5,
    ) -> None:
        self.label_column = label_column
        self.outlier_columns = outlier_columns or [
            "price_per_m2",
            "total_price_rub",
            "area_m2",
        ]
        self.duplicate_subset = duplicate_subset or [
            "price_per_m2",
            "area_m2",
            "total_price_rub",
            "source",
        ]
        self.iqr_multiplier = iqr_multiplier

    def _numeric_cols_present(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.outlier_columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    def _iqr_mask(self, s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() < 3:
            return pd.Series(False, index=s.index)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lo = q1 - self.iqr_multiplier * iqr
        hi = q3 + self.iqr_multiplier * iqr
        return (s < lo) | (s > hi)

    def detect_issues(self, df: pd.DataFrame) -> dict[str, Any]:
        df = df.copy()
        n = len(df)
        missing_per_col = df.isna().sum().to_dict()
        missing_total = int(df.isna().sum().sum())
        missing_frac = missing_total / (n * max(df.shape[1], 1)) if n else 0.0

        dup_full = int(df.duplicated().sum())
        sub = [c for c in self.duplicate_subset if c in df.columns]
        dup_sub = int(df.duplicated(subset=sub).sum()) if sub else 0

        outlier_info: dict[str, Any] = {"method": "IQR", "iqr_multiplier": self.iqr_multiplier, "per_column": {}}
        outlier_masks: list[pd.Series] = []
        for col in self._numeric_cols_present(df):
            m = self._iqr_mask(df[col])
            outlier_info["per_column"][col] = int(m.sum())
            outlier_masks.append(m.fillna(False))

        if outlier_masks:
            any_out = outlier_masks[0].copy()
            for m in outlier_masks[1:]:
                any_out = any_out | m
            rows_any_outlier = int(any_out.sum())
        else:
            rows_any_outlier = 0

        imbalance: dict[str, Any]
        if self.label_column in df.columns and df[self.label_column].notna().any():
            lab = pd.to_numeric(df[self.label_column], errors="coerce").dropna()
            if len(lab) < 2 or lab.nunique() < 2:
                imbalance = {
                    "kind": "continuous",
                    "note": "мало уникальных значений label",
                    "n_unique": int(lab.nunique()) if len(lab) else 0,
                }
            else:
                try:
                    cats = pd.qcut(lab, q=min(10, max(3, lab.nunique())), duplicates="drop")
                    vc = cats.value_counts()
                    entropy = _entropy_from_counts(vc.values.astype(float))
                    imbalance = {
                        "kind": "continuous_binned",
                        "bins": len(vc),
                        "entropy": entropy,
                        "bin_counts": {str(k): int(v) for k, v in vc.items()},
                    }
                except Exception:
                    imbalance = {"kind": "continuous", "note": "не удалось разбить на квантили", "skew": float(lab.skew())}
        else:
            imbalance = {"kind": "none", "note": "колонка label отсутствует или пуста"}

        return {
            "n_rows": n,
            "n_cols": df.shape[1],
            "missing": {
                "per_column": {k: int(v) for k, v in missing_per_col.items()},
                "total_cells": missing_total,
                "fraction_of_cells": float(missing_frac),
            },
            "duplicates": {
                "full_row": dup_full,
                "subset": sub,
                "subset_duplicates": dup_sub,
            },
            "outliers": {**outlier_info, "rows_with_any_outlier": rows_any_outlier},
            "imbalance": imbalance,
        }

    def fix(self, df: pd.DataFrame, strategy: dict[str, Any]) -> pd.DataFrame:
        df = df.copy()
        iqr_mult = float(strategy.get("iqr_multiplier", self.iqr_multiplier))
        old_mult = self.iqr_multiplier
        self.iqr_multiplier = iqr_mult
        try:
            miss = strategy.get("missing", {})
            num_mode = miss.get("numeric", "median")
            obj_fill = miss.get("object_fill", "")

            num_cols = df.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if not df[col].isna().any():
                    continue
                if df[col].notna().sum() == 0:
                    df[col] = 0.0
                    continue
                med = df[col].median()
                if pd.isna(med) and num_mode in ("median", "mean", None):
                    df[col] = df[col].fillna(0.0)
                elif num_mode == "median":
                    df[col] = df[col].fillna(med)
                elif num_mode == "mean":
                    m = df[col].mean()
                    df[col] = df[col].fillna(0.0 if pd.isna(m) else m)
                elif num_mode == "drop_rows":
                    df = df.dropna(subset=[col])
                else:
                    df[col] = df[col].fillna(med if not pd.isna(med) else 0.0)

            obj_cols = df.select_dtypes(include=["object", "string"]).columns
            for col in obj_cols:
                if df[col].isna().any():
                    if miss.get("object_mode") == "mode" and df[col].mode().size:
                        df[col] = df[col].fillna(df[col].mode().iloc[0])
                    else:
                        df[col] = df[col].fillna(obj_fill)

            dup = strategy.get("duplicates", {})
            if dup.get("mode") == "drop":
                sub = dup.get("subset")
                if sub:
                    use = [c for c in sub if c in df.columns]
                    if use:
                        keep = dup.get("keep", "first")
                        df = df.drop_duplicates(subset=use, keep=keep)

            out = strategy.get("outliers", {})
            mode = out.get("mode", "none")
            cols = [c for c in out.get("columns", self.outlier_columns) if c in df.columns]
            if mode == "clip_iqr":
                for col in cols:
                    s = pd.to_numeric(df[col], errors="coerce")
                    if s.notna().sum() < 2:
                        continue
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    if not math.isfinite(iqr) or iqr == 0:
                        continue
                    lo = q1 - iqr_mult * iqr
                    hi = q3 + iqr_mult * iqr
                    df[col] = s.clip(lower=lo, upper=hi)
            elif mode == "drop_iqr":
                mask = pd.Series(False, index=df.index)
                for col in cols:
                    mask = mask | self._iqr_mask(df[col]).fillna(False)
                df = df.loc[~mask]

            if strategy.get("coerce_city_or_region_str") and "city_or_region" in df.columns:
                df["city_or_region"] = df["city_or_region"].astype(str)

            return df.reset_index(drop=True)
        finally:
            self.iqr_multiplier = old_mult

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        r1 = self.detect_issues(df_before)
        r2 = self.detect_issues(df_after)

        def row(metric: str, before: Any, after: Any) -> dict[str, Any]:
            delta = after - before if isinstance(before, (int, float)) and isinstance(after, (int, float)) else None
            return {"metric": metric, "before": before, "after": after, "delta": delta}

        rows = [
            row("rows", r1["n_rows"], r2["n_rows"]),
            row("missing_cells_total", r1["missing"]["total_cells"], r2["missing"]["total_cells"]),
            row("duplicate_subset", r1["duplicates"]["subset_duplicates"], r2["duplicates"]["subset_duplicates"]),
            row("outlier_rows_any", r1["outliers"]["rows_with_any_outlier"], r2["outliers"]["rows_with_any_outlier"]),
        ]

        e1 = r1["imbalance"].get("entropy") if isinstance(r1["imbalance"], dict) else None
        e2 = r2["imbalance"].get("entropy") if isinstance(r2["imbalance"], dict) else None
        if e1 is not None or e2 is not None:
            rows.append(
                {
                    "metric": "label_binned_entropy",
                    "before": e1,
                    "after": e2,
                    "delta": (e2 - e1) if e1 is not None and e2 is not None else None,
                }
            )

        return pd.DataFrame(rows)


def load_strategy(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw if isinstance(raw, dict) else {}


def explain_and_recommend(report: dict[str, Any], task_description: str = "") -> str:
    key = __import__("os").environ.get("ANTHROPIC_API_KEY")
    if not key:
        return "LLM отключён: задайте ANTHROPIC_API_KEY для explain_and_recommend."
    return "LLM-рекомендация не реализована в мини-проекте; используйте отчёт detect_issues и strategy.yaml."


def save_report_json(report: dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
