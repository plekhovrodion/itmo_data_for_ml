"""
DataQualityAgent - агент-детектив для выявления и устранения проблем качества данных.
Обнаруживает: пропуски, дубликаты, выбросы (IQR/z-score), дисбаланс классов.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class DataQualityAgent:
    """
    Агент для автоматического обнаружения и устранения проблем качества данных.

    Скиллы:
    - detect_issues(df) -> QualityReport
    - fix(df, strategy) -> DataFrame
    - compare(df_before, df_after) -> ComparisonReport
    """

    # Колонки для анализа выбросов (производные числовые признаки для текстовых данных)
    DERIVED_NUMERIC_COLS = ['text_len', 'word_count']

    def __init__(self):
        self._last_report: Optional[Dict] = None

    def _add_derived_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет производные числовые колонки для анализа выбросов."""
        df = df.copy()
        if 'text' in df.columns:
            df['text_len'] = df['text'].astype(str).str.len()
            df['word_count'] = df['text'].astype(str).str.split().str.len()
        return df

    def detect_issues(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Обнаруживает проблемы качества данных.

        Returns:
            dict: {
                'missing': {col: {'count': N, 'pct': float}, ...},
                'duplicates': {'full': N, 'by_url': N},
                'outliers': {col: {'indices': [...], 'count': N, 'method': 'iqr'|'zscore'}, ...},
                'imbalance': {col: {'distribution': {...}, 'entropy': float, 'max_min_ratio': float}, ...}
            }
        """
        df_work = self._add_derived_numeric(df)
        report: Dict[str, Any] = {}

        # 1. Missing values
        report['missing'] = {}
        for col in df.columns:
            null_count = df[col].isna().sum()
            empty_count = 0
            if df[col].dtype == 'object':
                empty_count = (df[col].astype(str).str.strip() == '').sum()
            total_missing = null_count + empty_count
            if total_missing > 0:
                report['missing'][col] = {
                    'count': int(total_missing),
                    'pct': round(100 * total_missing / len(df), 2),
                    'null_count': int(null_count),
                    'empty_count': int(empty_count),
                }

        # 2. Duplicates
        full_dups = df.duplicated().sum()
        by_url = df['url'].duplicated().sum() if 'url' in df.columns else 0
        report['duplicates'] = {
            'full': int(full_dups),
            'by_url': int(by_url),
        }

        # 3. Outliers (IQR и z-score для числовых и производных колонок)
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            numeric_cols = [c for c in self.DERIVED_NUMERIC_COLS if c in df_work.columns]

        report['outliers'] = {}
        for col in numeric_cols:
            series = df_work[col].dropna()
            if len(series) < 4:
                continue

            # IQR method
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            iqr_mask = (df_work[col] < lower) | (df_work[col] > upper)
            iqr_indices = df_work.loc[iqr_mask].index.tolist()

            # Z-score method (|z| > 3)
            mean, std = series.mean(), series.std()
            if std > 0:
                z_scores = np.abs((df_work[col] - mean) / std)
                z_mask = z_scores > 3
                z_indices = df_work.loc[z_mask].index.tolist()
            else:
                z_indices = []

            report['outliers'][col] = {
                'indices_iqr': iqr_indices,
                'indices_zscore': z_indices,
                'count_iqr': len(iqr_indices),
                'count_zscore': len(z_indices),
                'indices': iqr_indices,  # по умолчанию используем IQR
            }

        # 4. Class imbalance (для категориальных колонок)
        report['imbalance'] = {}
        cat_cols = ['category'] if 'category' in df.columns else []
        for col in cat_cols:
            counts = df[col].value_counts(dropna=False)
            dist = counts.to_dict()
            total = counts.sum()
            if total == 0:
                continue
            probs = counts / total
            entropy = -np.sum(probs * np.log2(probs.replace(0, np.nan).fillna(1)))
            max_count = counts.max()
            min_count = counts[counts > 0].min() if (counts > 0).any() else 0
            max_min_ratio = max_count / min_count if min_count > 0 else float('inf')

            report['imbalance'][col] = {
                'distribution': {str(k): int(v) for k, v in dist.items()},
                'entropy': round(float(entropy), 4),
                'max_min_ratio': round(float(max_min_ratio), 2),
                'n_classes': len(counts),
            }

        self._last_report = report
        return report

    def fix(
        self,
        df: pd.DataFrame,
        strategy: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Применяет стратегии очистки к данным.

        Args:
            df: Исходный DataFrame
            strategy: dict с ключами 'missing', 'duplicates', 'outliers'.
                Пример: {'missing': 'median', 'duplicates': 'drop', 'outliers': 'clip_iqr'}

        Returns:
            Очищенный DataFrame
        """
        if strategy is None:
            strategy = {}

        df_clean = df.copy()

        # 1. Duplicates
        dup_strategy = strategy.get('duplicates', 'drop')
        if dup_strategy == 'drop':
            if 'url' in df_clean.columns:
                df_clean = df_clean.drop_duplicates(subset=['url'], keep='first')
            df_clean = df_clean.drop_duplicates(keep='first')

        # 2. Missing values
        miss_strategy = strategy.get('missing')
        if miss_strategy:
            for col in df_clean.columns:
                if col not in df_clean.columns:
                    continue
                null_mask = df_clean[col].isna()
                if df_clean[col].dtype == 'object':
                    empty_mask = df_clean[col].astype(str).str.strip() == ''
                    missing_mask = null_mask | empty_mask
                else:
                    missing_mask = null_mask

                if not missing_mask.any():
                    continue

                if miss_strategy == 'drop':
                    df_clean = df_clean[~missing_mask].copy()
                elif miss_strategy == 'mode':
                    mode_val = df_clean.loc[~missing_mask, col].mode()
                    fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
                    df_clean.loc[missing_mask, col] = fill_val
                elif miss_strategy == 'fill_unknown':
                    df_clean.loc[missing_mask, col] = 'Unknown'
                elif miss_strategy in ('median', 'mean') and np.issubdtype(df_clean[col].dtype, np.number):
                    fill_val = df_clean[col].median() if miss_strategy == 'median' else df_clean[col].mean()
                    df_clean.loc[missing_mask, col] = fill_val

        # 3. Outliers (числовые колонки: clip; производные text_len: drop, т.к. текст нельзя clip'нуть)
        out_strategy = strategy.get('outliers')
        if out_strategy:
            df_work = self._add_derived_numeric(df_clean)
            numeric_cols = list(
                df_work.select_dtypes(include=[np.number]).columns
            ) or [c for c in self.DERIVED_NUMERIC_COLS if c in df_work.columns]
            outlier_indices = set()

            for col in numeric_cols:
                series = df_work[col].dropna()
                if len(series) < 4:
                    continue

                if out_strategy == 'clip_iqr':
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        if col in df_clean.columns:
                            df_clean[col] = df_work[col].clip(lower=lower, upper=upper)
                        else:
                            mask = (df_work[col] < lower) | (df_work[col] > upper)
                            outlier_indices.update(df_work.loc[mask].index.tolist())
                elif out_strategy == 'clip_zscore':
                    mean, std = series.mean(), series.std()
                    if std > 0:
                        z_scores = (df_work[col] - mean) / std
                        if col in df_clean.columns:
                            z_clipped = np.clip(z_scores, -3, 3)
                            df_clean[col] = mean + z_clipped * std
                        else:
                            mask = np.abs(z_scores) > 3
                            outlier_indices.update(df_work.loc[mask].index.tolist())
                elif out_strategy == 'drop':
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    if iqr > 0:
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        mask = (df_work[col] < lower) | (df_work[col] > upper)
                        outlier_indices.update(df_work.loc[mask].index.tolist())

            if outlier_indices:
                df_clean = df_clean.drop(index=list(outlier_indices), errors='ignore')

        return df_clean.reset_index(drop=True)

    def compare(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> pd.DataFrame:
        """
        Сравнивает метрики качества до и после очистки.

        Returns:
            DataFrame с колонками: metric, before, after, change
        """
        def _metrics(d: pd.DataFrame) -> dict:
            m = {
                'n_rows': len(d),
                'n_duplicates_full': d.duplicated().sum(),
                'n_duplicates_url': d['url'].duplicated().sum() if 'url' in d.columns else 0,
            }
            for col in d.columns:
                nulls = d[col].isna().sum()
                if d[col].dtype == 'object':
                    empty = (d[col].astype(str).str.strip() == '').sum()
                    m[f'missing_{col}'] = nulls + empty
                else:
                    m[f'missing_{col}'] = nulls
            # Outliers по text_len
            if 'text' in d.columns:
                tl = d['text'].astype(str).str.len()
                q1, q3 = tl.quantile(0.25), tl.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    m['n_outliers_text_len'] = ((tl < low) | (tl > high)).sum()
                else:
                    m['n_outliers_text_len'] = 0
            return m

        mb = _metrics(df_before)
        ma = _metrics(df_after)
        all_keys = sorted(set(mb.keys()) | set(ma.keys()))
        rows = []
        for k in all_keys:
            b = mb.get(k, 0)
            a = ma.get(k, 0)
            ch = a - b
            rows.append({'metric': k, 'before': b, 'after': a, 'change': ch})
        return pd.DataFrame(rows)

    def explain_and_recommend(
        self,
        report: Dict[str, Any],
        task_description: str,
    ) -> str:
        """
        LLM-скилл: объясняет найденные проблемы и рекомендует стратегию чистки.

        Требует ANTHROPIC_API_KEY в окружении.
        """
        try:
            import anthropic
        except ImportError:
            return (
                "Для использования explain_and_recommend установите: pip install anthropic\n"
                "и задайте переменную окружения ANTHROPIC_API_KEY"
            )

        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            return "ANTHROPIC_API_KEY не задан. Установите ключ в переменных окружения."

        client = anthropic.Anthropic(api_key=api_key)
        prompt = f"""Ты эксперт по качеству данных. Проанализируй отчёт о качестве данных и дай рекомендации.

ОТЧЁТ:
{report}

ЗАДАЧА ML: {task_description}

Дай краткое объяснение найденных проблем (пропуски, дубликаты, выбросы, дисбаланс) и порекомендуй стратегию чистки в формате:
strategy = {{'missing': '...', 'duplicates': '...', 'outliers': '...'}}
С обоснованием выбора."""

        try:
            msg = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            return f"Ошибка вызова API: {e}"
