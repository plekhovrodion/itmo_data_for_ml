#!/usr/bin/env python3
"""Этап 2: detect_issues → fix → compare; артефакты в data/processed и reports/."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from agents.data_quality_agent import DataQualityAgent, load_strategy, save_report_json


def main() -> None:
    p = argparse.ArgumentParser(description="Очистка данных (DataQualityAgent)")
    p.add_argument("--input", default="data/raw/merged_price_m2.csv", help="Входной CSV после сбора")
    p.add_argument("--strategy", default="strategy.yaml", help="YAML со стратегией fix")
    p.add_argument("--out", default="data/processed/merged_clean.csv", help="Очищенный CSV")
    p.add_argument("--report-json", default="reports/quality_detect.json")
    p.add_argument("--comparison", default="reports/comparison.csv")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    input_path = root / args.input
    strategy_path = root / args.strategy
    out_path = root / args.out

    df = pd.read_csv(input_path, low_memory=False)
    agent = DataQualityAgent()
    report = agent.detect_issues(df)
    save_report_json(report, root / args.report_json)

    strategy = load_strategy(strategy_path)
    label_col = strategy.get("label_column", "label")
    agent.label_column = label_col
    if "iqr_multiplier" in strategy:
        agent.iqr_multiplier = float(strategy["iqr_multiplier"])

    df_clean = agent.fix(df, strategy)
    comp = agent.compare(df, df_clean)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_path, index=False, encoding="utf-8")
    comp.to_csv(root / args.comparison, index=False, encoding="utf-8")

    print(f"Строк до: {len(df)}, после: {len(df_clean)}")
    print(f"Очищенный файл: {out_path}")
    print(f"detect JSON: {root / args.report_json}")
    print(f"compare CSV: {root / args.comparison}")


if __name__ == "__main__":
    main()
