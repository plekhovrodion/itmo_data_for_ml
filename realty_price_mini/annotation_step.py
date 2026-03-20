#!/usr/bin/env python3
"""Этап 3: auto_label → generate_spec → export_to_labelstudio → review_queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from agents.annotation_agent import AnnotationAgent


def main() -> None:
    p = argparse.ArgumentParser(description="Авторазметка (AnnotationAgent)")
    p.add_argument("--input", default="data/processed/merged_clean.csv")
    p.add_argument("--config", default="annotation_config.yaml")
    p.add_argument("--labeled", default="data/labeled/labeled.csv")
    p.add_argument("--metrics", default="reports/annotation_metrics.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    df = pd.read_csv(root / args.input, low_memory=False)

    agent = AnnotationAgent(modality="tabular", config_path=root / args.config)
    labeled = agent.auto_label(df)
    spec_path = agent.generate_spec(labeled, task="price_segment")
    ls_path = agent.export_to_labelstudio(labeled, root / "labelstudio_import.json")
    rq_path = agent.export_low_confidence(labeled, root / "review_queue.csv")

    metrics = agent.check_quality(labeled)
    out_labeled = root / args.labeled
    out_labeled.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(out_labeled, index=False, encoding="utf-8")

    met_path = root / args.metrics
    met_path.parent.mkdir(parents=True, exist_ok=True)
    met_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Размечено строк: {len(labeled)}")
    print(f"Спецификация: {spec_path}")
    print(f"Label Studio: {ls_path}")
    print(f"Очередь проверки: {rq_path}")
    print(f"Метрики: {met_path}")
    print(f"labeled CSV: {out_labeled}")


if __name__ == "__main__":
    main()
