"""Реэкспорт для контракта: from annotation_agent import AnnotationAgent."""

from agents.annotation_agent import (
    AnnotationAgent,
    AnnotationSpec,
    TASK_LABELS,
    auto_label_op,
)

__all__ = [
    "AnnotationAgent",
    "AnnotationSpec",
    "TASK_LABELS",
    "auto_label_op",
]
