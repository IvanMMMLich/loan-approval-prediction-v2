import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.metrics import calculate_all_metrics


def test_metrics_perfect():
    """Идеальная модель должна давать ROC-AUC = 1.0"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = calculate_all_metrics(y_true, y_pred, y_proba)

    assert metrics['ROC-AUC'] == 1.0
    assert metrics['Accuracy'] == 1.0
    assert metrics['TP'] == 2
    assert metrics['TN'] == 2
    assert metrics['FP'] == 0
    assert metrics['FN'] == 0


def test_metrics_keys():
    """Проверяем что все нужные метрики присутствуют"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_proba = np.array([0.2, 0.8, 0.6, 0.4])

    metrics = calculate_all_metrics(y_true, y_pred, y_proba)

    for key in ['ROC-AUC', 'Accuracy', 'Precision',
                'Recall', 'F1-Score', 'Specificity',
                'TP', 'TN', 'FP', 'FN']:
        assert key in metrics


def test_metrics_range():
    """Все метрики должны быть в диапазоне [0, 1]"""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.3, 0.2, 0.9])

    metrics = calculate_all_metrics(y_true, y_pred, y_proba)

    for key in ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
        assert 0 <= metrics[key] <= 1


def test_ordinal_mappings():
    """Проверяем корректность маппингов Ordinal Encoding"""
    grade_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
    assert grade_mapping['A'] > grade_mapping['G']
    assert len(grade_mapping) == 7


def test_confusion_matrix_values():
    """Проверяем что TP+TN+FP+FN = общее количество примеров"""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.7, 0.3, 0.2, 0.8])

    metrics = calculate_all_metrics(y_true, y_pred, y_proba)

    total = metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN']
    assert total == len(y_true)