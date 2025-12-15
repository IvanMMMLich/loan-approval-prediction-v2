"""
Универсальная функция для расчёта метрик.
Используется на ВСЕХ этапах для единообразия.
"""

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import pandas as pd



def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """
    Рассчитывает все метрики для бинарной классификации.
    
    Parameters
    ----------
    y_true : array-like
        Истинные метки (0 или 1)
    y_pred : array-like
        Предсказанные метки (0 или 1)
    y_pred_proba : array-like
        Предсказанные вероятности класса 1
    
    Returns
    -------
    dict
        Словарь со всеми метриками
    """
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Основные метрики
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Специфичность
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'ROC-AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn)
    }


def print_metrics(metrics, title="Model Metrics"):
    """
    Красиво выводит метрики.
    
    Parameters
    ----------
    metrics : dict
        Словарь с метриками из calculate_all_metrics()
    title : str
        Заголовок
    """
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)
    
    print(f"ROC-AUC:      {metrics['ROC-AUC']:.4f}")
    print(f"Accuracy:     {metrics['Accuracy']:.4f}")
    print(f"Precision:    {metrics['Precision']:.4f}")
    print(f"Recall:       {metrics['Recall']:.4f}")
    print(f"F1-Score:     {metrics['F1-Score']:.4f}")
    print(f"Specificity:  {metrics['Specificity']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TN: {metrics['TN']:5d}  |  FP: {metrics['FP']:5d}")
    print(f"  FN: {metrics['FN']:5d}  |  TP: {metrics['TP']:5d}")
    print("="*60)


def save_metrics(metrics, filepath):
    """
    Сохраняет метрики в CSV файл.
    
    Parameters
    ----------
    metrics : dict
        Словарь с метриками
    filepath : str
        Путь для сохранения CSV
    """
    df = pd.DataFrame([metrics])
    df.to_csv(filepath, index=False)
    print(f"\nMetrics saved to: {filepath}")
