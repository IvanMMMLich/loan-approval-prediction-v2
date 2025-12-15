"""
Функции для визуализации метрик моделей.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Настройка шрифтов для поддержки русского языка
plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """
    Матрица ошибок.
    Сверху: ФАКТ (Actual)
    Слева: ПРЕДСКАЗАНО (Predicted)
    
    Стандартный формат:
    TP (левый верхний) | FP (правый верхний)
    FN (левый нижний)  | TN (правый нижний)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Создаём матрицу в нужном формате:
    # Строки: Predicted (Positive сверху, Negative снизу)
    # Столбцы: Actual (Positive слева, Negative справа)
    matrix = np.array([
        [tp, fp],  # Predicted Positive: TP, FP
        [fn, tn]   # Predicted Negative: FN, TN
    ])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Цвета: зелёный = правильно, красный = ошибка
    colors = np.array([
        ['#90EE90', '#FFB6C1'],  # TP (зелёный), FP (красный)
        ['#FFB6C1', '#90EE90']   # FN (красный), TN (зелёный)
    ])
    
    # Метки ячеек
    labels = [['TP', 'FP'], ['FN', 'TN']]
    
    # Рисуем каждую ячейку
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            color = colors[i, j]
            
            rect = plt.Rectangle((j, 1 - i), 1, 1, facecolor=color, 
                                edgecolor='black', linewidth=3)
            ax.add_patch(rect)
            
            # Число в центре ячейки
            ax.text(j + 0.5, 1.5 - i, f'{value:,}',
                   ha='center', va='center',
                   fontsize=28, fontweight='bold', color='black')
            
            # Метки (TP, FP, FN, TN)
            ax.text(j + 0.5, 1.15 - i, labels[i][j],
                   ha='center', va='top',
                   fontsize=14, fontweight='bold', color='darkblue')
    
    # Настройка осей
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    
    # Верх: ФАКТ (Actual)
    ax.set_xticklabels(['Одобрен (1)\nPositive', 'НЕ одобрен (0)\nNegative'], 
                       fontsize=12, fontweight='bold')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    
    # Слева: ПРЕДСКАЗАНО (Predicted) — ИСПРАВЛЕН ПОРЯДОК
    ax.set_yticklabels(['НЕ одобрен (0)\nNegative', 'Одобрен (1)\nPositive'], 
                       fontsize=12, fontweight='bold')
    
    ax.set_xlabel('ФАКТ (ACTUAL VALUES)', fontsize=14, fontweight='bold')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('ПРЕДСКАЗАНО (PREDICTED VALUES)', fontsize=14, fontweight='bold')
    
    plt.title(f'Матрица ошибок - {model_name}', fontsize=16, fontweight='bold', pad=60)
    
    # Пояснения
    legend_text = (
        f'TP (True Positive) = {tp:,}   Правильно предсказали "Одобрен"\n'
        f'FP (False Positive) = {fp:,}   Ошибочно предсказали "Одобрен" (был НЕ одобрен)\n'
        f'FN (False Negative) = {fn:,}   Ошибочно предсказали "НЕ одобрен" (был Одобрен)\n'
        f'TN (True Negative) = {tn:,}   Правильно предсказали "НЕ одобрен"'
    )
    plt.text(1, -0.25, legend_text, ha='center', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def plot_all_metrics_bars(metrics_dict, model_name="Model", save_path=None):
    """
    Сравнение всех 6 метрик на одном графике.
    """
    metrics_names = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity']
    metrics_names_ru = ['ROC-AUC', 'Точность', 'Precision', 'Recall', 'F1-мера', 'Specificity']
    values = [metrics_dict[m] for m in metrics_names]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#C7CEEA']
    bars = ax.bar(range(len(metrics_names_ru)), values, color=colors, 
                  edgecolor='black', alpha=0.85, linewidth=2)
    
    # Значения над столбцами
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
               f'{value:.4f}', ha='center', va='bottom',
               fontsize=13, fontweight='bold')
    
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Значение метрики', fontsize=13, fontweight='bold')
    ax.set_title(f'Сравнение всех метрик - {model_name}', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(range(len(metrics_names_ru)))
    ax.set_xticklabels(metrics_names_ru, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Линия случайного угадывания
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=2.5, 
              label='Случайное угадывание (0.5)')
    ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, model_name="Model", save_path=None):
    """
    ROC-кривая (Receiver Operating Characteristic).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # ROC-кривая
    ax.plot(fpr, tpr, color='darkorange', lw=3, 
           label=f'ROC-кривая (AUC = {roc_auc:.4f})')
    
    # Диагональ (случайное угадывание)
    ax.plot([0, 1], [0, 1], color='navy', lw=2.5, linestyle='--', 
           label='Случайное угадывание (AUC = 0.5000)')
    
    # Закрашиваем область под кривой
    ax.fill_between(fpr, tpr, alpha=0.2, color='darkorange')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (Доля ложноположительных)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall, Полнота)', 
                 fontsize=13, fontweight='bold')
    ax.set_title(f'ROC-кривая - {model_name}', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Добавляем текст с пояснением
    explanation = (
        'Чем выше кривая и больше площадь под ней (AUC),\n'
        'тем лучше модель различает классы.'
    )
    ax.text(0.98, 0.02, explanation, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_precision_recall_f1(metrics_dict, model_name="Model", save_path=None):
    """
    Визуализация Precision, Recall и F1-Score.
    Только столбчатая диаграмма.
    """
    precision = metrics_dict['Precision']
    recall = metrics_dict['Recall']
    f1 = metrics_dict['F1-Score']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Столбцы
    metrics_names = ['Precision\n(Точность)', 'Recall\n(Полнота)', 'F1-Score\n(Баланс)']
    values = [precision, recall, f1]
    colors_list = ['#45B7D1', '#FFA07A', '#98D8C8']
    
    bars = ax.bar(metrics_names, values, color=colors_list, 
                  edgecolor='black', alpha=0.85, linewidth=2.5, width=0.6)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{value:.4f}', ha='center', va='bottom',
                fontsize=16, fontweight='bold')
    
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Значение метрики', fontsize=14, fontweight='bold')
    ax.set_title(f'Precision, Recall и F1-Score - {model_name}', 
                fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.tick_params(axis='x', labelsize=13)
    
    # Пояснение
    explanation = (
        'Precision: из предсказанных "Одобрен" - сколько правильных\n'
        'Recall: из реальных "Одобрен" - сколько нашли\n'
        'F1-Score: гармоническое среднее Precision и Recall (баланс)'
    )
    ax.text(0.5, -0.18, explanation, ha='center', transform=ax.transAxes,
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()


def plot_all_model_visualizations(y_true, y_pred, y_pred_proba, metrics_dict, 
                                   model_name="Model", save_dir=None):
    """
    Создаёт все 4 графика для модели.
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nСоздание графиков для {model_name}...")
        
        # 1. Матрица ошибок
        cm_path = os.path.join(save_dir, f'{model_name}_01_confusion_matrix.png')
        plot_confusion_matrix(y_true, y_pred, model_name, cm_path)
        print(f"  1/4 Матрица ошибок")
        
        # 2. Все метрики
        all_metrics_path = os.path.join(save_dir, f'{model_name}_02_all_metrics.png')
        plot_all_metrics_bars(metrics_dict, model_name, all_metrics_path)
        print(f"  2/4 Сравнение всех метрик")
        
        # 3. ROC-кривая
        roc_path = os.path.join(save_dir, f'{model_name}_03_roc_curve.png')
        plot_roc_curve(y_true, y_pred_proba, model_name, roc_path)
        print(f"  3/4 ROC-кривая")
        
        # 4. Precision + Recall + F1
        prf_path = os.path.join(save_dir, f'{model_name}_04_precision_recall_f1.png')
        plot_precision_recall_f1(metrics_dict, model_name, prf_path)
        print(f"  4/4 Precision, Recall, F1-Score")
        
        print(f"\nВсе графики сохранены в: {save_dir}")