"""
Method 2: Point-Biserial Correlation (Точечно-бисериальная корреляция)

Описание:
Измеряет связь между непрерывным (числовым) признаком и дихотомической 
(бинарной) переменной. Это частный случай корреляции Пирсона для бинарного Y.

Применяется к числовым признакам + порядковый loan_grade (закодированный).

Результат: число от -1 до +1
- +1: идеальная прямая связь
- -1: идеальная обратная связь
- 0: нет связи

Формула:
r_pb = (M1 - M0) / S * sqrt(n1 * n0 / (n * (n-1)))
где:
M1 = среднее значение признака для группы Y=1 (одобрено)
M0 = среднее значение признака для группы Y=0 (отклонено)
S = стандартное отклонение признака по всей выборке
n1 = количество Y=1
n0 = количество Y=0
n = n1 + n0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pointbiserialr

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method2_point_biserial'
TARGET_COL = 'loan_status'

# Числовые признаки
NUMERIC_FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

# Порядковый признак (кодируем как качество клиента)
ORDINAL_FEATURE = 'loan_grade'
ORDINAL_MAPPING = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы (лет)',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка (%)',
    'loan_percent_income': 'Процент дохода на кредит (%)',
    'cb_person_cred_hist_length': 'Длина кредитной истории (лет)',
    'loan_grade': 'Грейд кредита'
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 2: POINT-BISERIAL CORRELATION")
    print("="*70 + "\n")
    
    # Создание папок
    tables_dir = RESULTS_DIR / 'tables'
    figures_dir = RESULTS_DIR / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv(DATA_FILE)
    print(f"Загружено: {df.shape[0]:,} строк\n")
    
    # Кодирование грейда
    df['loan_grade_encoded'] = df[ORDINAL_FEATURE].map(ORDINAL_MAPPING)
    
    # Вычисление point-biserial корреляций
    print("Вычисление point-biserial корреляций...\n")
    
    correlations = {}
    p_values = {}
    
    # 1. Числовые признаки
    for feature in NUMERIC_FEATURES:
        corr, p_val = pointbiserialr(df[TARGET_COL], df[feature])
        correlations[feature] = corr
        p_values[feature] = p_val
        
        significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(f"   {FEATURE_NAMES_RU[feature]:45} | r_pb = {corr:+.4f} | p = {p_val:.2e} {significance}")
    
    # 2. Грейд кредита (закодированный)
    corr, p_val = pointbiserialr(df[TARGET_COL], df['loan_grade_encoded'])
    correlations[ORDINAL_FEATURE] = corr
    p_values[ORDINAL_FEATURE] = p_val
    
    significance = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
    print(f"   {FEATURE_NAMES_RU[ORDINAL_FEATURE]:45} | r_pb = {corr:+.4f} | p = {p_val:.2e} {significance}")
    
    print()
    print("   Обозначения: *** p<0.001, ** p<0.01, * p<0.05, ns - не значимо")
    print()
    
    # Создание DataFrame с результатами
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in correlations.keys()],
        'Признак (англ)': list(correlations.keys()),
        'Point-Biserial r': list(correlations.values()),
        'p-value': list(p_values.values()),
        'Значимость': ['***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')) 
                       for p in p_values.values()],
        'Абсолютное значение': [abs(v) for v in correlations.values()]
    })
    
    # Сортировка по абсолютному значению (по убыванию)
    result_df = result_df.sort_values('Абсолютное значение', ascending=False).reset_index(drop=True)
    
    # Сохранение таблицы
    csv_path = tables_dir / 'point_biserial_correlation.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'point_biserial_correlation_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 2 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'point_biserial_correlation_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для point-biserial корреляций.
    Положительные - зелёные, отрицательные - красные.
    Звёздочки обозначают статистическую значимость.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Цвет в зависимости от знака
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['Point-Biserial r']]
    
    # Bar chart
    bars = ax.barh(
        range(len(df)),
        df['Point-Biserial r'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Вертикальная линия на нуле
    ax.axvline(0, color='black', linewidth=1, linestyle='-')
    
    # Заголовок и подписи
    ax.set_title('Point-Biserial Correlation с целевой переменной (loan_status)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Point-Biserial r', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Расширяем ось X чтобы подписи влезли
    ax.set_xlim(-0.55, 0.55)
    
    # Добавляем значения на столбцах со звёздочками значимости
    for i, (bar, value, significance) in enumerate(zip(bars, df['Point-Biserial r'], df['Значимость'])):
        x_pos = value + (0.02 if value > 0 else -0.02)
        ha = 'left' if value > 0 else 'right'
        ax.text(
            x_pos,
            i,
            f'{value:+.4f} {significance}',
            va='center',
            ha=ha,
            fontsize=10,
            fontweight='bold'
        )
    
    # Легенда для звёздочек
    legend_text = "Значимость: *** p<0.001  ** p<0.01  * p<0.05  ns - не значимо"
    ax.text(
        0.5, -0.12,
        legend_text,
        transform=ax.transAxes,
        ha='center',
        fontsize=10,
        style='italic'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен: {save_path}")


if __name__ == '__main__':
    main()