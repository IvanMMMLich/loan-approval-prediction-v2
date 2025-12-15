"""
Method 3: Chi-Square Test + Cramér's V (Критерий хи-квадрат + V Крамера)

Описание:
Проверяет статистическую независимость между категориальным признаком 
и целевой переменной.

Chi-Square (χ²): показывает НАЛИЧИЕ связи (зависит от размера выборки и категорий)
Cramér's V: показывает СИЛУ связи (от 0 до 1, можно сравнивать между признаками)

Применяется только к категориальным признакам.

Интерпретация Cramér's V:
- 0.00-0.10: нет связи
- 0.10-0.20: слабая связь
- 0.20-0.40: средняя связь
- 0.40-0.60: сильная связь
- 0.60-1.00: очень сильная связь
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method3_chi_square'
TARGET_COL = 'loan_status'

# Только категориальные признаки
CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_home_ownership': 'Владение жильем',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта в истории'
}

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def calculate_cramers_v(contingency_table):
    """
    Рассчитывает Cramér's V из таблицы сопряжённости.
    
    Формула: V = sqrt(χ² / (n × min(r-1, c-1)))
    где:
    - χ² = Chi-Square статистика
    - n = размер выборки
    - r = количество строк
    - c = количество столбцов
    """
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    
    # Защита от деления на ноль
    if min_dim == 0:
        return 0, chi2, p
    
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    return cramers_v, chi2, p

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 3: CHI-SQUARE TEST + CRAMER'S V")
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
    
    # Вычисление Chi-Square и Cramér's V
    print("Вычисление Chi-Square и Cramér's V...\n")
    
    results = []
    
    for feature in CATEGORICAL_FEATURES:
        # Таблица сопряжённости
        contingency_table = pd.crosstab(df[feature], df[TARGET_COL])
        n_categories = df[feature].nunique()
        
        # Расчёт метрик
        cramers_v, chi2, p_value = calculate_cramers_v(contingency_table)
        
        # Значимость
        significance = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
        
        results.append({
            'Признак': FEATURE_NAMES_RU[feature],
            'Признак (англ)': feature,
            'Категорий': n_categories,
            'Chi-Square': chi2,
            'Cramers V': cramers_v,
            'p-value': p_value,
            'Значимость': significance
        })
        
        print(f"   {FEATURE_NAMES_RU[feature]:35} | категорий: {n_categories} | χ² = {chi2:10.2f} | V = {cramers_v:.4f} | {significance}")
    
    print()
    print("   Cramér's V: 0.00-0.10 нет связи | 0.10-0.20 слабая | 0.20-0.40 средняя | 0.40+ сильная")
    print()
    
    # Создание DataFrame
    result_df = pd.DataFrame(results)
    
    # Сортировка по Cramér's V (по убыванию)
    result_df = result_df.sort_values('Cramers V', ascending=False).reset_index(drop=True)
    
    # Сохранение таблицы
    csv_path = tables_dir / 'chi_square_cramers_v.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графиков
    print("Создание графиков...")
    create_cramers_v_chart(result_df, figures_dir / 'cramers_v_bar.png')
    create_chi_square_chart(result_df, figures_dir / 'chi_square_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 3 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График Cramér's V: {figures_dir / 'cramers_v_bar.png'}")
    print(f"   График Chi-Square: {figures_dir / 'chi_square_bar.png'}")
    print()


def create_cramers_v_chart(df, save_path):
    """
    Создаёт bar chart для Cramér's V (нормализованная метрика, можно сравнивать).
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Цвет в зависимости от силы связи
    def get_color(v):
        if v >= 0.40:
            return '#27ae60'  # Зелёный - сильная
        elif v >= 0.20:
            return '#f39c12'  # Оранжевый - средняя
        elif v >= 0.10:
            return '#e74c3c'  # Красный - слабая
        else:
            return '#95a5a6'  # Серый - нет связи
    
    colors = [get_color(v) for v in df['Cramers V']]
    
    bars = ax.barh(
        range(len(df)),
        df['Cramers V'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Заголовок и подписи
    ax.set_title("Cramér's V для категориальных признаков\n(нормализованная мера — можно сравнивать)", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Cramér's V", fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 0.5)
    
    # Добавляем значения на столбцах
    for i, (bar, value, significance) in enumerate(zip(bars, df['Cramers V'], df['Значимость'])):
        ax.text(
            value + 0.01,
            i,
            f'{value:.4f} {significance}',
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )
    
    # Легенда
    legend_text = "Сила связи: серый <0.10 нет | красный 0.10-0.20 слабая | оранжевый 0.20-0.40 средняя | зелёный >0.40 сильная"
    ax.text(
        0.5, -0.12,
        legend_text,
        transform=ax.transAxes,
        ha='center',
        fontsize=9,
        style='italic'
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"   График Cramér's V сохранен: {save_path}")


def create_chi_square_chart(df, save_path):
    """
    Создаёт bar chart для Chi-Square (для справки, сравнивать нельзя).
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bars = ax.barh(
        range(len(df)),
        df['Chi-Square'],
        color='#3498db',
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Заголовок и подписи
    ax.set_title('Chi-Square Test для категориальных признаков\n(НЕ нормализован — сравнивать нельзя, зависит от числа категорий)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Chi-Square статистика', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцах
    for i, (bar, value, significance, n_cat) in enumerate(zip(bars, df['Chi-Square'], df['Значимость'], df['Категорий'])):
        ax.text(
            value + max(df['Chi-Square']) * 0.02,
            i,
            f'{value:.2f} {significance} ({n_cat} кат.)',
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold'
        )
    
    # Легенда
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
    
    print(f"   График Chi-Square сохранен: {save_path}")


if __name__ == '__main__':
    main()
