"""
Method 1: Pearson Correlation (Корреляция Пирсона)

Описание:
Измеряет линейную связь между числовым признаком и бинарным target.
Работает для числовых признаков + порядковых категориальных (loan_grade).

Результат: число от -1 до +1
- +1: идеальная прямая связь (↑ признак → ↑ одобрение)
- -1: идеальная обратная связь (↑ признак → ↓ одобрение)
- 0: нет линейной связи
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method1_pearson'
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

# Порядковый признак (будем кодировать)
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
    print("METHOD 1: PEARSON CORRELATION")
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
    
    # Вычисление корреляций
    print("Вычисление корреляций...\n")
    
    correlations = {}
    
    # 1. Числовые признаки
    for feature in NUMERIC_FEATURES:
        corr = df[feature].corr(df[TARGET_COL])
        correlations[feature] = corr
        print(f"   {FEATURE_NAMES_RU[feature]:45} | {corr:+.4f}")
    
    # 2. loan_grade (ordinal encoding)
    df_temp = df.copy()
    df_temp['loan_grade_encoded'] = df_temp[ORDINAL_FEATURE].map(ORDINAL_MAPPING)
    corr_grade = df_temp['loan_grade_encoded'].corr(df_temp[TARGET_COL])
    correlations[ORDINAL_FEATURE] = corr_grade
    print(f"   {FEATURE_NAMES_RU[ORDINAL_FEATURE]:45} | {corr_grade:+.4f}")
    
    print()
    
    # Создание DataFrame с результатами
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in correlations.keys()],
        'Признак (англ)': list(correlations.keys()),
        'Корреляция Пирсона': list(correlations.values()),
        'Абсолютное значение': [abs(v) for v in correlations.values()]
    })
    
    # Сортировка по абсолютному значению (по убыванию)
    result_df = result_df.sort_values('Абсолютное значение', ascending=False).reset_index(drop=True)
    
    # Сохранение таблицы
    csv_path = tables_dir / 'pearson_correlation.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'pearson_correlation_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 1 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'pearson_correlation_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для корреляций.
    Положительные - зелёные, отрицательные - красные.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Цвет в зависимости от знака
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df['Корреляция Пирсона']]
    
    # Bar chart
    bars = ax.barh(
        range(len(df)),
        df['Корреляция Пирсона'],
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
    ax.set_title('Корреляция Пирсона с целевой переменной (loan_status)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Корреляция Пирсона', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.5, 0.5)
    
    # Добавляем значения на столбцах
    for i, (bar, value) in enumerate(zip(bars, df['Корреляция Пирсона'])):
        x_pos = value + (0.02 if value > 0 else -0.02)
        ha = 'left' if value > 0 else 'right'
        ax.text(
            x_pos,
            i,
            f'{value:+.4f}',
            va='center',
            ha=ha,
            fontsize=10,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=350, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен: {save_path}")


if __name__ == '__main__':
    main()