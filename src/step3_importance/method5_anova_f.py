"""
Method 5: ANOVA F-statistic (Дисперсионный анализ)

Описание:
Проверяет различаются ли средние значения числового признака между 
группами целевой переменной (loan_status: 0 vs 1). Основан на сравнении 
межгрупповой и внутригрупповой дисперсии.

Применяется только к числовым признакам vs бинарный/категориальный target.

Результат:
- F-статистика: отношение межгрупповой дисперсии к внутригрупповой
  Чем больше F, тем сильнее различаются группы
- p-value: значимость различий (p < 0.05 = различия значимы)

Интерпретация:
- Высокий F + низкий p-value = признак хорошо разделяет классы
- Низкий F + высокий p-value = признак не разделяет классы

Связь с другими методами:
- Для бинарного target, ANOVA F эквивалентно квадрату Point-Biserial r
- ANOVA подходит для multi-class, Point-Biserial только для binary
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import f_classif

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method5_anova_f'
TARGET_COL = 'loan_status'

# Только числовые признаки
NUMERIC_FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы (лет)',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка (%)',
    'loan_percent_income': 'Процент дохода на кредит (%)',
    'cb_person_cred_hist_length': 'Длина кредитной истории (лет)'
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 5: ANOVA F-STATISTIC")
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
    
    # Подготовка данных
    X = df[NUMERIC_FEATURES]
    y = df[TARGET_COL]
    
    # Вычисление ANOVA F-статистики
    print("Вычисление ANOVA F-статистики...\n")
    
    f_scores, p_values = f_classif(X, y)
    
    # Вывод результатов
    print("Результаты:\n")
    
    f_results = {}
    p_val_results = {}
    
    for i, feature in enumerate(NUMERIC_FEATURES):
        f_results[feature] = f_scores[i]
        p_val_results[feature] = p_values[i]
        
        # Показываем значимость
        significance = "***" if p_values[i] < 0.001 else ("**" if p_values[i] < 0.01 else ("*" if p_values[i] < 0.05 else "ns"))
        print(f"   {FEATURE_NAMES_RU[feature]:45} | F = {f_scores[i]:10.2f} | p-value = {p_values[i]:.4e} {significance}")
    
    print()
    print("   Обозначения: *** p<0.001, ** p<0.01, * p<0.05, ns - не значимо")
    print()
    
    # Создание DataFrame с результатами
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in f_results.keys()],
        'Признак (англ)': list(f_results.keys()),
        'F-статистика': list(f_results.values()),
        'p-value': list(p_val_results.values()),
        'Значимость': ['***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns')) 
                       for p in p_val_results.values()]
    })
    
    # Сортировка по F-статистике (по убыванию)
    result_df = result_df.sort_values('F-статистика', ascending=False).reset_index(drop=True)
    result_df.insert(0, 'Ранг', range(1, len(result_df) + 1))
    
    # Сохранение таблицы
    csv_path = tables_dir / 'anova_f_statistic.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'anova_f_statistic_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 5 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'anova_f_statistic_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для ANOVA F-статистики.
    Все столбцы фиолетовые (F всегда >= 0).
    Звёздочки обозначают статистическую значимость.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Все столбцы фиолетовые (F всегда >= 0)
    bars = ax.barh(
        range(len(df)),
        df['F-статистика'],
        color='#9b59b6',
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Заголовок и подписи
    ax.set_title('ANOVA F-статистика для числовых признаков', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('F-статистика', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцах со звёздочками значимости
    for i, (bar, value, rank, significance) in enumerate(zip(bars, df['F-статистика'], df['Ранг'], df['Значимость'])):
        ax.text(
            value + max(df['F-статистика']) * 0.02,
            i,
            f'#{rank}  {value:.2f} {significance}',
            va='center',
            ha='left',
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен: {save_path}")


if __name__ == '__main__':
    main()