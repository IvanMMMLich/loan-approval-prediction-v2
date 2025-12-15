"""
Method 4: Mutual Information (Взаимная информация)

Описание:
Измеряет количество информации о целевой переменной, которое содержится 
в признаке. В отличие от корреляции Пирсона, Mutual Information ловит 
ЛЮБЫЕ зависимости (линейные, нелинейные, сложные взаимодействия).

Применяется к ВСЕМ признакам (числовым + категориальным после кодирования).

Результат: число >= 0
- 0: признак не содержит информации о target (независим)
- > 0: чем больше, тем больше информации
- Нет верхней границы (зависит от данных)

Преимущества:
- Ловит нелинейные зависимости (U-образные, пороговые и т.д.)
- Работает с категориальными признаками
- Не требует предположений о распределении

Недостатки:
- Медленнее чем корреляция
- Сложнее интерпретировать абсолютное значение
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method4_mutual_information'
TARGET_COL = 'loan_status'

# ВСЕ признаки (числовые + категориальные)
NUMERIC_FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы (лет)',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка (%)',
    'loan_percent_income': 'Процент дохода на кредит (%)',
    'cb_person_cred_hist_length': 'Длина кредитной истории (лет)',
    'person_home_ownership': 'Владение жильем',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта в истории'
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 4: MUTUAL INFORMATION")
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
    
    # Временное кодирование категориальных признаков
    print("Кодирование категориальных признаков...")
    df_encoded = df.copy()
    
    for feature in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_encoded[feature] = le.fit_transform(df[feature])
        n_categories = df[feature].nunique()
        print(f"   {FEATURE_NAMES_RU[feature]:45} | {n_categories} категорий")
    
    print()
    
    # Подготовка данных
    X = df_encoded[ALL_FEATURES]
    y = df_encoded[TARGET_COL]
    
    # Вычисление Mutual Information
    print("Вычисление Mutual Information...")
    print("(может занять некоторое время для больших датасетов)\n")
    
    mi_scores = mutual_info_classif(
        X, y,
        discrete_features=[False]*len(NUMERIC_FEATURES) + [True]*len(CATEGORICAL_FEATURES),
        random_state=42,
        n_neighbors=3
    )
    
    # Вывод результатов
    print("Результаты:\n")
    
    mi_results = {}
    for i, feature in enumerate(ALL_FEATURES):
        mi_results[feature] = mi_scores[i]
        feature_type = "числовой" if feature in NUMERIC_FEATURES else "категориальный"
        print(f"   {FEATURE_NAMES_RU[feature]:45} | MI = {mi_scores[i]:.6f} | ({feature_type})")
    
    print()
    
    # Создание DataFrame с результатами
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in mi_results.keys()],
        'Признак (англ)': list(mi_results.keys()),
        'Тип': ['Числовой' if f in NUMERIC_FEATURES else 'Категориальный' for f in mi_results.keys()],
        'Mutual Information': list(mi_results.values())
    })
    
    # Сортировка по MI (по убыванию)
    result_df = result_df.sort_values('Mutual Information', ascending=False).reset_index(drop=True)
    result_df.insert(0, 'Ранг', range(1, len(result_df) + 1))
    
    # Сохранение таблицы
    csv_path = tables_dir / 'mutual_information.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'mutual_information_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 4 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'mutual_information_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для Mutual Information.
    Цвет зависит от типа признака (числовой/категориальный).
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Цвет в зависимости от типа признака
    colors = ['#3498db' if t == 'Числовой' else '#e67e22' for t in df['Тип']]
    
    # Bar chart
    bars = ax.barh(
        range(len(df)),
        df['Mutual Information'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Заголовок и подписи
    ax.set_title('Mutual Information для всех признаков', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Mutual Information (bits)', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцах с рангом
    for i, (bar, value, rank) in enumerate(zip(bars, df['Mutual Information'], df['Ранг'])):
        ax.text(
            value + max(df['Mutual Information']) * 0.02,
            i,
            f'#{rank}  {value:.6f}',
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )
    
    # Легенда для типов признаков
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Числовой'),
        Patch(facecolor='#e67e22', edgecolor='black', label='Категориальный')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"График сохранен: {save_path}")


if __name__ == '__main__':
    main()