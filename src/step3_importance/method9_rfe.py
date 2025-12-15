"""
Method 9: Recursive Feature Elimination (Рекурсивное исключение признаков)

Описание:
Итеративно обучает модель, находит наименее важный признак, удаляет его,
и повторяет процесс. В результате каждый признак получает ранг:
- Ранг 1 = самый важный (удаляется последним)
- Ранг 11 = самый слабый (удаляется первым)

КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:
- Грейд кредита: Ordinal (A=7, G=1) — по качеству клиента
- Наличие дефолта: Бинарный (N=1, Y=0) — N лучше
- Владение жильём: Ordinal по Approval Rate ОБРАТНО (RENT=1, OWN=4)
- Цель кредита: Ordinal по Approval Rate ОБРАТНО (DEBTCONSOLIDATION=1, VENTURE=6)

Логика: Выше число = лучше для клиента = РЕЖЕ одобряют (субстандартный кредитор)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method9_rfe'
TARGET_COL = 'loan_status'

# ВСЕ признаки
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

# ============================================================================
# МАППИНГИ ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# Логика: Выше число = лучше для клиента = РЕЖЕ одобряют
# ============================================================================

# Грейд кредита: по качеству клиента (A = лучший, G = худший)
GRADE_MAPPING = {
    'A': 7,  # 4.92% approval  — лучший клиент
    'B': 6,  # 10.23% approval
    'C': 5,  # 13.54% approval
    'D': 4,  # 59.36% approval
    'E': 3,  # 62.54% approval
    'F': 2,  # 61.07% approval
    'G': 1   # 81.82% approval — худший клиент
}

# Наличие дефолта: N = лучше (не было дефолта)
DEFAULT_MAPPING = {
    'N': 1,  # 11.51% approval — лучше (не было дефолта)
    'Y': 0   # 29.89% approval — хуже (был дефолт)
}

# Владение жильём: OWN = лучше (владеет жильём)
HOME_MAPPING = {
    'RENT': 1,      # 22.26% approval — хуже
    'OTHER': 2,     # 16.85% approval
    'MORTGAGE': 3,  # 5.97% approval
    'OWN': 4        # 1.37% approval  — лучше
}

# Цель кредита: по Approval Rate ОБРАТНО
INTENT_MAPPING = {
    'DEBTCONSOLIDATION': 1,  # 18.93% approval — хуже
    'MEDICAL': 2,            # 17.83% approval
    'HOMEIMPROVEMENT': 3,    # 17.37% approval
    'PERSONAL': 4,           # 13.28% approval
    'EDUCATION': 5,          # 10.77% approval
    'VENTURE': 6             # 9.28% approval  — лучше
}

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

# Гиперпараметры Random Forest
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Параметры RFE
RFE_PARAMS = {
    'n_features_to_select': 1,
    'step': 1,
    'verbose': 0
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 9: RECURSIVE FEATURE ELIMINATION (RFE)")
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
    
    # Кодирование категориальных признаков
    print("Кодирование категориальных признаков...")
    print("Логика: Выше число = лучше для клиента = РЕЖЕ одобряют\n")
    df_encoded = df.copy()
    
    # Грейд кредита
    df_encoded['loan_grade'] = df['loan_grade'].map(GRADE_MAPPING)
    print(f"   Грейд кредита: A=7 (лучший) ... G=1 (худший)")
    
    # Наличие дефолта
    df_encoded['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(DEFAULT_MAPPING)
    print(f"   Наличие дефолта: N=1 (лучше), Y=0 (хуже)")
    
    # Владение жильём
    df_encoded['person_home_ownership'] = df['person_home_ownership'].map(HOME_MAPPING)
    print(f"   Владение жильем: OWN=4 (лучше) ... RENT=1 (хуже)")
    
    # Цель кредита
    df_encoded['loan_intent'] = df['loan_intent'].map(INTENT_MAPPING)
    print(f"   Цель кредита: VENTURE=6 (лучше) ... DEBTCONSOLIDATION=1 (хуже)")
    
    print()
    
    # Подготовка данных
    X = df_encoded[ALL_FEATURES]
    y = df_encoded[TARGET_COL]
    
    # Train/Validation Split
    print("Train/Validation Split (80/20, stratified)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"   Train: {X_train.shape[0]:,} строк")
    print(f"   Val:   {X_val.shape[0]:,} строк\n")
    
    # Создание базовой модели для RFE
    print("Создание базовой модели Random Forest для RFE...")
    print(f"Параметры RF: {RF_PARAMS}\n")
    
    base_model = RandomForestClassifier(**RF_PARAMS)
    
    # Запуск RFE
    print("Запуск RFE (рекурсивное исключение признаков)...")
    print(f"Параметры RFE: {RFE_PARAMS}")
    print("(может занять 1-2 минуты)\n")
    
    rfe = RFE(estimator=base_model, **RFE_PARAMS)
    rfe.fit(X_train, y_train)
    
    print("RFE завершен!\n")
    
    # Извлечение рангов
    print("Результаты (ранги признаков):\n")
    print("Интерпретация:")
    print("   Ранг 1 = самый важный (удаляется последним)")
    print(f"   Ранг {len(ALL_FEATURES)} = самый слабый (удаляется первым)\n")
    
    ranks = rfe.ranking_
    
    rank_results = {}
    for i, feature in enumerate(ALL_FEATURES):
        rank_results[feature] = ranks[i]
        feature_type = "числовой" if feature in NUMERIC_FEATURES else "категориальный"
        
        importance_desc = "КРИТИЧЕН" if ranks[i] == 1 else (
            "Важен" if ranks[i] <= 3 else (
                "Средний" if ranks[i] <= 6 else "Слабый"
            )
        )
        
        print(f"   {FEATURE_NAMES_RU[feature]:45} | Ранг {ranks[i]:2d} | {importance_desc:10} | ({feature_type})")
    
    print()
    
    # Создание DataFrame
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in rank_results.keys()],
        'Признак (англ)': list(rank_results.keys()),
        'Тип': ['Числовой' if f in NUMERIC_FEATURES else 'Категориальный' for f in rank_results.keys()],
        'RFE Ранг': list(rank_results.values()),
        'Важность': ['КРИТИЧЕН' if r == 1 else (
            'Важен' if r <= 3 else (
                'Средний' if r <= 6 else 'Слабый'
            )
        ) for r in rank_results.values()]
    })
    
    # Сортировка
    result_df = result_df.sort_values('RFE Ранг', ascending=True).reset_index(drop=True)
    
    # Сохранение таблицы
    csv_path = tables_dir / 'rfe_ranking.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'rfe_ranking_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 9 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'rfe_ranking_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для RFE рангов.
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Инвертируем ранги для визуализации
    max_rank = df['RFE Ранг'].max()
    inverted_ranks = max_rank - df['RFE Ранг'] + 1
    
    # Цвет в зависимости от типа признака
    colors = ['#3498db' if t == 'Числовой' else '#e67e22' for t in df['Тип']]
    
    # Bar chart
    bars = ax.barh(
        range(len(df)),
        inverted_ranks,
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Заголовок
    ax.set_title('RFE Ranking (Recursive Feature Elimination)\n'
                 'Кодировка: выше число = лучше для клиента = реже одобряют', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Важность (инвертированный ранг: выше = важнее)', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцах
    for i, (bar, rank, importance) in enumerate(zip(bars, df['RFE Ранг'], df['Важность'])):
        ax.text(
            inverted_ranks.iloc[i] + max_rank * 0.02,
            i,
            f'Ранг {rank} ({importance})',
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )
    
    # Легенда
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', edgecolor='black', label='Числовой'),
        Patch(facecolor='#e67e22', edgecolor='black', label='Категориальный')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    # Пояснение
    ax.text(
        0.5, -0.10,
        "Ранг 1 = самый важный (удаляется последним), выше ранг = менее важен",
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