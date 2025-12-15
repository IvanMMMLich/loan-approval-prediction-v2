"""
Method 6: Random Forest Feature Importance (Важность признаков через RF)

Описание:
Обучает Random Forest и извлекает важность признаков на основе того, 
насколько часто каждый признак используется для разделения узлов в деревьях.
Важность = среднее уменьшение Gini impurity при использовании признака.

Применяется к ВСЕМ признакам (числовым + категориальным после кодирования).

Результат: importance от 0 до 1 (сумма всех = 1.0)
- 0.30 (30%): признак участвует в 30% важных разделений
- 0.02 (2%): признак почти не используется деревьями

Преимущества:
- Ловит нелинейные зависимости
- Ловит взаимодействия признаков
- Работает с категориальными после кодирования
- Показывает "реальную" важность для tree-based моделей

Недостатки:
- Зависит от гиперпараметров модели
- Может быть нестабильным (используем random_state для воспроизводимости)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method6_rf_importance'
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

# Гиперпараметры Random Forest (простые для baseline)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'class_weight': 'balanced',  # ОБЯЗАТЕЛЬНО для дисбаланса!
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 6: RANDOM FOREST FEATURE IMPORTANCE")
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
    
    # Проверка баланса классов
    train_balance = y_train.value_counts(normalize=True)
    print(f"Баланс классов в Train:")
    print(f"   Класс 0 (отклонено): {train_balance[0]:.2%}")
    print(f"   Класс 1 (одобрено):  {train_balance[1]:.2%}\n")
    
    # Обучение Random Forest
    print("Обучение Random Forest...")
    print(f"Параметры: {RF_PARAMS}\n")
    
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    print("Обучение завершено!\n")
    
    # Оценка качества (для справки)
    train_score = rf.score(X_train, y_train)
    val_score = rf.score(X_val, y_val)
    print(f"Accuracy:")
    print(f"   Train: {train_score:.4f}")
    print(f"   Val:   {val_score:.4f}\n")
    
    # Извлечение Feature Importance
    print("Извлечение Feature Importance...\n")
    
    importances = rf.feature_importances_
    
    # Вывод результатов
    importance_results = {}
    for i, feature in enumerate(ALL_FEATURES):
        importance_results[feature] = importances[i]
        feature_type = "числовой" if feature in NUMERIC_FEATURES else "категориальный"
        print(f"   {FEATURE_NAMES_RU[feature]:45} | {importances[i]:.6f} ({importances[i]*100:5.2f}%) | ({feature_type})")
    
    print()
    print(f"Сумма всех importance: {sum(importances):.6f} (должна быть 1.0)\n")
    
    # Создание DataFrame с результатами
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in importance_results.keys()],
        'Признак (англ)': list(importance_results.keys()),
        'Тип': ['Числовой' if f in NUMERIC_FEATURES else 'Категориальный' for f in importance_results.keys()],
        'Importance': list(importance_results.values()),
        'Importance (%)': [v * 100 for v in importance_results.values()]
    })
    
    # Сортировка по Importance (по убыванию)
    result_df = result_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    result_df.insert(0, 'Ранг', range(1, len(result_df) + 1))
    
    # Сохранение таблицы
    csv_path = tables_dir / 'rf_feature_importance.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'rf_feature_importance_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 6 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'rf_feature_importance_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для RF Feature Importance.
    Цвет зависит от типа признака (числовой/категориальный).
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Цвет в зависимости от типа признака
    colors = ['#3498db' if t == 'Числовой' else '#e67e22' for t in df['Тип']]
    
    # Bar chart
    bars = ax.barh(
        range(len(df)),
        df['Importance'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Заголовок и подписи
    ax.set_title('Random Forest Feature Importance', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Importance', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцах с рангом и процентами
    for i, (bar, value, rank, pct) in enumerate(zip(bars, df['Importance'], df['Ранг'], df['Importance (%)'])):
        ax.text(
            value + max(df['Importance']) * 0.02,
            i,
            f'#{rank}  {pct:.2f}%',
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