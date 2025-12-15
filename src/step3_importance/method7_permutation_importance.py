"""
Method 7: Permutation Importance (Перестановочная важность)

Описание:
Измеряет важность признака путем случайного перемешивания его значений 
и наблюдения за падением качества модели. Если при перемешивании признака 
качество сильно падает → признак важен.

Применяется к ВСЕМ признакам (числовым + категориальным после кодирования).

Результат: importance (может быть отрицательной!)
- > 0: перемешивание ухудшает модель (признак важен)
- ≈ 0: перемешивание не влияет (признак бесполезен)
- < 0: перемешивание УЛУЧШАЕТ модель (шум/переобучение)

Преимущества:
- Показывает "реальную" важность для конкретной модели
- Не зависит от внутреннего устройства модели (работает с любой)
- Ловит признаки которые RF importance может пропустить

Недостатки:
- Медленнее чем RF importance (требует много предсказаний)
- Зависит от качества модели
- Может быть нестабильным (используем n_repeats для усреднения)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method7_permutation_importance'
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

# Параметры Permutation Importance
PERM_PARAMS = {
    'n_repeats': 10,      # Сколько раз перемешивать каждый признак
    'random_state': 42,
    'n_jobs': -1,
    'scoring': 'accuracy'  # Метрика для оценки падения качества
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 7: PERMUTATION IMPORTANCE")
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
    
    # Обучение Random Forest
    print("Обучение Random Forest...")
    print(f"Параметры: {RF_PARAMS}\n")
    
    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_train, y_train)
    
    print("Обучение завершено!\n")
    
    # Baseline accuracy на валидации
    baseline_score = rf.score(X_val, y_val)
    print(f"Baseline Accuracy (Val): {baseline_score:.4f}\n")
    
    # Вычисление Permutation Importance
    print("Вычисление Permutation Importance...")
    print(f"Параметры: {PERM_PARAMS}")
    print("(может занять 1-2 минуты)\n")
    
    perm_result = permutation_importance(
        rf, X_val, y_val,
        **PERM_PARAMS
    )
    
    print("Вычисление завершено!\n")
    
    # Извлечение результатов
    print("Результаты:\n")
    
    importance_results = {}
    std_results = {}
    
    for i, feature in enumerate(ALL_FEATURES):
        importance_results[feature] = perm_result.importances_mean[i]
        std_results[feature] = perm_result.importances_std[i]
        feature_type = "числовой" if feature in NUMERIC_FEATURES else "категориальный"
        
        print(f"   {FEATURE_NAMES_RU[feature]:45} | {perm_result.importances_mean[i]:+.6f} ± {perm_result.importances_std[i]:.6f} | ({feature_type})")
    
    print()
    
    # Создание DataFrame с результатами
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in importance_results.keys()],
        'Признак (англ)': list(importance_results.keys()),
        'Тип': ['Числовой' if f in NUMERIC_FEATURES else 'Категориальный' for f in importance_results.keys()],
        'Importance (mean)': list(importance_results.values()),
        'Importance (std)': list(std_results.values())
    })
    
    # Сортировка по Importance (по убыванию)
    result_df = result_df.sort_values('Importance (mean)', ascending=False).reset_index(drop=True)
    result_df.insert(0, 'Ранг', range(1, len(result_df) + 1))
    
    # Сохранение таблицы
    csv_path = tables_dir / 'permutation_importance.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'permutation_importance_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 7 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'permutation_importance_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для Permutation Importance с error bars.
    Цвет зависит от типа признака (числовой/категориальный).
    Положительные - вправо, отрицательные (если есть) - влево.
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Цвет в зависимости от типа признака
    colors = ['#3498db' if t == 'Числовой' else '#e67e22' for t in df['Тип']]
    
    # Bar chart с error bars
    bars = ax.barh(
        range(len(df)),
        df['Importance (mean)'],
        xerr=df['Importance (std)'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1,
        error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3}
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Вертикальная линия на нуле
    ax.axvline(0, color='red', linewidth=2, linestyle='--', alpha=0.7)
    
    # Заголовок и подписи
    ax.set_title('Permutation Importance (с ошибками по 10 повторениям)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Importance (падение Accuracy при перемешивании)', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(-0.015, 0.075)
    # Добавляем значения на столбцах с рангом
    for i, (bar, value, rank, std) in enumerate(zip(bars, df['Importance (mean)'], df['Ранг'], df['Importance (std)'])):
        x_pos = value + (max(df['Importance (mean)']) * 0.02 if value > 0 else -max(df['Importance (mean)']) * 0.02)
        ha = 'left' if value > 0 else 'right'
        ax.text(
            x_pos,
            i,
            f'#{rank}  {value:+.6f}',
            va='center',
            ha=ha,
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