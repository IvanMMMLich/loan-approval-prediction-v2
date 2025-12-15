"""
Method 8: Logistic Regression Coefficients (Коэффициенты логистической регрессии)

Описание:
Обучает Logistic Regression и извлекает веса (коэффициенты) для каждого признака.
Коэффициент показывает насколько изменение признака на 1 единицу влияет на 
log-odds одобрения кредита.

КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:
- Грейд кредита: Ordinal (A=1, G=7) — по уровню риска
- Наличие дефолта: Бинарный (N=0, Y=1)
- Владение жильём: Ordinal по Approval Rate (OWN=1, RENT=4)
- Цель кредита: Ordinal по Approval Rate (VENTURE=1, DEBTCONSOLIDATION=6)

Результат: коэффициент (вес)
- Положительный: увеличение признака → увеличение вероятности одобрения
- Отрицательный: увеличение признака → уменьшение вероятности одобрения
- Величина |вес|: сила влияния
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'method8_logreg_coefficients'
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
# ============================================================================

# Грейд кредита: по уровню риска (A = низкий риск, G = высокий риск)
GRADE_MAPPING = {
    'A': 7,  # 4.92% approval
    'B': 6,  # 10.23% approval
    'C': 5,  # 13.54% approval
    'D': 4,  # 59.36% approval
    'E': 3,  # 62.54% approval
    'F': 2,  # 61.07% approval
    'G': 1   # 81.82% approval
}

# Наличие дефолта: бинарный
DEFAULT_MAPPING = {
    'N': 1,  # 11.51% approval — не было дефолта
    'Y': 0   # 29.89% approval — был дефолт
}

# Владение жильём: по Approval Rate (от низкого к высокому)
HOME_MAPPING = {
    'OWN': 4,       # 1.37% approval
    'MORTGAGE': 3,  # 5.97% approval
    'OTHER': 2,     # 16.85% approval
    'RENT': 1       # 22.26% approval
}

# Цель кредита: по Approval Rate (от низкого к высокому)
INTENT_MAPPING = {
    'VENTURE': 6,           # 9.28% approval
    'EDUCATION': 5,         # 10.77% approval
    'PERSONAL': 4,          # 13.28% approval
    'HOMEIMPROVEMENT': 3,   # 17.37% approval
    'MEDICAL': 2,           # 17.83% approval
    'DEBTCONSOLIDATION': 1  # 18.93% approval
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

# Гиперпараметры Logistic Regression
LOGREG_PARAMS = {
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': 42,
    'solver': 'lbfgs'
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("METHOD 8: LOGISTIC REGRESSION COEFFICIENTS")
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
    print("Кодирование категориальных признаков (Ordinal по Approval Rate)...\n")
    df_encoded = df.copy()
    
    # Грейд кредита
    df_encoded['loan_grade'] = df['loan_grade'].map(GRADE_MAPPING)
    print(f"   Грейд кредита: A=1 (4.92%) ... G=7 (81.82%)")
    
    # Наличие дефолта
    df_encoded['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(DEFAULT_MAPPING)
    print(f"   Наличие дефолта: N=0 (11.51%), Y=1 (29.89%)")
    
    # Владение жильём
    df_encoded['person_home_ownership'] = df['person_home_ownership'].map(HOME_MAPPING)
    print(f"   Владение жильем: OWN=1 (1.37%) ... RENT=4 (22.26%)")
    
    # Цель кредита
    df_encoded['loan_intent'] = df['loan_intent'].map(INTENT_MAPPING)
    print(f"   Цель кредита: VENTURE=1 (9.28%) ... DEBTCONSOLIDATION=6 (18.93%)")
    
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
    
    # StandardScaler
    print("Масштабирование признаков (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print("   Все признаки приведены к mean=0, std=1\n")
    
    # Обучение Logistic Regression
    print("Обучение Logistic Regression...")
    print(f"Параметры: {LOGREG_PARAMS}\n")
    
    logreg = LogisticRegression(**LOGREG_PARAMS)
    logreg.fit(X_train_scaled, y_train)
    
    print("Обучение завершено!\n")
    
    # Оценка качества
    train_score = logreg.score(X_train_scaled, y_train)
    val_score = logreg.score(X_val_scaled, y_val)
    print(f"Accuracy:")
    print(f"   Train: {train_score:.4f}")
    print(f"   Val:   {val_score:.4f}\n")
    
    # Извлечение коэффициентов
    print("Извлечение коэффициентов...\n")
    
    coefficients = logreg.coef_[0]
    
    print("Результаты:\n")
    
    coef_results = {}
    abs_coef_results = {}
    
    for i, feature in enumerate(ALL_FEATURES):
        coef_results[feature] = coefficients[i]
        abs_coef_results[feature] = abs(coefficients[i])
        feature_type = "числовой" if feature in NUMERIC_FEATURES else "категориальный"
        
        direction = "↑" if coefficients[i] > 0 else "↓"
        print(f"   {FEATURE_NAMES_RU[feature]:45} | {coefficients[i]:+.4f} | {direction} | ({feature_type})")
    
    print()
    
    # Создание DataFrame
    result_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU[f] for f in coef_results.keys()],
        'Признак (англ)': list(coef_results.keys()),
        'Тип': ['Числовой' if f in NUMERIC_FEATURES else 'Категориальный' for f in coef_results.keys()],
        'Коэффициент': list(coef_results.values()),
        'Абсолютное значение': list(abs_coef_results.values()),
        'Направление': ['↑' if c > 0 else '↓' for c in coef_results.values()]
    })
    
    # Сортировка
    result_df = result_df.sort_values('Абсолютное значение', ascending=False).reset_index(drop=True)
    result_df.insert(0, 'Ранг', range(1, len(result_df) + 1))
    
    # Сохранение таблицы
    csv_path = tables_dir / 'logreg_coefficients.csv'
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Таблица сохранена: {csv_path}\n")
    
    # Создание графика
    print("Создание графика...")
    create_bar_chart(result_df, figures_dir / 'logreg_coefficients_bar.png')
    
    print("\n" + "="*70)
    print("METHOD 8 ЗАВЕРШЕН")
    print("="*70 + "\n")
    
    print(f"Результаты:")
    print(f"   Таблица: {csv_path}")
    print(f"   График: {figures_dir / 'logreg_coefficients_bar.png'}")
    print()


def create_bar_chart(df, save_path):
    """
    Создаёт bar chart для LogReg коэффициентов.
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Цвет в зависимости от типа признака
    colors = ['#3498db' if t == 'Числовой' else '#e67e22' for t in df['Тип']]
    
    # Bar chart
    bars = ax.barh(
        range(len(df)),
        df['Абсолютное значение'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Признак'], fontsize=11)
    
    # Заголовок и подписи
    ax.set_title('Logistic Regression Coefficients (абсолютные значения)\n'
                 'Категориальные признаки закодированы по Approval Rate', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('|Коэффициент| (важность признака)', fontsize=14)
    ax.set_ylabel('Признак', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцах
    for i, (bar, value, rank, direction) in enumerate(zip(
        bars, 
        df['Абсолютное значение'], 
        df['Ранг'], 
        df['Направление']
    )):
        ax.text(
            value + max(df['Абсолютное значение']) * 0.02,
            i,
            f'#{rank}  {value:.4f} {direction}',
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
        "Направление: ↑ = увеличение признака увеличивает одобрение, ↓ = уменьшает",
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
