"""
Method 19: Permutation Conditional Importance (Условная важность)

Описание:
Измеряет, как важность одного признака зависит от наличия другого.

Метод:
1. Вычисляем базовую важность каждого признака (permutation importance)
2. Для каждой пары (A, B):
   - Перемешиваем признак A (делаем его бесполезным)
   - Заново вычисляем важность признака B
   - Сравниваем: изменилась ли важность B?

Интерпретация:
- Δ > 0 (важность B выросла): A и B дублируют информацию
  → Когда A "сломан", модель больше полагается на B
- Δ ≈ 0 (важность B не изменилась): A и B независимы
- Δ < 0 (важность B упала): A усиливал B (редкий случай)

Применение:
- Поиск дублирующих признаков (кандидаты на удаление)
- Понимание зависимостей между признаками

Результат: Матрица 11×11, где ячейка (A, B) = изменение важности B при удалении A
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'model_based' / 'method19_permutation_conditional'

# Все признаки
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

TARGET = 'loan_status'

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка',
    'loan_percent_income': '% дохода на кредит',
    'cb_person_cred_hist_length': 'Длина кред. истории',
    'person_home_ownership': 'Владение жильём',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта'
}

# Размер выборки
SAMPLE_SIZE = 5000

# ============================================================================
# ФУНКЦИИ
# ============================================================================

def prepare_data(df):
    """Подготавливает данные: кодирует категориальные признаки."""
    
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    return df_encoded, label_encoders


def compute_permutation_importance(model, X, y, n_repeats=5):
    """Вычисляет permutation importance для всех признаков."""
    
    result = permutation_importance(
        model, X, y, 
        n_repeats=n_repeats, 
        random_state=42, 
        n_jobs=-1,
        scoring='roc_auc'
    )
    
    importance_dict = {}
    for i, feature in enumerate(X.columns):
        importance_dict[feature] = result.importances_mean[i]
    
    return importance_dict


def compute_conditional_importance(model, X, y, features, n_repeats=5):
    """
    Вычисляет условную важность: как меняется важность B при "удалении" A.
    
    Матрица[A, B] = важность B после перемешивания A - базовая важность B
    """
    
    n_features = len(features)
    
    # Базовая важность каждого признака
    print("Вычисление базовой важности признаков...")
    base_importance = compute_permutation_importance(model, X, y, n_repeats)
    
    print("\nБазовая важность признаков:")
    for feat in sorted(base_importance.keys(), key=lambda x: base_importance[x], reverse=True):
        print(f"   {FEATURE_NAMES_RU.get(feat, feat):30} | {base_importance[feat]:.4f}")
    
    # Матрица изменений важности
    delta_matrix = np.zeros((n_features, n_features))
    
    total_features = n_features
    
    print("\n" + "="*60)
    print("Вычисление условной важности...")
    print("="*60 + "\n")
    
    for i, feature_to_permute in enumerate(features):
        print(f"Перемешиваем {i+1}/{total_features}: {feature_to_permute}")
        
        # Создаём копию данных с перемешанным признаком
        X_permuted = X.copy()
        X_permuted[feature_to_permute] = np.random.permutation(X_permuted[feature_to_permute].values)
        
        # Вычисляем важность остальных признаков
        new_importance = compute_permutation_importance(model, X_permuted, y, n_repeats=3)
        
        # Записываем изменения
        for j, feature_measured in enumerate(features):
            if i == j:
                # Диагональ: сам с собой не сравниваем
                delta_matrix[i, j] = 0
            else:
                # Δ = новая важность - базовая важность
                delta = new_importance[feature_measured] - base_importance[feature_measured]
                delta_matrix[i, j] = delta
    
    print()
    
    # Создаём DataFrame с русскими названиями
    features_ru = [FEATURE_NAMES_RU.get(f, f) for f in features]
    delta_df = pd.DataFrame(delta_matrix, index=features_ru, columns=features_ru)
    
    # Также создаём DataFrame базовой важности
    base_importance_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU.get(f, f) for f in features],
        'Признак_EN': features,
        'Базовая важность': [base_importance[f] for f in features]
    }).sort_values('Базовая важность', ascending=False).reset_index(drop=True)
    
    return delta_df, base_importance_df


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 19: PERMUTATION CONDITIONAL IMPORTANCE")
    print("="*80)
    print("\nИзмеряем, как важность признака B зависит от наличия признака A.")
    print("Если важность B растёт при 'удалении' A → они дублируют информацию.\n")
    
    # Создание папок
    tables_dir = RESULTS_DIR / 'tables'
    figures_dir = RESULTS_DIR / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv(DATA_FILE)
    print(f"Загружено: {df.shape[0]:,} строк\n")
    
    # Сэмплирование для ускорения
    if len(df) > SAMPLE_SIZE:
        print(f"Сэмплирование до {SAMPLE_SIZE:,} строк...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"После сэмплирования: {len(df):,} строк\n")
    
    # Подготовка данных
    print("Кодирование категориальных признаков...")
    df_encoded, _ = prepare_data(df)
    print("Готово!\n")
    
    X = df_encoded[ALL_FEATURES]
    y = df_encoded[TARGET]
    
    # Train/test split для более честной оценки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train):,} строк")
    print(f"Test:  {len(X_test):,} строк\n")
    
    # Обучение модели
    print("Обучение Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("Модель обучена!\n")
    
    # Вычисление условной важности
    delta_matrix, base_importance = compute_conditional_importance(
        model, X_test, y_test, ALL_FEATURES
    )
    
    # Сохранение матрицы
    matrix_csv_path = tables_dir / 'conditional_importance_delta_matrix_11x11.csv'
    delta_matrix.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица изменений важности сохранена: {matrix_csv_path}")
    
    # Сохранение базовой важности
    base_csv_path = tables_dir / 'base_permutation_importance.csv'
    base_importance.to_csv(base_csv_path, index=False, encoding='utf-8-sig')
    print(f"Базовая важность сохранена: {base_csv_path}")
    
    # Извлечение топ пар (дублирующих)
    print("\nИзвлечение пар с сильным дублированием...\n")
    duplicate_pairs = extract_duplicate_pairs(delta_matrix, n=15)
    
    print("ТОП-15 ПАР С ДУБЛИРОВАНИЕМ ИНФОРМАЦИИ:")
    print("(Δ > 0 означает: при удалении A важность B растёт → они дублируют)\n")
    
    for i, row in duplicate_pairs.iterrows():
        interpretation = get_interpretation(row['Δ важности'])
        print(f"{i+1:2d}. Удалён: {row['Удалённый (A)']:25} → Изменение {row['Измеряемый (B)']:25} | Δ = {row['Δ важности']:+.4f} ({interpretation})")
    print()
    
    # Сохранение топ пар
    pairs_csv_path = tables_dir / 'duplicate_pairs.csv'
    duplicate_pairs.to_csv(pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Пары с дублированием сохранены: {pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'conditional_importance_heatmap.png'
    create_heatmap(delta_matrix, heatmap_path)
    
    # Создание bar chart базовой важности
    print("Создание bar chart базовой важности...")
    bar_path = figures_dir / 'base_importance_bar.png'
    create_importance_bar(base_importance, bar_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    delta_values = delta_matrix.values.copy()
    np.fill_diagonal(delta_values, 0)
    
    # Только верхний треугольник для уникальных пар
    all_deltas = delta_values.flatten()
    all_deltas = all_deltas[all_deltas != 0]
    
    strong_duplicate = (all_deltas > 0.01).sum()
    weak_duplicate = ((all_deltas > 0) & (all_deltas <= 0.01)).sum()
    independent = ((all_deltas >= -0.005) & (all_deltas <= 0.005)).sum()
    
    print(f"\nСильное дублирование (Δ > 0.01):  {strong_duplicate} пар")
    print(f"Слабое дублирование (0 < Δ ≤ 0.01): {weak_duplicate} пар")
    print(f"Независимые пары (|Δ| < 0.005):    {independent} пар")
    
    print(f"\nМаксимальный Δ: {all_deltas.max():+.4f}")
    print(f"Средний Δ:      {all_deltas.mean():+.4f}")
    print(f"Минимальный Δ:  {all_deltas.min():+.4f}")
    
    # Находим самые дублирующие пары
    print("\n" + "-"*60)
    print("ВЫВОД: Кандидаты на удаление (сильное дублирование):")
    print("-"*60)
    
    top_duplicates = duplicate_pairs[duplicate_pairs['Δ важности'] > 0.005].head(5)
    for _, row in top_duplicates.iterrows():
        print(f"   {row['Удалённый (A)']} ↔ {row['Измеряемый (B)']}: дублируют информацию")
    
    print("\n" + "="*80)
    print("METHOD 19 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица Δ:         {matrix_csv_path}")
    print(f"   Базовая важность:  {base_csv_path}")
    print(f"   Дублирующие пары:  {pairs_csv_path}")
    print(f"   Heatmap:           {heatmap_path}")
    print(f"   Bar chart:         {bar_path}")
    print()


def extract_duplicate_pairs(delta_matrix, n=15):
    """Извлекает пары с наибольшим дублированием (Δ > 0)."""
    
    pairs = []
    features = delta_matrix.index.tolist()
    
    for i, feat_removed in enumerate(features):
        for j, feat_measured in enumerate(features):
            if i != j:
                delta = delta_matrix.iloc[i, j]
                pairs.append({
                    'Удалённый (A)': feat_removed,
                    'Измеряемый (B)': feat_measured,
                    'Δ важности': delta,
                    'Интерпретация': get_interpretation(delta)
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('Δ важности', ascending=False).head(n).reset_index(drop=True)
    
    return pairs_df


def get_interpretation(delta):
    """Возвращает интерпретацию изменения важности."""
    if delta > 0.02:
        return "сильное дублирование"
    elif delta > 0.005:
        return "дублирование"
    elif delta > -0.005:
        return "независимы"
    else:
        return "усиление"


def create_heatmap(delta_matrix, save_path):
    """Создаёт heatmap изменений важности."""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Симметричная шкала
    max_abs = max(abs(delta_matrix.values.min()), abs(delta_matrix.values.max()), 0.02)
    
    im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto', vmin=-max_abs, vmax=max_abs)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(delta_matrix.columns)))
    ax.set_yticks(np.arange(len(delta_matrix.index)))
    ax.set_xticklabels(delta_matrix.columns, rotation=45, ha='right', fontsize=10, weight='bold')
    ax.set_yticklabels(delta_matrix.index, fontsize=10, weight='bold')
    
    # Подписи осей
    ax.set_ylabel('Удалённый признак (A)', fontsize=12, weight='bold')
    ax.set_xlabel('Измеряемый признак (B)', fontsize=12, weight='bold')
    
    # Ручное добавление текста
    for i in range(len(delta_matrix.index)):
        for j in range(len(delta_matrix.columns)):
            value = delta_matrix.iloc[i, j]
            text_color = 'white' if abs(value) > max_abs * 0.5 else 'black'
            ax.text(j, i, f'{value:+.3f}', ha='center', va='center',
                   color=text_color, fontsize=7, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(delta_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(delta_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Δ важности B при удалении A\n(красный = дублирование, синий = усиление)', 
                   rotation=270, labelpad=25, fontsize=11, weight='bold')
    
    # Заголовок
    ax.set_title('Conditional Permutation Importance\n(как меняется важность B при "удалении" A)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


def create_importance_bar(importance_df, save_path):
    """Создаёт bar chart базовой важности."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance_df)))[::-1]
    
    bars = ax.barh(importance_df['Признак'], importance_df['Базовая важность'], color=colors)
    
    ax.set_xlabel('Permutation Importance (ROC-AUC drop)', fontsize=12, weight='bold')
    ax.set_title('Базовая важность признаков\n(на сколько падает ROC-AUC при перемешивании признака)', 
                fontsize=14, fontweight='bold')
    
    # Добавляем значения на бары
    for bar, val in zip(bars, importance_df['Базовая важность']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
               f'{val:.4f}', va='center', fontsize=10, weight='bold')
    
    ax.invert_yaxis()
    ax.set_xlim(0, importance_df['Базовая важность'].max() * 1.15)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Bar chart сохранен: {save_path}")


if __name__ == '__main__':
    main()
