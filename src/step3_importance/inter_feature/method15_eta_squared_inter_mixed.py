"""
Method 15: Eta-squared (η²) - Связь между числовыми и категориальными признаками

Описание:
Вычисляет Eta-squared для всех пар (категориальный → числовой).
Показывает, какую долю дисперсии числового признака объясняет категориальный.

Применяется: 4 категориальных × 7 числовых = 28 пар
Результат: Матрица 4×7, heatmap, таблица всех пар

Формула:
η² = SS_between / SS_total
где:
- SS_between - сумма квадратов между группами
- SS_total - общая сумма квадратов

Интерпретация:
- η² = 0: категория не объясняет числовой признак
- η² = 0.01-0.06: малый эффект
- η² = 0.06-0.14: средний эффект
- η² > 0.14: большой эффект
- η² = 1: категория полностью объясняет числовой признак

Примечание:
- Диапазон от 0 до 1
- Направленная мера: категориальный → числовой
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'mixed_correlations' / 'method15_eta_squared'

# Категориальные признаки
CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

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

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_home_ownership': 'Владение жильём',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта',
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка',
    'loan_percent_income': '% дохода на кредит',
    'cb_person_cred_hist_length': 'Длина кред. истории'
}

# ============================================================================
# ФУНКЦИИ ДЛЯ ETA-SQUARED
# ============================================================================

def eta_squared(categorical, numeric):
    """
    Вычисляет Eta-squared (η²) для пары категориальный → числовой.
    
    η² = SS_between / SS_total
    """
    # Общее среднее
    grand_mean = numeric.mean()
    
    # Общая сумма квадратов (SS_total)
    ss_total = ((numeric - grand_mean) ** 2).sum()
    
    if ss_total == 0:
        return 0.0
    
    # Сумма квадратов между группами (SS_between)
    ss_between = 0
    
    for category in categorical.unique():
        mask = (categorical == category)
        group = numeric[mask]
        group_mean = group.mean()
        group_size = len(group)
        
        ss_between += group_size * ((group_mean - grand_mean) ** 2)
    
    # Eta-squared
    eta_sq = ss_between / ss_total
    
    return eta_sq


def compute_eta_squared_matrix(df, categorical_features, numeric_features):
    """
    Вычисляет матрицу Eta-squared для всех пар (категориальный × числовой).
    """
    n_cat = len(categorical_features)
    n_num = len(numeric_features)
    
    # Инициализация матрицы
    eta_matrix = np.zeros((n_cat, n_num))
    
    total_pairs = n_cat * n_num
    current_pair = 0
    
    for i, cat_feat in enumerate(categorical_features):
        for j, num_feat in enumerate(numeric_features):
            current_pair += 1
            print(f"   Пара {current_pair}/{total_pairs}: {cat_feat} -> {num_feat}")
            
            eta_sq = eta_squared(df[cat_feat], df[num_feat])
            eta_matrix[i, j] = eta_sq
    
    return pd.DataFrame(
        eta_matrix, 
        index=categorical_features, 
        columns=numeric_features
    )


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 15: ETA-SQUARED (КАТЕГОРИАЛЬНЫЕ -> ЧИСЛОВЫЕ)")
    print("="*80 + "\n")
    
    # Создание папок
    tables_dir = RESULTS_DIR / 'tables'
    figures_dir = RESULTS_DIR / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv(DATA_FILE)
    print(f"Загружено: {df.shape[0]:,} строк\n")
    
    print(f"Категориальных признаков: {len(CATEGORICAL_FEATURES)}")
    print(f"Числовых признаков: {len(NUMERIC_FEATURES)}")
    print(f"Количество пар: {len(CATEGORICAL_FEATURES) * len(NUMERIC_FEATURES)}\n")
    
    # Вычисление матрицы Eta-squared
    print("Вычисление матрицы Eta-squared...\n")
    
    eta_matrix = compute_eta_squared_matrix(df, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    
    # Переименовываем индексы и колонки на русский
    eta_matrix.index = [FEATURE_NAMES_RU[col] for col in eta_matrix.index]
    eta_matrix.columns = [FEATURE_NAMES_RU[col] for col in eta_matrix.columns]
    
    print("\nГотово!\n")
    
    # Сохранение полной матрицы
    matrix_csv_path = tables_dir / 'eta_squared_matrix_4x7.csv'
    eta_matrix.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица Eta-squared сохранена: {matrix_csv_path}\n")
    
    # Извлечение всех пар
    print("Извлечение всех пар...\n")
    all_pairs = extract_all_pairs(eta_matrix)
    
    # Вывод топ-10 в консоль
    print("ТОП-10 СИЛЬНЕЙШИХ СВЯЗЕЙ (Eta-squared):\n")
    print("Читается как: 'Категориальный признак объясняет X% дисперсии числового'\n")
    for i, row in all_pairs.head(10).iterrows():
        effect = get_effect_size(row['η²'])
        pct = row['η²'] * 100
        print(f"{i+1:2d}. {row['Категориальный']:20} -> {row['Числовой']:20} | η² = {row['η²']:.4f} ({pct:.1f}%) - {effect}")
    print()
    
    # Сохранение всех пар
    all_pairs_csv_path = tables_dir / 'eta_squared_all_pairs.csv'
    all_pairs.to_csv(all_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Все пары сохранены: {all_pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'eta_squared_heatmap.png'
    create_heatmap(eta_matrix, heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    eta_values = eta_matrix.values.flatten()
    
    large = (eta_values > 0.14).sum()
    medium = ((eta_values > 0.06) & (eta_values <= 0.14)).sum()
    small = ((eta_values > 0.01) & (eta_values <= 0.06)).sum()
    negligible = (eta_values <= 0.01).sum()
    
    print(f"\nБольшой эффект (η² > 0.14):     {large} пар")
    print(f"Средний эффект (0.06 < η² <= 0.14): {medium} пар")
    print(f"Малый эффект (0.01 < η² <= 0.06):  {small} пар")
    print(f"Незначимый (η² <= 0.01):       {negligible} пар")
    
    # Максимальная связь
    max_idx = eta_matrix.stack().idxmax()
    max_value = eta_matrix.loc[max_idx]
    print(f"\nСамая сильная связь:")
    print(f"   {max_idx[0]} -> {max_idx[1]}")
    print(f"   η² = {max_value:.4f} ({max_value*100:.1f}% дисперсии)")
    
    # Анализ по категориальным признакам
    print("\n" + "-"*40)
    print("СРЕДНИЙ η² ПО КАТЕГОРИАЛЬНЫМ ПРИЗНАКАМ:")
    print("-"*40)
    for cat in eta_matrix.index:
        mean_eta = eta_matrix.loc[cat].mean()
        print(f"   {cat:25} | средний η² = {mean_eta:.4f}")
    
    # Анализ по числовым признакам
    print("\n" + "-"*40)
    print("СРЕДНИЙ η² ПО ЧИСЛОВЫМ ПРИЗНАКАМ:")
    print("-"*40)
    for num in eta_matrix.columns:
        mean_eta = eta_matrix[num].mean()
        print(f"   {num:25} | средний η² = {mean_eta:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 15 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица:  {matrix_csv_path}")
    print(f"   Пары:     {all_pairs_csv_path}")
    print(f"   Heatmap:  {heatmap_path}")
    print()


def extract_all_pairs(eta_matrix):
    """Извлекает все пары и сортирует по убыванию η²."""
    
    pairs = []
    
    for cat in eta_matrix.index:
        for num in eta_matrix.columns:
            eta_value = eta_matrix.loc[cat, num]
            pairs.append({
                'Категориальный': cat,
                'Числовой': num,
                'η²': eta_value,
                'Процент дисперсии': f"{eta_value*100:.1f}%",
                'Размер эффекта': get_effect_size(eta_value)
            })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('η²', ascending=False).reset_index(drop=True)
    
    return pairs_df


def get_effect_size(eta_sq):
    """Возвращает текстовое описание размера эффекта."""
    if eta_sq > 0.14:
        return "большой"
    elif eta_sq > 0.06:
        return "средний"
    elif eta_sq > 0.01:
        return "малый"
    else:
        return "незначимый"


def create_heatmap(eta_matrix, save_path):
    """Создаёт heatmap матрицы Eta-squared."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Heatmap (диапазон 0 до max значения для лучшей видимости)
    max_val = eta_matrix.values.max()
    im = ax.imshow(eta_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max(max_val, 0.3))
    
    # Настройка осей
    ax.set_xticks(np.arange(len(eta_matrix.columns)))
    ax.set_yticks(np.arange(len(eta_matrix.index)))
    ax.set_xticklabels(eta_matrix.columns, rotation=45, ha='right', fontsize=11, weight='bold')
    ax.set_yticklabels(eta_matrix.index, fontsize=11, weight='bold')
    
    # Подписи осей
    ax.set_xlabel('Числовой признак', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Категориальный признак', fontsize=12, weight='bold', labelpad=10)
    
    # Ручное добавление текста
    for i in range(len(eta_matrix.index)):
        for j in range(len(eta_matrix.columns)):
            value = eta_matrix.iloc[i, j]
            text_color = 'white' if value > 0.15 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=10, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(eta_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(eta_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Eta-squared (η²)', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title('Матрица Eta-squared (η²)\n(доля дисперсии числового признака, объяснённая категориальным)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()
