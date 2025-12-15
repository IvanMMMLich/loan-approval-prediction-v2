"""
Method 16: Kruskal-Wallis H-test (Связь между числовыми и категориальными признаками)

Описание:
Непараметрический тест для проверки, различаются ли распределения числового
признака между группами категориального признака.

Применяется: 4 категориальных × 7 числовых = 28 пар
Результат: Матрица 4×7 (H-статистика), heatmap, таблица с p-values

Преимущества перед Eta-squared:
- Не требует нормальности распределения
- Устойчив к выбросам (работает с рангами)
- Даёт p-value для проверки значимости

Интерпретация H-статистики:
- H ≈ 0: распределения одинаковы во всех группах
- H большое: распределения сильно различаются
- Чем больше H → тем сильнее связь

Интерпретация p-value:
- p < 0.001: очень сильное доказательство различий (***)
- p < 0.01: сильное доказательство (**)
- p < 0.05: достаточное доказательство (*)
- p >= 0.05: различия не доказаны (ns)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'mixed_correlations' / 'method16_kruskal_wallis'

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
# ФУНКЦИИ ДЛЯ KRUSKAL-WALLIS
# ============================================================================

def kruskal_wallis_test(df, categorical_col, numeric_col):
    """
    Выполняет Kruskal-Wallis H-test для пары признаков.
    Возвращает H-статистику и p-value.
    """
    # Группируем числовой признак по категориям
    groups = []
    for category in df[categorical_col].unique():
        group_data = df[df[categorical_col] == category][numeric_col].dropna()
        if len(group_data) > 0:
            groups.append(group_data.values)
    
    # Нужно минимум 2 группы
    if len(groups) < 2:
        return 0.0, 1.0
    
    # Kruskal-Wallis тест
    h_stat, p_value = kruskal(*groups)
    
    return h_stat, p_value


def compute_kruskal_wallis_matrix(df, categorical_features, numeric_features):
    """
    Вычисляет матрицы H-статистики и p-values для всех пар.
    """
    n_cat = len(categorical_features)
    n_num = len(numeric_features)
    
    # Инициализация матриц
    h_matrix = np.zeros((n_cat, n_num))
    p_matrix = np.zeros((n_cat, n_num))
    
    total_pairs = n_cat * n_num
    current_pair = 0
    
    for i, cat_feat in enumerate(categorical_features):
        for j, num_feat in enumerate(numeric_features):
            current_pair += 1
            print(f"   Пара {current_pair}/{total_pairs}: {cat_feat} -> {num_feat}")
            
            h_stat, p_value = kruskal_wallis_test(df, cat_feat, num_feat)
            h_matrix[i, j] = h_stat
            p_matrix[i, j] = p_value
    
    h_df = pd.DataFrame(h_matrix, index=categorical_features, columns=numeric_features)
    p_df = pd.DataFrame(p_matrix, index=categorical_features, columns=numeric_features)
    
    return h_df, p_df


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 16: KRUSKAL-WALLIS H-TEST (КАТЕГОРИАЛЬНЫЕ -> ЧИСЛОВЫЕ)")
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
    
    # Вычисление Kruskal-Wallis
    print("Вычисление Kruskal-Wallis H-test...\n")
    
    h_matrix, p_matrix = compute_kruskal_wallis_matrix(df, CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    
    # Переименовываем индексы и колонки на русский
    h_matrix.index = [FEATURE_NAMES_RU[col] for col in h_matrix.index]
    h_matrix.columns = [FEATURE_NAMES_RU[col] for col in h_matrix.columns]
    p_matrix.index = [FEATURE_NAMES_RU[col] for col in p_matrix.index]
    p_matrix.columns = [FEATURE_NAMES_RU[col] for col in p_matrix.columns]
    
    print("\nГотово!\n")
    
    # Сохранение матрицы H-статистики
    h_csv_path = tables_dir / 'kruskal_wallis_h_matrix_4x7.csv'
    h_matrix.to_csv(h_csv_path, encoding='utf-8-sig')
    print(f"Матрица H-статистики сохранена: {h_csv_path}")
    
    # Сохранение матрицы p-values
    p_csv_path = tables_dir / 'kruskal_wallis_pvalue_matrix_4x7.csv'
    p_matrix.to_csv(p_csv_path, encoding='utf-8-sig')
    print(f"Матрица p-values сохранена: {p_csv_path}\n")
    
    # Извлечение всех пар
    print("Извлечение всех пар...\n")
    all_pairs = extract_all_pairs(h_matrix, p_matrix)
    
    # Вывод топ-10 в консоль
    print("ТОП-10 СИЛЬНЕЙШИХ СВЯЗЕЙ (Kruskal-Wallis H):\n")
    for i, row in all_pairs.head(10).iterrows():
        stars = get_significance_stars(row['p-value'])
        print(f"{i+1:2d}. {row['Категориальный']:20} -> {row['Числовой']:20} | H = {row['H-статистика']:10.2f} | p = {row['p-value']:.2e} {stars}")
    print()
    
    # Сохранение всех пар
    all_pairs_csv_path = tables_dir / 'kruskal_wallis_all_pairs.csv'
    all_pairs.to_csv(all_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Все пары сохранены: {all_pairs_csv_path}\n")
    
    # Создание heatmap для H-статистики
    print("Создание heatmap H-статистики...")
    h_heatmap_path = figures_dir / 'kruskal_wallis_h_heatmap.png'
    create_h_heatmap(h_matrix, p_matrix, h_heatmap_path)
    
    # Создание heatmap для p-values
    print("Создание heatmap p-values...")
    p_heatmap_path = figures_dir / 'kruskal_wallis_pvalue_heatmap.png'
    create_pvalue_heatmap(p_matrix, p_heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА ЗНАЧИМОСТИ:")
    print("="*80)
    
    p_values = p_matrix.values.flatten()
    
    very_sig = (p_values < 0.001).sum()
    sig = ((p_values >= 0.001) & (p_values < 0.01)).sum()
    moderate = ((p_values >= 0.01) & (p_values < 0.05)).sum()
    not_sig = (p_values >= 0.05).sum()
    
    print(f"\nОчень значимые (p < 0.001) ***: {very_sig} пар")
    print(f"Значимые (p < 0.01) **:      {sig} пар")
    print(f"Умеренно значимые (p < 0.05) *: {moderate} пар")
    print(f"Не значимые (p >= 0.05) ns:   {not_sig} пар")
    
    # Максимальная H-статистика
    max_idx = h_matrix.stack().idxmax()
    max_h = h_matrix.loc[max_idx]
    max_p = p_matrix.loc[max_idx]
    print(f"\nСамая сильная связь:")
    print(f"   {max_idx[0]} -> {max_idx[1]}")
    print(f"   H = {max_h:.2f}, p = {max_p:.2e}")
    
    # Анализ по категориальным признакам
    print("\n" + "-"*40)
    print("СРЕДНЯЯ H-СТАТИСТИКА ПО КАТЕГОРИАЛЬНЫМ:")
    print("-"*40)
    for cat in h_matrix.index:
        mean_h = h_matrix.loc[cat].mean()
        print(f"   {cat:25} | средний H = {mean_h:.2f}")
    
    print("\n" + "="*80)
    print("METHOD 16 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   H-матрица:  {h_csv_path}")
    print(f"   P-матрица:  {p_csv_path}")
    print(f"   Пары:       {all_pairs_csv_path}")
    print(f"   H-heatmap:  {h_heatmap_path}")
    print(f"   P-heatmap:  {p_heatmap_path}")
    print()


def extract_all_pairs(h_matrix, p_matrix):
    """Извлекает все пары и сортирует по убыванию H-статистики."""
    
    pairs = []
    
    for cat in h_matrix.index:
        for num in h_matrix.columns:
            h_value = h_matrix.loc[cat, num]
            p_value = p_matrix.loc[cat, num]
            pairs.append({
                'Категориальный': cat,
                'Числовой': num,
                'H-статистика': h_value,
                'p-value': p_value,
                'Значимость': get_significance_stars(p_value)
            })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('H-статистика', ascending=False).reset_index(drop=True)
    
    return pairs_df


def get_significance_stars(p_value):
    """Возвращает звёздочки значимости."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def create_h_heatmap(h_matrix, p_matrix, save_path):
    """Создаёт heatmap H-статистики со звёздочками значимости."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Логарифмическая шкала для лучшей визуализации
    h_log = np.log10(h_matrix + 1)
    
    im = ax.imshow(h_log, cmap='YlOrRd', aspect='auto')
    
    # Настройка осей
    ax.set_xticks(np.arange(len(h_matrix.columns)))
    ax.set_yticks(np.arange(len(h_matrix.index)))
    ax.set_xticklabels(h_matrix.columns, rotation=45, ha='right', fontsize=11, weight='bold')
    ax.set_yticklabels(h_matrix.index, fontsize=11, weight='bold')
    
    # Подписи осей
    ax.set_xlabel('Числовой признак', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Категориальный признак', fontsize=12, weight='bold', labelpad=10)
    
    # Ручное добавление текста с H-значением и звёздочками
    for i in range(len(h_matrix.index)):
        for j in range(len(h_matrix.columns)):
            h_value = h_matrix.iloc[i, j]
            p_value = p_matrix.iloc[i, j]
            stars = get_significance_stars(p_value)
            
            text_color = 'white' if h_log.iloc[i, j] > h_log.values.max() * 0.6 else 'black'
            
            # Форматируем число
            if h_value >= 1000:
                text = f'{h_value:.0f}\n{stars}'
            else:
                text = f'{h_value:.1f}\n{stars}'
            
            ax.text(j, i, text, ha='center', va='center',
                   color=text_color, fontsize=9, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(h_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(h_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('log10(H + 1)', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title('Kruskal-Wallis H-статистика\n(*** p<0.001, ** p<0.01, * p<0.05, ns - не значимо)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"H-heatmap сохранен: {save_path}")


def create_pvalue_heatmap(p_matrix, save_path):
    """Создаёт heatmap p-values."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Логарифмическая шкала для p-values
    p_log = -np.log10(p_matrix + 1e-300)  # Добавляем маленькое число чтобы избежать log(0)
    
    im = ax.imshow(p_log, cmap='RdYlGn_r', aspect='auto')
    
    # Настройка осей
    ax.set_xticks(np.arange(len(p_matrix.columns)))
    ax.set_yticks(np.arange(len(p_matrix.index)))
    ax.set_xticklabels(p_matrix.columns, rotation=45, ha='right', fontsize=11, weight='bold')
    ax.set_yticklabels(p_matrix.index, fontsize=11, weight='bold')
    
    # Подписи осей
    ax.set_xlabel('Числовой признак', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Категориальный признак', fontsize=12, weight='bold', labelpad=10)
    
    # Ручное добавление текста
    for i in range(len(p_matrix.index)):
        for j in range(len(p_matrix.columns)):
            p_value = p_matrix.iloc[i, j]
            stars = get_significance_stars(p_value)
            
            text_color = 'white' if p_log.iloc[i, j] > 10 else 'black'
            
            if p_value < 0.001:
                text = f'<0.001\n{stars}'
            else:
                text = f'{p_value:.3f}\n{stars}'
            
            ax.text(j, i, text, ha='center', va='center',
                   color=text_color, fontsize=9, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(p_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(p_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('-log10(p-value)', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title('Kruskal-Wallis p-values\n(чем краснее - тем значимее различия)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"P-value heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()
