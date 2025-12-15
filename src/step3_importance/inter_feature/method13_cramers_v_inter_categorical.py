"""
Method 13: Cramér's V Matrix (Корреляция Крамера между категориальными признаками)

Описание:
Вычисляет матрицу корреляций Крамера для всех пар категориальных признаков.
Cramér's V - это мера связи между двумя категориальными переменными,
основанная на Chi-Square статистике.

Применяется: К 4 категориальным признакам
Результат: Матрица 4×4, heatmap, топ пар

Формула:
V = sqrt(chi2 / (n * min(r-1, c-1)))
где:
- chi2 - статистика хи-квадрат
- n - количество наблюдений
- r, c - количество строк и столбцов в таблице сопряженности

Интерпретация:
- V = 0: нет связи
- V = 0.1-0.3: слабая связь
- V = 0.3-0.5: средняя связь
- V > 0.5: сильная связь
- V = 1: полная связь

Примечание: 
- Диапазон от 0 до 1 (симметричная мера)
- V(A,B) = V(B,A)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2_contingency

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'categorical_correlations' / 'method13_cramers_v'

# Категориальные признаки
CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_home_ownership': 'Владение жильём',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта в истории'
}

# ============================================================================
# ФУНКЦИИ ДЛЯ CRAMÉR'S V
# ============================================================================

def cramers_v(x, y):
    """
    Вычисляет Cramér's V для двух категориальных переменных.
    """
    # Таблица сопряженности
    contingency_table = pd.crosstab(x, y)
    
    # Chi-Square тест
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Количество наблюдений
    n = len(x)
    
    # Количество строк и столбцов
    r, c = contingency_table.shape
    
    # Cramér's V
    v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))
    
    return v


def compute_cramers_v_matrix(df, features):
    """
    Вычисляет матрицу Cramér's V для всех пар признаков.
    """
    n_features = len(features)
    
    # Инициализация матрицы
    v_matrix = np.zeros((n_features, n_features))
    
    total_pairs = n_features * (n_features - 1) // 2
    current_pair = 0
    
    for i in range(n_features):
        v_matrix[i, i] = 1.0  # Диагональ = 1
        
        for j in range(i + 1, n_features):
            current_pair += 1
            print(f"   Пара {current_pair}/{total_pairs}: {features[i]} <-> {features[j]}")
            
            v = cramers_v(df[features[i]], df[features[j]])
            
            v_matrix[i, j] = v
            v_matrix[j, i] = v  # Симметричная матрица
    
    return pd.DataFrame(v_matrix, index=features, columns=features)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 13: CRAMER'S V MATRIX (КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ)")
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
    
    # Выбор только категориальных признаков
    df_categorical = df[CATEGORICAL_FEATURES].copy()
    
    print(f"Категориальных признаков: {len(CATEGORICAL_FEATURES)}")
    print(f"Количество пар: {len(CATEGORICAL_FEATURES) * (len(CATEGORICAL_FEATURES) - 1) // 2}\n")
    
    # Вычисление матрицы Cramér's V
    print("Вычисление матрицы Cramer's V...\n")
    
    v_matrix = compute_cramers_v_matrix(df_categorical, CATEGORICAL_FEATURES)
    
    # Переименовываем индексы и колонки на русский
    v_matrix.index = [FEATURE_NAMES_RU[col] for col in v_matrix.index]
    v_matrix.columns = [FEATURE_NAMES_RU[col] for col in v_matrix.columns]
    
    print("\nГотово!\n")
    
    # Сохранение полной матрицы
    matrix_csv_path = tables_dir / 'cramers_v_matrix_4x4.csv'
    v_matrix.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица Cramer's V сохранена: {matrix_csv_path}\n")
    
    # Извлечение топ пар
    print("Извлечение сильнейших корреляций...\n")
    top_pairs = extract_top_pairs(v_matrix, n=6)  # 6 пар всего для 4 признаков
    
    # Вывод в консоль
    print("ВСЕ ПАРЫ КОРРЕЛЯЦИЙ (Cramer's V):\n")
    for i, row in top_pairs.iterrows():
        strength = get_strength_label(row['Корреляция'])
        print(f"{i+1}. {row['Признак 1']:30} <-> {row['Признак 2']:30} | V = {row['Корреляция']:.4f} ({strength})")
    print()
    
    # Сохранение топ пар
    top_pairs_csv_path = tables_dir / 'cramers_v_all_pairs.csv'
    top_pairs.to_csv(top_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Все пары сохранены: {top_pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'cramers_v_heatmap.png'
    create_correlation_heatmap(v_matrix, heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    # Копия матрицы без диагонали
    v_values = v_matrix.values.copy()
    np.fill_diagonal(v_values, 0)
    
    strong = (v_values > 0.5).sum() // 2
    medium = ((v_values > 0.3) & (v_values <= 0.5)).sum() // 2
    weak = ((v_values > 0.1) & (v_values <= 0.3)).sum() // 2
    very_weak = ((v_values > 0) & (v_values <= 0.1)).sum() // 2
    
    print(f"\nСильные связи (V > 0.5):     {strong} пар")
    print(f"Средние связи (0.3 < V <= 0.5):  {medium} пар")
    print(f"Слабые связи (0.1 < V <= 0.3):   {weak} пар")
    print(f"Очень слабые (V <= 0.1):     {very_weak} пар")
    
    # Максимальная корреляция
    v_no_diag = v_matrix.copy()
    np.fill_diagonal(v_no_diag.values, 0)
    max_corr_idx = v_no_diag.stack().idxmax()
    max_corr_value = v_matrix.loc[max_corr_idx]
    print(f"\nМаксимальная связь:")
    print(f"   {max_corr_idx[0]} <-> {max_corr_idx[1]}")
    print(f"   V = {max_corr_value:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 13 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица:  {matrix_csv_path}")
    print(f"   Пары:     {top_pairs_csv_path}")
    print(f"   Heatmap:  {heatmap_path}")
    print()


def extract_top_pairs(corr_matrix, n=6):
    """Извлекает топ-N пар по значению корреляции."""
    
    pairs = []
    features = corr_matrix.index.tolist()
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i < j:
                corr_value = corr_matrix.loc[feat1, feat2]
                pairs.append({
                    'Признак 1': feat1,
                    'Признак 2': feat2,
                    'Корреляция': corr_value,
                    'Сила связи': get_strength_label(corr_value)
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('Корреляция', ascending=False).reset_index(drop=True)
    
    return pairs_df.head(n)


def get_strength_label(v):
    """Возвращает текстовое описание силы связи."""
    if v > 0.5:
        return "сильная"
    elif v > 0.3:
        return "средняя"
    elif v > 0.1:
        return "слабая"
    else:
        return "очень слабая"


def create_correlation_heatmap(corr_matrix, save_path):
    """Создаёт heatmap корреляционной матрицы."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap (диапазон 0-1)
    im = ax.imshow(corr_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=12, weight='bold')
    ax.set_yticklabels(corr_matrix.index, fontsize=12, weight='bold')
    
    # Ручное добавление текста
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=14, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("Cramer's V", rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title("Матрица Cramer's V (категориальные признаки)\n(мера связи между категориями, диапазон 0-1)", 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()