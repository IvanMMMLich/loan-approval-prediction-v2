"""
Method 14: Uncertainty Coefficient (Theil's U) - Категориальные признаки

Описание:
Вычисляет коэффициент неопределённости (Theil's U) для всех пар категориальных признаков.
Это АСИММЕТРИЧНАЯ мера - U(A→B) показывает, насколько знание A уменьшает неопределённость B.

Применяется: К 4 категориальным признакам
Результат: Матрица 4×4 (НЕ симметричная!), heatmap, таблица всех пар

Формула:
U(X→Y) = (H(Y) - H(Y|X)) / H(Y)
где:
- H(Y) - энтропия Y
- H(Y|X) - условная энтропия Y при известном X

Интерпретация:
- U = 0: X не даёт информации о Y
- U = 0.1-0.3: слабая предсказательная сила
- U = 0.3-0.5: средняя предсказательная сила
- U > 0.5: сильная предсказательная сила
- U = 1: X полностью определяет Y

Примечание:
- Диапазон от 0 до 1
- АСИММЕТРИЧНЫЙ: U(A→B) ≠ U(B→A)
- Читается как "насколько A предсказывает B"
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
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'categorical_correlations' / 'method14_uncertainty_coef'

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
    'cb_person_default_on_file': 'Наличие дефолта'
}

# ============================================================================
# ФУНКЦИИ ДЛЯ UNCERTAINTY COEFFICIENT
# ============================================================================

def entropy(x):
    """
    Вычисляет энтропию категориальной переменной.
    H(X) = -sum(p_i * log2(p_i))
    """
    # Частоты
    value_counts = x.value_counts(normalize=True)
    
    # Убираем нули (log(0) = undefined)
    value_counts = value_counts[value_counts > 0]
    
    # Энтропия
    h = -np.sum(value_counts * np.log2(value_counts))
    
    return h


def conditional_entropy(x, y):
    """
    Вычисляет условную энтропию H(Y|X).
    H(Y|X) = sum_x P(X=x) * H(Y|X=x)
    """
    # Уникальные значения X
    x_values = x.unique()
    
    h_y_given_x = 0
    n = len(x)
    
    for x_val in x_values:
        # Маска для текущего значения X
        mask = (x == x_val)
        
        # P(X = x_val)
        p_x = mask.sum() / n
        
        # Y при X = x_val
        y_given_x = y[mask]
        
        # H(Y | X = x_val)
        h_y_x = entropy(y_given_x)
        
        h_y_given_x += p_x * h_y_x
    
    return h_y_given_x


def uncertainty_coefficient(x, y):
    """
    Вычисляет Uncertainty Coefficient (Theil's U).
    U(X→Y) = (H(Y) - H(Y|X)) / H(Y)
    
    Показывает: насколько знание X уменьшает неопределённость Y.
    """
    h_y = entropy(y)
    
    # Если H(Y) = 0, Y константа, возвращаем 0
    if h_y == 0:
        return 0.0
    
    h_y_given_x = conditional_entropy(x, y)
    
    u = (h_y - h_y_given_x) / h_y
    
    return u


def compute_uncertainty_matrix(df, features):
    """
    Вычисляет матрицу Uncertainty Coefficient для всех пар признаков.
    Матрица АСИММЕТРИЧНАЯ: элемент [i,j] = U(feature_i → feature_j)
    """
    n_features = len(features)
    
    # Инициализация матрицы
    u_matrix = np.zeros((n_features, n_features))
    
    total_pairs = n_features * n_features
    current_pair = 0
    
    for i in range(n_features):
        for j in range(n_features):
            current_pair += 1
            
            if i == j:
                u_matrix[i, j] = 1.0  # Диагональ = 1 (признак полностью предсказывает сам себя)
            else:
                print(f"   Пара {current_pair}/{total_pairs}: {features[i]} -> {features[j]}")
                u = uncertainty_coefficient(df[features[i]], df[features[j]])
                u_matrix[i, j] = u
    
    return pd.DataFrame(u_matrix, index=features, columns=features)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 14: UNCERTAINTY COEFFICIENT (THEIL'S U)")
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
    print(f"Количество направленных пар: {len(CATEGORICAL_FEATURES) * (len(CATEGORICAL_FEATURES) - 1)}\n")
    
    # Вычисление матрицы Uncertainty Coefficient
    print("Вычисление матрицы Uncertainty Coefficient...\n")
    
    u_matrix = compute_uncertainty_matrix(df_categorical, CATEGORICAL_FEATURES)
    
    # Переименовываем индексы и колонки на русский
    u_matrix.index = [FEATURE_NAMES_RU[col] for col in u_matrix.index]
    u_matrix.columns = [FEATURE_NAMES_RU[col] for col in u_matrix.columns]
    
    print("\nГотово!\n")
    
    # Сохранение полной матрицы
    matrix_csv_path = tables_dir / 'uncertainty_coef_matrix_4x4.csv'
    u_matrix.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица Uncertainty Coefficient сохранена: {matrix_csv_path}\n")
    
    # Извлечение всех направленных пар
    print("Извлечение всех направленных пар...\n")
    all_pairs = extract_all_pairs(u_matrix)
    
    # Вывод в консоль
    print("ВСЕ НАПРАВЛЕННЫЕ ПАРЫ (Uncertainty Coefficient):\n")
    print("Читается как: 'Признак 1 предсказывает Признак 2 на U%'\n")
    for i, row in all_pairs.iterrows():
        strength = get_strength_label(row['U'])
        print(f"{i+1:2d}. {row['Признак 1']:25} -> {row['Признак 2']:25} | U = {row['U']:.4f} ({strength})")
    print()
    
    # Сохранение всех пар
    all_pairs_csv_path = tables_dir / 'uncertainty_coef_all_pairs.csv'
    all_pairs.to_csv(all_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Все пары сохранены: {all_pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'uncertainty_coef_heatmap.png'
    create_heatmap(u_matrix, heatmap_path)
    
    # Анализ асимметрии
    print("\n" + "="*80)
    print("АНАЛИЗ АСИММЕТРИИ:")
    print("="*80)
    print("\nСравнение U(A→B) vs U(B→A):\n")
    
    features_ru = list(u_matrix.index)
    for i in range(len(features_ru)):
        for j in range(i + 1, len(features_ru)):
            u_ab = u_matrix.iloc[i, j]
            u_ba = u_matrix.iloc[j, i]
            diff = abs(u_ab - u_ba)
            
            print(f"   {features_ru[i]:25} <-> {features_ru[j]:25}")
            print(f"      U({features_ru[i][:10]}... -> {features_ru[j][:10]}...) = {u_ab:.4f}")
            print(f"      U({features_ru[j][:10]}... -> {features_ru[i][:10]}...) = {u_ba:.4f}")
            print(f"      Разница: {diff:.4f}")
            print()
    
    # Статистика
    print("="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    # Копия матрицы без диагонали
    u_values = u_matrix.values.copy()
    np.fill_diagonal(u_values, 0)
    
    # Только уникальные пары для статистики (верхний треугольник)
    strong = (u_values > 0.3).sum()
    medium = ((u_values > 0.1) & (u_values <= 0.3)).sum()
    weak = ((u_values > 0) & (u_values <= 0.1)).sum()
    
    print(f"\nСильная предсказательная сила (U > 0.3): {strong} направленных пар")
    print(f"Средняя предсказательная сила (0.1 < U <= 0.3): {medium} направленных пар")
    print(f"Слабая предсказательная сила (U <= 0.1): {weak} направленных пар")
    
    # Максимальная асимметричная пара
    u_no_diag = u_matrix.copy()
    np.fill_diagonal(u_no_diag.values, 0)
    max_idx = u_no_diag.stack().idxmax()
    max_value = u_matrix.loc[max_idx]
    print(f"\nСамая сильная предсказательная связь:")
    print(f"   {max_idx[0]} -> {max_idx[1]}")
    print(f"   U = {max_value:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 14 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица:  {matrix_csv_path}")
    print(f"   Пары:     {all_pairs_csv_path}")
    print(f"   Heatmap:  {heatmap_path}")
    print()


def extract_all_pairs(u_matrix):
    """Извлекает все направленные пары."""
    
    pairs = []
    features = u_matrix.index.tolist()
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i != j:
                u_value = u_matrix.loc[feat1, feat2]
                pairs.append({
                    'Признак 1': feat1,
                    'Признак 2': feat2,
                    'U': u_value,
                    'Сила связи': get_strength_label(u_value),
                    'Интерпретация': f"'{feat1}' предсказывает '{feat2}' на {u_value*100:.1f}%"
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('U', ascending=False).reset_index(drop=True)
    
    return pairs_df


def get_strength_label(u):
    """Возвращает текстовое описание силы связи."""
    if u > 0.3:
        return "сильная"
    elif u > 0.1:
        return "средняя"
    elif u > 0.05:
        return "слабая"
    else:
        return "очень слабая"


def create_heatmap(u_matrix, save_path):
    """Создаёт heatmap матрицы Uncertainty Coefficient."""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Heatmap (диапазон 0-1)
    im = ax.imshow(u_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(u_matrix.columns)))
    ax.set_yticks(np.arange(len(u_matrix.index)))
    ax.set_xticklabels(u_matrix.columns, rotation=45, ha='right', fontsize=12, weight='bold')
    ax.set_yticklabels(u_matrix.index, fontsize=12, weight='bold')
    
    # Подписи осей
    ax.set_xlabel('Предсказываемый признак (Y)', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Предсказывающий признак (X)', fontsize=12, weight='bold', labelpad=10)
    
    # Ручное добавление текста
    for i in range(len(u_matrix.index)):
        for j in range(len(u_matrix.columns)):
            value = u_matrix.iloc[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=14, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(u_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(u_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("Uncertainty Coefficient (Theil's U)", rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title("Матрица Uncertainty Coefficient (Theil's U)\n(АСИММЕТРИЧНАЯ: строка предсказывает столбец)", 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()
