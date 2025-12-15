"""
Method 12: Distance Correlation Matrix (Дистанционная корреляция между числовыми признаками)

Описание:
Вычисляет матрицу дистанционных корреляций для всех пар числовых признаков.
Показывает ЛЮБЫЕ зависимости (линейные, нелинейные, U-образные, синусоиды и т.д.)

КРИТИЧНО: Требуется стандартизация данных!
Без стандартизации признаки с большими значениями (Доход: 20,000) будут
доминировать над признаками с малыми значениями (% дохода: 25%).

Применяется: К 7 числовым признакам (после StandardScaler)
Результат: Матрица 7×7, heatmap, топ-10 пар

Отличия от Pearson/Spearman:
- Pearson: только линейные зависимости
- Spearman: только монотонные зависимости
- Distance Correlation: ЛЮБЫЕ зависимости!
- dcor = 0 означает ТОЧНУЮ независимость

Интерпретация:
- dcor = 0: признаки ТОЧНО независимы
- dcor = 0.1-0.3: слабая зависимость
- dcor = 0.3-0.5: средняя зависимость
- dcor > 0.5: сильная зависимость

Примечание: Диапазон от 0 до 1 (не от -1 до +1!)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler  # ← ДОБАВИЛИ

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'numeric_correlations' / 'method12_distance_corr'

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
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы (лет)',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка (%)',
    'loan_percent_income': 'Процент дохода на кредит (%)',
    'cb_person_cred_hist_length': 'Длина кредитной истории (лет)'
}

# Размер выборки для ускорения (Distance Correlation - O(n^2))
SAMPLE_SIZE = 5000

# ============================================================================
# ФУНКЦИИ ДЛЯ DISTANCE CORRELATION
# ============================================================================

def distance_matrix_centered(x):
    """
    Вычисляет центрированную матрицу расстояний для вектора x.
    Double centering: A_ij = a_ij - mean(row_i) - mean(col_j) + mean(all)
    """
    x = np.asarray(x).reshape(-1, 1)
    
    # Матрица расстояний
    dist = squareform(pdist(x, metric='euclidean'))
    
    # Double centering
    row_mean = dist.mean(axis=1, keepdims=True)
    col_mean = dist.mean(axis=0, keepdims=True)
    total_mean = dist.mean()
    
    centered = dist - row_mean - col_mean + total_mean
    
    return centered


def distance_covariance(x, y):
    """
    Вычисляет дистанционную ковариацию между x и y.
    dCov^2(X,Y) = (1/n^2) * sum(A_ij * B_ij)
    """
    n = len(x)
    
    A = distance_matrix_centered(x)
    B = distance_matrix_centered(y)
    
    dcov_squared = (A * B).sum() / (n * n)
    
    return np.sqrt(max(dcov_squared, 0))


def distance_correlation(x, y):
    """
    Вычисляет дистанционную корреляцию между x и y.
    dCor(X,Y) = dCov(X,Y) / sqrt(dVar(X) * dVar(Y))
    """
    dcov_xy = distance_covariance(x, y)
    dcov_xx = distance_covariance(x, x)
    dcov_yy = distance_covariance(y, y)
    
    if dcov_xx * dcov_yy == 0:
        return 0.0
    
    dcor = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
    
    return dcor


def compute_distance_correlation_matrix(df):
    """
    Вычисляет матрицу дистанционных корреляций для всех пар признаков.
    
    ВАЖНО: df должен быть уже стандартизован!
    """
    features = df.columns.tolist()
    n_features = len(features)
    
    # Инициализация матрицы
    dcor_matrix = np.zeros((n_features, n_features))
    
    total_pairs = n_features * (n_features - 1) // 2
    current_pair = 0
    
    for i in range(n_features):
        dcor_matrix[i, i] = 1.0  # Диагональ = 1
        
        for j in range(i + 1, n_features):
            current_pair += 1
            print(f"   Пара {current_pair}/{total_pairs}: {features[i]} <-> {features[j]}", end='\r')
            
            x = df.iloc[:, i].values
            y = df.iloc[:, j].values
            
            dcor = distance_correlation(x, y)
            
            dcor_matrix[i, j] = dcor
            dcor_matrix[j, i] = dcor  # Симметричная матрица
    
    print()  # Новая строка после прогресса
    
    return pd.DataFrame(dcor_matrix, index=features, columns=features)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 12: DISTANCE CORRELATION MATRIX (ЧИСЛОВЫЕ ПРИЗНАКИ)")
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
    
    # Выбор только числовых признаков
    df_numeric = df[NUMERIC_FEATURES].copy()
    
    # Сэмплирование для ускорения
    if len(df_numeric) > SAMPLE_SIZE:
        print(f"Сэмплирование до {SAMPLE_SIZE:,} строк для ускорения вычислений...")
        df_numeric = df_numeric.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"После сэмплирования: {len(df_numeric):,} строк\n")
    
    # ========================================================================
    # КРИТИЧНО: СТАНДАРТИЗАЦИЯ ДАННЫХ
    # ========================================================================
    print("СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ (StandardScaler)...")
    print("Причина: Разные масштабы (Доход: 20,000 vs % дохода: 25%)")
    print("Решение: Приводим все к mean=0, std=1\n")
    
    scaler = StandardScaler()
    df_numeric_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns,
        index=df_numeric.index
    )
    
    print("Стандартизация завершена!")
    print(f"   Теперь все признаки: mean ≈ 0, std ≈ 1\n")
    
    # Переименовываем колонки
    df_numeric_scaled.columns = [FEATURE_NAMES_RU[col] for col in df_numeric_scaled.columns]
    
    print(f"Числовых признаков: {len(NUMERIC_FEATURES)}")
    print(f"Количество пар: {len(NUMERIC_FEATURES) * (len(NUMERIC_FEATURES) - 1) // 2}\n")
    
    # Вычисление матрицы Distance Correlation (на СТАНДАРТИЗОВАННЫХ данных)
    print("Вычисление матрицы Distance Correlation...")
    print("(Это может занять 1-2 минуты)\n")
    
    dcor_matrix = compute_distance_correlation_matrix(df_numeric_scaled)  # ← ИЗМЕНИЛИ
    print("\nГотово!\n")
    
    # Сохранение полной матрицы
    matrix_csv_path = tables_dir / 'distance_corr_matrix_7x7.csv'
    dcor_matrix.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица Distance Correlation сохранена: {matrix_csv_path}\n")
    
    # Извлечение топ-10 пар
    print("Извлечение топ-10 сильнейших корреляций...\n")
    top_pairs = extract_top_pairs(dcor_matrix, n=10)
    
    # Вывод топ-10 в консоль
    print("ТОП-10 СИЛЬНЕЙШИХ КОРРЕЛЯЦИЙ (Distance Correlation):\n")
    for i, row in top_pairs.iterrows():
        print(f"{i+1:2d}. {row['Признак 1']:45} <-> {row['Признак 2']:45} | dcor = {row['Корреляция']:.4f}")
    print()
    
    # Сохранение топ-10
    top_pairs_csv_path = tables_dir / 'distance_corr_top10_pairs.csv'
    top_pairs.to_csv(top_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Топ-10 пар сохранены: {top_pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'distance_corr_heatmap.png'
    create_correlation_heatmap(dcor_matrix, heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    # Копия матрицы без диагонали
    dcor_values = dcor_matrix.values.copy()
    np.fill_diagonal(dcor_values, 0)
    
    strong = (dcor_values > 0.5).sum() // 2
    medium = ((dcor_values > 0.3) & (dcor_values <= 0.5)).sum() // 2
    weak = ((dcor_values > 0) & (dcor_values <= 0.3)).sum() // 2
    
    print(f"\nСильные зависимости (dcor > 0.5):   {strong} пар")
    print(f"Средние зависимости (0.3 < dcor <= 0.5): {medium} пар")
    print(f"Слабые зависимости (dcor <= 0.3):   {weak} пар")
    
    # Максимальная корреляция
    dcor_no_diag = dcor_matrix.copy()
    np.fill_diagonal(dcor_no_diag.values, 0)
    max_corr_idx = dcor_no_diag.stack().idxmax()
    max_corr_value = dcor_matrix.loc[max_corr_idx]
    print(f"\nМаксимальная зависимость:")
    print(f"   {max_corr_idx[0]} <-> {max_corr_idx[1]}")
    print(f"   dcor = {max_corr_value:.4f}")
    
    print("\n" + "="*80)
    print("METHOD 12 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица:  {matrix_csv_path}")
    print(f"   Топ-10:   {top_pairs_csv_path}")
    print(f"   Heatmap:  {heatmap_path}")
    print()


def extract_top_pairs(corr_matrix, n=10):
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
                    'Абсолютное значение': abs(corr_value)
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('Абсолютное значение', ascending=False).head(n).reset_index(drop=True)
    
    return pairs_df


def create_correlation_heatmap(corr_matrix, save_path):
    """Создаёт heatmap корреляционной матрицы."""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Heatmap (диапазон 0-1, не -1 до +1!)
    im = ax.imshow(corr_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=11, weight='bold')
    ax.set_yticklabels(corr_matrix.index, fontsize=11, weight='bold')
    
    # Ручное добавление текста
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=9, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Distance Correlation', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title('Матрица Distance Correlation (числовые признаки)\n'
                 '(ловит ЛЮБЫЕ зависимости, диапазон 0-1, данные стандартизованы)', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=350, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()