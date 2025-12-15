"""
Method 11: Spearman Rank Correlation Matrix (Ранговая корреляция Спирмена между числовыми признаками)

Описание:
Вычисляет матрицу ранговых корреляций Спирмена для всех пар числовых признаков.
Показывает МОНОТОННЫЕ связи (не обязательно линейные!).

Применяется: К 7 числовым признакам
Результат: Матрица 7×7, heatmap, топ-10 пар

Отличия от Pearson:
- Pearson: линейные зависимости (Y = aX + b)
- Spearman: монотонные зависимости (Y = X², Y = log(X), любые монотонные)
- Spearman устойчив к выбросам!

Интерпретация:
- |ρ| > 0.7 = сильная монотонная связь
- |ρ| = 0.3-0.7 = средняя связь
- |ρ| < 0.3 = слабая связь
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
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'numeric_correlations' / 'method11_spearman'

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

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 11: SPEARMAN RANK CORRELATION MATRIX (ЧИСЛОВЫЕ ПРИЗНАКИ)")
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
    print(f"Числовых признаков: {len(NUMERIC_FEATURES)}")
    print(f"Количество пар: {len(NUMERIC_FEATURES) * (len(NUMERIC_FEATURES) - 1) // 2}\n")
    
    # Переименовываем колонки
    df_numeric.columns = [FEATURE_NAMES_RU[col] for col in df_numeric.columns]
    
    # Вычисление корреляционной матрицы Спирмена
    print("Вычисление корреляционной матрицы Спирмена...")
    corr_matrix_spearman = df_numeric.corr(method='spearman')
    print("Готово!\n")
    
    # Сохранение полной матрицы
    matrix_csv_path = tables_dir / 'spearman_matrix_7x7.csv'
    corr_matrix_spearman.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица Спирмена сохранена: {matrix_csv_path}\n")
    
    # Извлечение топ-10 пар
    print("Извлечение топ-10 сильнейших корреляций...\n")
    top_pairs = extract_top_pairs(corr_matrix_spearman, n=10)
    
    # Вывод топ-10 в консоль
    print("ТОП-10 СИЛЬНЕЙШИХ КОРРЕЛЯЦИЙ (Spearman):\n")
    for i, row in top_pairs.iterrows():
        print(f"{i+1:2d}. {row['Признак 1']:45} ↔ {row['Признак 2']:45} | p = {row['Корреляция']:+.4f}")
    print()
    
    # Сохранение топ-10
    top_pairs_csv_path = tables_dir / 'spearman_top10_pairs.csv'
    top_pairs.to_csv(top_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Топ-10 пар сохранены: {top_pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'spearman_heatmap.png'
    create_correlation_heatmap(corr_matrix_spearman, heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    abs_corr = corr_matrix_spearman.abs()
    np.fill_diagonal(abs_corr.values, 0)
    
    strong = (abs_corr > 0.7).sum().sum() // 2
    medium = ((abs_corr > 0.3) & (abs_corr <= 0.7)).sum().sum() // 2
    weak = ((abs_corr > 0) & (abs_corr <= 0.3)).sum().sum() // 2
    
    print(f"\nСильные корреляции (|p| > 0.7):   {strong} пар")
    print(f"Средние корреляции (0.3 < |p| <= 0.7): {medium} пар")
    print(f"Слабые корреляции (|p| <= 0.3):   {weak} пар")
    
    # Максимальная корреляция
    max_corr_idx = abs_corr.stack().idxmax()
    max_corr_value = corr_matrix_spearman.loc[max_corr_idx]
    print(f"\nМаксимальная корреляция:")
    print(f"   {max_corr_idx[0]} <-> {max_corr_idx[1]}")
    print(f"   p = {max_corr_value:+.4f}")
    
    print("\n" + "="*80)
    print("METHOD 11 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица:  {matrix_csv_path}")
    print(f"   Топ-10:   {top_pairs_csv_path}")
    print(f"   Heatmap:  {heatmap_path}")
    print()


def extract_top_pairs(corr_matrix, n=10):
    """Извлекает топ-N пар по абсолютному значению."""
    
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
    
    # Heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=11, weight='bold')
    ax.set_yticklabels(corr_matrix.index, fontsize=11, weight='bold')
    
    # Ручное добавление текста
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            text_color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=9, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Корреляция Spearman', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title('Матрица корреляций Spearman (числовые признаки)', 
                fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()