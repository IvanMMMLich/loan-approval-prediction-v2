"""
Method 10: Pearson Correlation Matrix (Корреляция Пирсона между числовыми признаками)

Описание:
Вычисляет матрицу корреляций Пирсона для всех пар числовых признаков.
Показывает ЛИНЕЙНЫЕ связи между признаками (не с target!).

Применяется: К 7 числовым признакам
Результат: Матрица 7×7, heatmap, топ-10 пар по абсолютному значению

Интерпретация:
- |r| > 0.7 = СИЛЬНАЯ связь (возможна мультиколлинеарность!)
- |r| = 0.3-0.7 = средняя связь
- |r| < 0.3 = слабая связь

Зачем нужно:
- Найти дублирующие признаки (для удаления)
- Понять структуру данных
- Подготовить к Feature Engineering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'numeric_correlations' / 'method10_pearson'

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
    print("METHOD 10: PEARSON CORRELATION MATRIX (ЧИСЛОВЫЕ ПРИЗНАКИ)")
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
    
    # ИСПРАВЛЕНО: Переименовываем колонки ПЕРЕД вычислением корреляции
    df_numeric.columns = [FEATURE_NAMES_RU[col] for col in df_numeric.columns]
    
    # Вычисление корреляционной матрицы (теперь уже с русскими названиями)
    print("Вычисление корреляционной матрицы Пирсона...")
    corr_matrix_ru = df_numeric.corr(method='pearson')
    print("Готово!\n")
    
    # Сохранение полной матрицы
    matrix_csv_path = tables_dir / 'pearson_matrix_7x7.csv'
    corr_matrix_ru.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица корреляций сохранена: {matrix_csv_path}\n")
    
    # Извлечение топ-10 пар
    print("Извлечение топ-10 сильнейших корреляций...\n")
    top_pairs = extract_top_pairs_fixed(corr_matrix_ru, n=10)
    
    # Вывод топ-10 в консоль
    print("ТОП-10 СИЛЬНЕЙШИХ КОРРЕЛЯЦИЙ:\n")
    for i, row in top_pairs.iterrows():
        print(f"{i+1:2d}. {row['Признак 1']:45} ↔ {row['Признак 2']:45} | r = {row['Корреляция']:+.4f}")
    print()
    
    # Сохранение топ-10
    top_pairs_csv_path = tables_dir / 'pearson_top10_pairs.csv'
    top_pairs.to_csv(top_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Топ-10 пар сохранены: {top_pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'pearson_heatmap.png'
    create_correlation_heatmap(corr_matrix_ru, heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    # Количество сильных корреляций
    abs_corr = corr_matrix_ru.abs()
    np.fill_diagonal(abs_corr.values, 0)  # Убираем диагональ
    
    strong = (abs_corr > 0.7).sum().sum() // 2
    medium = ((abs_corr > 0.3) & (abs_corr <= 0.7)).sum().sum() // 2
    weak = ((abs_corr > 0) & (abs_corr <= 0.3)).sum().sum() // 2
    
    print(f"\nСильные корреляции (|r| > 0.7):   {strong} пар")
    print(f"Средние корреляции (0.3 < |r| <= 0.7): {medium} пар")
    print(f"Слабые корреляции (|r| <= 0.3):   {weak} пар")
    
    # Максимальная корреляция
    max_corr_idx = abs_corr.stack().idxmax()
    max_corr_value = corr_matrix_ru.loc[max_corr_idx]
    print(f"\nМаксимальная корреляция:")
    print(f"   {max_corr_idx[0]} ↔ {max_corr_idx[1]}")
    print(f"   r = {max_corr_value:+.4f}")
    
    print("\n" + "="*80)
    print("METHOD 10 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица:  {matrix_csv_path}")
    print(f"   Топ-10:   {top_pairs_csv_path}")
    print(f"   Heatmap:  {heatmap_path}")
    print()


def extract_top_pairs_fixed(corr_matrix_ru, n=10):
    """
    Извлекает топ-N пар признаков по абсолютному значению корреляции.
    Работает с уже переименованной матрицей (русские названия).
    """
    
    # Создаём список всех пар (без дубликатов и диагонали)
    pairs = []
    features = corr_matrix_ru.index.tolist()
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i < j:  # Только верхняя треугольная матрица
                corr_value = corr_matrix_ru.loc[feat1, feat2]
                pairs.append({
                    'Признак 1': feat1,
                    'Признак 2': feat2,
                    'Корреляция': corr_value,
                    'Абсолютное значение': abs(corr_value)
                })
    
    # Сортировка по абсолютному значению
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('Абсолютное значение', ascending=False).head(n).reset_index(drop=True)
    
    return pairs_df


def create_correlation_heatmap(corr_matrix_ru, save_path):
    """
    Создаёт красивый heatmap корреляционной матрицы.
    С РУЧНЫМ добавлением текста для гарантии отображения.
    """
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Создание heatmap БЕЗ аннотаций
    im = ax.imshow(corr_matrix_ru, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(corr_matrix_ru.columns)))
    ax.set_yticks(np.arange(len(corr_matrix_ru.index)))
    ax.set_xticklabels(corr_matrix_ru.columns, rotation=45, ha='right', fontsize=11, weight='bold')
    ax.set_yticklabels(corr_matrix_ru.index, fontsize=11, weight='bold')
    
    # РУЧНОЕ добавление текста в каждую ячейку
    for i in range(len(corr_matrix_ru.index)):
        for j in range(len(corr_matrix_ru.columns)):
            value = corr_matrix_ru.iloc[i, j]
            
            # Выбор цвета текста (белый на тёмном фоне, чёрный на светлом)
            text_color = 'white' if abs(value) > 0.5 else 'black'
            
            # Добавление текста
            text = ax.text(
                j, i, f'{value:.3f}',
                ha='center', va='center',
                color=text_color,
                fontsize=9,
                weight='bold'
            )
    
    # Сетка
    ax.set_xticks(np.arange(len(corr_matrix_ru.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix_ru.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Корреляция Пирсона', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title(
        'Матрица корреляций Пирсона (числовые признаки)', 
        fontsize=18, 
        fontweight='bold', 
        pad=20
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()

