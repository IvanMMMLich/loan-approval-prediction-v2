"""
Step 2.1: Анализ числовых признаков
Полный анализ всех 7 числовых признаков с консолидированными результатами
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.numeric_analysis import (
    calculate_descriptive_stats,
    calculate_skewness_kurtosis,
    detect_outliers_iqr,
    plot_distribution_single,
    plot_boxplot_single,
    plot_class_comparison_single
)

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step2_eda' / 'numeric_features'
TARGET_COL = 'loan_status'

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
    print("\n" + "="*70)
    print("STEP 2.1: АНАЛИЗ ЧИСЛОВЫХ ПРИЗНАКОВ")
    print("="*70 + "\n")
    
    # Загрузка данных
    print("Загрузка данных...")
    df = pd.read_csv(DATA_FILE)
    print(f"Загружено: {df.shape[0]:,} строк x {df.shape[1]} столбцов\n")
    
    # Создание базовых папок
    tables_dir = RESULTS_DIR / 'tables'
    figures_base_dir = RESULTS_DIR / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Списки для консолидированных таблиц
    all_stats = []
    all_skew_kurt = []
    all_outliers = []
    
    # Проход по всем признакам
    print(f"Анализ {len(NUMERIC_FEATURES)} числовых признаков:\n")
    
    for i, feature in enumerate(NUMERIC_FEATURES, 1):
        print(f"   [{i}/{len(NUMERIC_FEATURES)}] {feature} ({FEATURE_NAMES_RU[feature]})...", end=' ')
        
        # Создание папки для графиков
        feature_figures_dir = figures_base_dir / feature
        feature_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Описательная статистика
        stats_df = calculate_descriptive_stats(
            df, [feature],
            save_table_image_path=feature_figures_dir / '01_descriptive_stats_table.png'
        )
        all_stats.append(stats_df)
        
        # 2. Асимметрия и эксцесс
        skew_kurt_df = calculate_skewness_kurtosis(
            df, [feature],
            save_table_image_path=feature_figures_dir / '02_skewness_kurtosis_table.png'
        )
        all_skew_kurt.append(skew_kurt_df)
        
        # 3. Распределение (Histogram + KDE)
        plot_distribution_single(
            df, feature,
            save_path=feature_figures_dir / '03_distribution_hist_kde.png'
        )
        
        # 4. Boxplot
        plot_boxplot_single(
            df, feature,
            save_path=feature_figures_dir / '04_boxplot.png'
        )
        
        # 5. Сравнение по классам (ГЛАВНЫЙ!)
        plot_class_comparison_single(
            df, feature, TARGET_COL,
            save_path=feature_figures_dir / '05_class_comparison_kde.png'
        )
        
        # 6. Выбросы IQR
        outliers_df = detect_outliers_iqr(
            df, [feature],
            save_table_image_path=feature_figures_dir / '06_outliers_table.png'
        )
        all_outliers.append(outliers_df)
        
        print("OK")
    
    print()
    
    # Создание консолидированных таблиц
    print("Создание консолидированных таблиц...")
    
    # 1. Описательная статистика
    consolidated_stats = pd.concat(all_stats, ignore_index=True)
    consolidated_stats['Признак'] = consolidated_stats['Признак'].map(FEATURE_NAMES_RU)
    consolidated_stats.to_csv(
        tables_dir / 'consolidated_descriptive_stats.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print("   OK: consolidated_descriptive_stats.csv")
    
    # 2. Асимметрия и эксцесс
    consolidated_skew_kurt = pd.concat(all_skew_kurt, ignore_index=True)
    consolidated_skew_kurt['Признак'] = consolidated_skew_kurt['Признак'].map(FEATURE_NAMES_RU)
    consolidated_skew_kurt.to_csv(
        tables_dir / 'consolidated_skewness_kurtosis.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print("   OK: consolidated_skewness_kurtosis.csv")
    
    # 3. Выбросы
    consolidated_outliers = pd.concat(all_outliers, ignore_index=True)
    consolidated_outliers['Признак'] = consolidated_outliers['Признак'].map(FEATURE_NAMES_RU)
    consolidated_outliers.to_csv(
        tables_dir / 'consolidated_outliers.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print("   OK: consolidated_outliers.csv")
    
    print()
    
    # Итоги
    print("="*70)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("="*70 + "\n")
    
    print("Результаты:")
    print(f"   Консолидированные таблицы: {tables_dir}/")
    print(f"      - consolidated_descriptive_stats.csv (все 7 признаков)")
    print(f"      - consolidated_skewness_kurtosis.csv (все 7 признаков)")
    print(f"      - consolidated_outliers.csv (все 7 признаков)")
    print()
    print(f"   Графики по признакам: {figures_base_dir}/")
    print(f"      - {len(NUMERIC_FEATURES)} папок x 6 PNG = {len(NUMERIC_FEATURES) * 6} графиков")
    print()

if __name__ == '__main__':
    main()