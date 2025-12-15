"""
Step 2.2: Анализ категориальных признаков
Полный анализ всех 4 категориальных признаков с консолидированными результатами
"""

import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.categorical_analysis import (
    calculate_frequency_distribution,
    calculate_approval_rate_by_category,
    plot_frequency_bar,
    plot_approval_rate_bar,
    plot_stacked_bar_by_class
)

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step2_eda' / 'categorical_features'
TARGET_COL = 'loan_status'

# Категориальные признаки
CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_home_ownership': 'Владение жильем',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта в истории'
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 2.2: АНАЛИЗ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
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
    all_frequencies = []
    all_approval_rates = []
    
    # Проход по всем признакам
    print(f"Анализ {len(CATEGORICAL_FEATURES)} категориальных признаков:\n")
    
    for i, feature in enumerate(CATEGORICAL_FEATURES, 1):
        print(f"   [{i}/{len(CATEGORICAL_FEATURES)}] {feature} ({FEATURE_NAMES_RU[feature]})...", end=' ')
        
        # Создание папки для графиков
        feature_figures_dir = figures_base_dir / feature
        feature_figures_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Частотное распределение
        freq_df = calculate_frequency_distribution(
            df, feature,
            save_table_image_path=feature_figures_dir / '01_frequency_table.png'
        )
        # Добавляем колонку с названием признака для консолидации
        freq_df.insert(0, 'Признак', FEATURE_NAMES_RU[feature])
        all_frequencies.append(freq_df)
        
        # 2. Approval rate по категориям (КЛЮЧЕВОЕ!)
        approval_df = calculate_approval_rate_by_category(
            df, feature, TARGET_COL,
            save_table_image_path=feature_figures_dir / '02_approval_rate_table.png'
        )
        # Добавляем колонку с названием признака
        approval_df.insert(0, 'Признак', FEATURE_NAMES_RU[feature])
        all_approval_rates.append(approval_df)
        
        # 3. Bar chart частот
        plot_frequency_bar(
            df, feature,
            save_path=feature_figures_dir / '03_frequency_bar.png'
        )
        
        # 4. Bar chart approval rate (ГЛАВНЫЙ!)
        plot_approval_rate_bar(
            df, feature, TARGET_COL,
            save_path=feature_figures_dir / '04_approval_rate_bar.png'
        )
        
        # 5. Stacked bar chart
        plot_stacked_bar_by_class(
            df, feature, TARGET_COL,
            save_path=feature_figures_dir / '05_stacked_bar.png'
        )
        
        print("OK")
    
    print()
    
    # Создание консолидированных таблиц
    print("Создание консолидированных таблиц...")
    
    # 1. Частотное распределение
    consolidated_freq = pd.concat(all_frequencies, ignore_index=True)
    consolidated_freq.to_csv(
        tables_dir / 'consolidated_frequency_distribution.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print("   OK: consolidated_frequency_distribution.csv")
    
    # 2. Approval rate
    consolidated_approval = pd.concat(all_approval_rates, ignore_index=True)
    consolidated_approval.to_csv(
        tables_dir / 'consolidated_approval_rate.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print("   OK: consolidated_approval_rate.csv")
    
    print()
    
    # Итоги
    print("="*70)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("="*70 + "\n")
    
    print("Результаты:")
    print(f"   Консолидированные таблицы: {tables_dir}/")
    print(f"      - consolidated_frequency_distribution.csv (все 4 признака)")
    print(f"      - consolidated_approval_rate.csv (все 4 признака)")
    print()
    print(f"   Графики по признакам: {figures_base_dir}/")
    print(f"      - {len(CATEGORICAL_FEATURES)} папок x 5 PNG = {len(CATEGORICAL_FEATURES) * 5} графиков")
    print()

if __name__ == '__main__':
    main()