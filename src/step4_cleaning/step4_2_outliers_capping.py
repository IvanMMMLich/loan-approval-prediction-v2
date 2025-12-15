"""
Step 4.2: Обработка выбросов (Capping) для моделей деревьев решений

============================================================================
ПОЧЕМУ ИМЕННО CAPPING, А НЕ НОРМАЛИЗАЦИЯ?
============================================================================

Мы планируем использовать ансамблевые модели деревьев решений:
- XGBoost
- LightGBM  
- CatBoost

ОСОБЕННОСТИ ДЕРЕВЬЕВ РЕШЕНИЙ:

    1. НЕ требуют StandardScaler/MinMaxScaler
       Деревья работают с ПОРЯДКОМ значений, а не с абсолютными величинами.
       Split "income > 50000" работает одинаково для сырых и масштабированных данных.
    
    2. НЕ требуют Log-трансформации для skewness 
       Деревья сами разбивают данные на интервалы.
       Асимметрия не мешает — дерево адаптируется к любому распределению.
    
    3. ТРЕБУЮТ удаления явных ошибок данных
       Выбросы, которые являются багами (стаж 123 года), могут создавать
       нерелевантные splits и снижать качество модели.

ЧТО ДЕЛАЕМ В ЭТОМ ФАЙЛЕ:

    Capping (обрезка) двух признаков с явными ошибками:
    
    ┌───────────────────┬─────────────┬─────────────┬─────────────────────────┐
    │ Признак           │ Текущий Max │ Cap до      │ Обоснование             │
    ├───────────────────┼─────────────┼─────────────┼─────────────────────────┤
    │ person_income     │ 1,900,000   │ 99 перц.    │ 1.9M нереален для       │
    │                   │             │             │ subprime заёмщика       │
    ├───────────────────┼─────────────┼─────────────┼─────────────────────────┤
    │ person_emp_length │ 123 года    │ 50 лет      │ 123 года стажа —        │
    │                   │             │             │ физически невозможно    │
    └───────────────────┴─────────────┴─────────────┴─────────────────────────┘

ЧТО НЕ ДЕЛАЕМ (и почему):

    - StandardScaler: деревья не требуют
    - Log-трансформация: деревья сами разбивают на интервалы
    - Удаление выбросов: теряем данные, лучше обрезать

============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step4_cleaning'

# Входные файлы (после step4_1)
TRAIN_INPUT = INPUT_DIR / 'train_step4_1.csv'
TEST_INPUT = INPUT_DIR / 'test_step4_1.csv'

# Конфигурация Capping
CAPPING_CONFIG = {
    'person_income': {
        'method': 'percentile',
        'upper_percentile': 99,
        'lower_percentile': None,  # Не обрезаем снизу
        'reason': 'Max=1.9M нереален для subprime заёмщика (в 33 раза > медианы)'
    },
    'person_emp_length': {
        'method': 'fixed',
        'upper_limit': 50,
        'lower_limit': None,  # Не обрезаем снизу
        'reason': 'Max=123 года стажа — физически невозможно'
    }
}

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы (лет)',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка (%)',
    'loan_percent_income': 'Процент дохода на кредит (%)'
}

# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def calculate_stats(series, name):
    """Вычисляет статистики для признака."""
    return {
        'Признак': name,
        'Min': series.min(),
        'Q1 (25%)': series.quantile(0.25),
        'Медиана': series.median(),
        'Среднее': series.mean(),
        'Q3 (75%)': series.quantile(0.75),
        'P99': series.quantile(0.99),
        'Max': series.max(),
        'Std': series.std(),
        'Skewness': series.skew()
    }


def apply_capping(df, feature, config, reference_df=None):
    """
    Применяет capping к признаку.
    
    Args:
        df: DataFrame для обработки
        feature: название признака
        config: конфигурация capping
        reference_df: DataFrame для вычисления границ (обычно train)
                     Если None, используется df
    
    Returns:
        Series с обрезанными значениями, верхняя граница
    """
    if reference_df is None:
        reference_df = df
    
    series = df[feature].copy()
    upper_cap = None
    lower_cap = None
    
    if config['method'] == 'percentile':
        if config.get('upper_percentile'):
            upper_cap = reference_df[feature].quantile(config['upper_percentile'] / 100)
        if config.get('lower_percentile'):
            lower_cap = reference_df[feature].quantile(config['lower_percentile'] / 100)
    
    elif config['method'] == 'fixed':
        upper_cap = config.get('upper_limit')
        lower_cap = config.get('lower_limit')
    
    # Применяем capping
    if upper_cap is not None:
        series = series.clip(upper=upper_cap)
    if lower_cap is not None:
        series = series.clip(lower=lower_cap)
    
    return series, upper_cap, lower_cap


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 4.2: ОБРАБОТКА ВЫБРОСОВ (CAPPING)")
    print("="*70)
    print("\nОптимизация для моделей деревьев решений (XGBoost/LightGBM/CatBoost)")
    print()
    
    # Создание папок
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / 'tables'
    figures_dir = RESULTS_DIR / 'figures'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. ЗАГРУЗКА ДАННЫХ
    # -------------------------------------------------------------------------
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 40)
    
    train_df = pd.read_csv(TRAIN_INPUT)
    test_df = pd.read_csv(TEST_INPUT)
    
    print(f"\n   Train: {train_df.shape[0]:,} строк × {train_df.shape[1]} столбцов")
    print(f"   Test:  {test_df.shape[0]:,} строк × {test_df.shape[1]} столбцов")
    print(f"\n   Источник: step4_1 (после удаления признаков)")
    print()
    
    # -------------------------------------------------------------------------
    # 2. АНАЛИЗ ДО CAPPING
    # -------------------------------------------------------------------------
    print("2. СТАТИСТИКА ДО CAPPING")
    print("-" * 40)
    
    stats_before = []
    
    for feature in CAPPING_CONFIG.keys():
        feature_ru = FEATURE_NAMES_RU.get(feature, feature)
        stats = calculate_stats(train_df[feature], feature_ru)
        stats_before.append(stats)
        
        config = CAPPING_CONFIG[feature]
        
        print(f"\n    {feature_ru} ({feature})")
        print(f"      Min:      {stats['Min']:,.2f}")
        print(f"      Медиана:  {stats['Медиана']:,.2f}")
        print(f"      Среднее:  {stats['Среднее']:,.2f}")
        print(f"      P99:      {stats['P99']:,.2f}")
        print(f"      Max:      {stats['Max']:,.2f}")
        print(f"      Skewness: {stats['Skewness']:.2f}")
        print(f"      Проблема: {config['reason']}")
    
    print()
    
    # -------------------------------------------------------------------------
    # 3. ПРИМЕНЕНИЕ CAPPING
    # -------------------------------------------------------------------------
    print("3. ПРИМЕНЕНИЕ CAPPING")
    print("-" * 40)
    
    train_capped = train_df.copy()
    test_capped = test_df.copy()
    
    capping_report = []
    capping_bounds = {}  # Сохраняем границы для test
    
    for feature, config in CAPPING_CONFIG.items():
        feature_ru = FEATURE_NAMES_RU.get(feature, feature)
        
        # Считаем сколько значений будет обрезано
        original_values = train_df[feature]
        
        # Применяем capping (границы вычисляем по train!)
        train_capped[feature], upper_cap, lower_cap = apply_capping(
            train_df, feature, config, reference_df=train_df
        )
        
        # Для test используем границы от train (важно!)
        capping_bounds[feature] = {'upper': upper_cap, 'lower': lower_cap}
        test_capped[feature] = test_df[feature].clip(
            lower=lower_cap, 
            upper=upper_cap
        )
        
        # Считаем статистику по обрезке
        if upper_cap:
            n_capped_upper = (original_values > upper_cap).sum()
            pct_capped_upper = n_capped_upper / len(original_values) * 100
        else:
            n_capped_upper = 0
            pct_capped_upper = 0
        
        print(f"\n     {feature_ru} ({feature})")
        print(f"       Метод:           {config['method']}")
        if upper_cap:
            print(f"       Верхняя граница: {upper_cap:,.2f}")
        print(f"       Обрезано:        {n_capped_upper:,} значений ({pct_capped_upper:.2f}%)")
        print(f"       Было Max:        {original_values.max():,.2f}")
        print(f"       Стало Max:       {train_capped[feature].max():,.2f}")
        
        capping_report.append({
            'Признак': feature_ru,
            'Признак (англ)': feature,
            'Метод': config['method'],
            'Верхняя граница': upper_cap,
            'Нижняя граница': lower_cap,
            'Обрезано значений': n_capped_upper,
            'Процент обрезанных': f"{pct_capped_upper:.2f}%",
            'Max до': original_values.max(),
            'Max после': train_capped[feature].max(),
            'Причина': config['reason']
        })
    
    print()
    
    # -------------------------------------------------------------------------
    # 4. АНАЛИЗ ПОСЛЕ CAPPING
    # -------------------------------------------------------------------------
    print("4. СТАТИСТИКА ПОСЛЕ CAPPING")
    print("-" * 40)
    
    stats_after = []
    
    for feature in CAPPING_CONFIG.keys():
        feature_ru = FEATURE_NAMES_RU.get(feature, feature)
        stats = calculate_stats(train_capped[feature], feature_ru)
        stats_after.append(stats)
        
        # Находим stats до для сравнения
        stats_b = next(s for s in stats_before if s['Признак'] == feature_ru)
        
        print(f"\n    {feature_ru} ({feature})")
        print(f"      Max:      {stats_b['Max']:>12,.2f}  →  {stats['Max']:>12,.2f}")
        print(f"      Среднее:  {stats_b['Среднее']:>12,.2f}  →  {stats['Среднее']:>12,.2f}")
        print(f"      Skewness: {stats_b['Skewness']:>12.2f}  →  {stats['Skewness']:>12.2f}")
    
    print()
    
    # -------------------------------------------------------------------------
    # 5. ВИЗУАЛИЗАЦИЯ (BOXPLOT ДО/ПОСЛЕ)
    # -------------------------------------------------------------------------
    print("5. СОЗДАНИЕ ВИЗУАЛИЗАЦИИ")
    print("-" * 40)
    
    create_comparison_boxplot(
        train_df, train_capped, 
        list(CAPPING_CONFIG.keys()),
        figures_dir / 'step4_2_capping_comparison.png'
    )
    
    print()
    
    # -------------------------------------------------------------------------
    # 6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # -------------------------------------------------------------------------
    print("6. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 40)
    
    # Сохраняем данные
    train_output = OUTPUT_DIR / 'train_step4_2.csv'
    test_output = OUTPUT_DIR / 'test_step4_2.csv'
    
    train_capped.to_csv(train_output, index=False)
    test_capped.to_csv(test_output, index=False)
    
    print(f"\n   Train: {train_output}")
    print(f"   Test:  {test_output}")
    
    # Сохраняем отчёт
    report_df = pd.DataFrame(capping_report)
    report_path = tables_dir / 'step4_2_capping_report.csv'
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print(f"\n   Отчёт: {report_path}")
    
    # Сохраняем границы capping (для воспроизводимости)
    bounds_df = pd.DataFrame([
        {
            'Признак': f,
            'Верхняя граница': b['upper'],
            'Нижняя граница': b['lower']
        }
        for f, b in capping_bounds.items()
    ])
    bounds_path = tables_dir / 'step4_2_capping_bounds.csv'
    bounds_df.to_csv(bounds_path, index=False, encoding='utf-8-sig')
    
    print(f"   Границы: {bounds_path}")
    
    # -------------------------------------------------------------------------
    # 7. ИТОГОВАЯ СВОДКА
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4.2 ЗАВЕРШЁН")
    print("="*70)
    
    print(f"""
СВОДКА CAPPING:

    ┌───────────────────┬─────────────┬─────────────┬─────────────┐
    │ Признак           │ Max до      │ Max после   │ Обрезано    │
    ├───────────────────┼─────────────┼─────────────┼─────────────┤""")
    
    for r in capping_report:
        print(f"    │ {r['Признак']:17} │ {r['Max до']:>11,.0f} │ {r['Max после']:>11,.0f} │ {r['Процент обрезанных']:>11} │")
    
    print(f"""    └───────────────────┴─────────────┴─────────────┴─────────────┘

ПОЧЕМУ ТОЛЬКО CAPPING (для деревьев решений):

    StandardScaler   — деревья не требуют (работают с порядком)
    Log-трансформация — деревья сами разбивают на интервалы
    Capping          — убираем ошибки данных (стаж 123 года)

ВАЖНО ДЛЯ TEST:

    Границы capping вычислены по TRAIN и применены к TEST.
    Это предотвращает data leakage.

СЛЕДУЮЩИЙ ШАГ:

    step4_3_encoding.py — кодирование категориальных признаков
    (Ordinal для loan_grade, Label для остальных)
    """)
    
    return train_capped, test_capped


def create_comparison_boxplot(df_before, df_after, features, save_path):
    """Создаёт boxplot сравнения до/после capping."""
    
    n_features = len(features)
    fig, axes = plt.subplots(n_features, 2, figsize=(14, 5 * n_features))
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(features):
        feature_ru = FEATURE_NAMES_RU.get(feature, feature)
        
        # До capping
        ax1 = axes[i, 0]
        bp1 = ax1.boxplot(df_before[feature].dropna(), vert=True, patch_artist=True)
        bp1['boxes'][0].set_facecolor('#e74c3c')
        bp1['boxes'][0].set_alpha(0.7)
        ax1.set_title(f'{feature_ru}\nДО capping', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Значение')
        ax1.grid(True, alpha=0.3)
        
        # Добавляем статистики
        stats_before = df_before[feature].describe()
        ax1.text(0.02, 0.98, 
                f"Max: {stats_before['max']:,.0f}\nMedian: {stats_before['50%']:,.0f}",
                transform=ax1.transAxes, verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # После capping
        ax2 = axes[i, 1]
        bp2 = ax2.boxplot(df_after[feature].dropna(), vert=True, patch_artist=True)
        bp2['boxes'][0].set_facecolor('#27ae60')
        bp2['boxes'][0].set_alpha(0.7)
        ax2.set_title(f'{feature_ru}\nПОСЛЕ capping', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Значение')
        ax2.grid(True, alpha=0.3)
        
        # Добавляем статистики
        stats_after = df_after[feature].describe()
        ax2.text(0.02, 0.98,
                f"Max: {stats_after['max']:,.0f}\nMedian: {stats_after['50%']:,.0f}",
                transform=ax2.transAxes, verticalalignment='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Сравнение распределений ДО и ПОСЛЕ Capping\n(Оптимизация для деревьев решений)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n   График сохранён: {save_path}")


if __name__ == '__main__':
    train_capped, test_capped = main()