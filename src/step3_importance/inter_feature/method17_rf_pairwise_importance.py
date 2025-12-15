"""
Method 17: Random Forest Pairwise Importance (Важность пар признаков)

Описание:
Обучает Random Forest на каждой паре признаков отдельно и измеряет ROC-AUC.
Показывает, насколько хорошо каждая пара признаков предсказывает target.

Применяется: Ко всем 11 признакам (7 числовых + 4 категориальных)
Результат: Матрица 11×11, где каждая ячейка = ROC-AUC модели на этой паре

Интерпретация:
- ROC-AUC = 0.5: пара не лучше случайного угадывания
- ROC-AUC = 0.6-0.7: слабая предсказательная сила
- ROC-AUC = 0.7-0.8: средняя предсказательная сила
- ROC-AUC > 0.8: сильная предсказательная сила

Дополнительно:
- Вычисляем "синергию" = ROC-AUC пары - max(ROC-AUC признака 1, ROC-AUC признака 2)
- Синергия > 0: признаки вместе работают лучше, чем по отдельности
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'model_based' / 'method17_rf_pairwise'

# Все признаки
NUMERIC_FEATURES = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length'
]

CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = 'loan_status'

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка',
    'loan_percent_income': '% дохода на кредит',
    'cb_person_cred_hist_length': 'Длина кред. истории',
    'person_home_ownership': 'Владение жильём',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта'
}

# Размер выборки для ускорения
SAMPLE_SIZE = 10000

# ============================================================================
# ФУНКЦИИ
# ============================================================================

def prepare_data(df):
    """Подготавливает данные: кодирует категориальные признаки."""
    
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    
    return df_encoded, label_encoders


def get_single_feature_auc(X, y, feature):
    """Вычисляет ROC-AUC для одного признака."""
    
    X_single = X[[feature]].values
    
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    scores = cross_val_score(rf, X_single, y, cv=3, scoring='roc_auc')
    
    return scores.mean()


def get_pair_auc(X, y, feature1, feature2):
    """Вычисляет ROC-AUC для пары признаков."""
    
    X_pair = X[[feature1, feature2]].values
    
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    scores = cross_val_score(rf, X_pair, y, cv=3, scoring='roc_auc')
    
    return scores.mean()


def compute_pairwise_importance(df, features, target):
    """Вычисляет матрицу ROC-AUC для всех пар признаков."""
    
    n_features = len(features)
    
    X = df[features]
    y = df[target]
    
    # Сначала вычисляем AUC для каждого признака отдельно
    print("Вычисление ROC-AUC для отдельных признаков...\n")
    single_auc = {}
    
    for i, feat in enumerate(features):
        print(f"   {i+1}/{n_features}: {feat}")
        single_auc[feat] = get_single_feature_auc(X, y, feat)
    
    print("\n" + "-"*40)
    print("ROC-AUC отдельных признаков:")
    print("-"*40)
    for feat, auc in sorted(single_auc.items(), key=lambda x: x[1], reverse=True):
        print(f"   {FEATURE_NAMES_RU.get(feat, feat):30} | AUC = {auc:.4f}")
    
    # Теперь вычисляем AUC для каждой пары
    print("\n" + "="*40)
    print("Вычисление ROC-AUC для пар признаков...")
    print("="*40 + "\n")
    
    auc_matrix = np.zeros((n_features, n_features))
    synergy_matrix = np.zeros((n_features, n_features))
    
    total_pairs = n_features * (n_features - 1) // 2
    current_pair = 0
    
    for i in range(n_features):
        # Диагональ = AUC одного признака
        auc_matrix[i, i] = single_auc[features[i]]
        synergy_matrix[i, i] = 0
        
        for j in range(i + 1, n_features):
            current_pair += 1
            print(f"   Пара {current_pair}/{total_pairs}: {features[i]} + {features[j]}", end='\r')
            
            pair_auc = get_pair_auc(X, y, features[i], features[j])
            
            auc_matrix[i, j] = pair_auc
            auc_matrix[j, i] = pair_auc  # Симметричная матрица
            
            # Синергия = AUC пары - max(AUC признака 1, AUC признака 2)
            max_single = max(single_auc[features[i]], single_auc[features[j]])
            synergy = pair_auc - max_single
            
            synergy_matrix[i, j] = synergy
            synergy_matrix[j, i] = synergy
    
    print()
    
    # Создаём DataFrame с русскими названиями
    features_ru = [FEATURE_NAMES_RU.get(f, f) for f in features]
    
    auc_df = pd.DataFrame(auc_matrix, index=features_ru, columns=features_ru)
    synergy_df = pd.DataFrame(synergy_matrix, index=features_ru, columns=features_ru)
    single_auc_df = pd.DataFrame({
        'Признак': [FEATURE_NAMES_RU.get(f, f) for f in features],
        'ROC-AUC': [single_auc[f] for f in features]
    }).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
    
    return auc_df, synergy_df, single_auc_df


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 17: RANDOM FOREST PAIRWISE IMPORTANCE")
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
    
    # Сэмплирование для ускорения
    if len(df) > SAMPLE_SIZE:
        print(f"Сэмплирование до {SAMPLE_SIZE:,} строк для ускорения...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"После сэмплирования: {len(df):,} строк\n")
    
    # Подготовка данных
    print("Кодирование категориальных признаков...")
    df_encoded, _ = prepare_data(df)
    print("Готово!\n")
    
    print(f"Всего признаков: {len(ALL_FEATURES)}")
    print(f"Количество пар: {len(ALL_FEATURES) * (len(ALL_FEATURES) - 1) // 2}\n")
    
    # Вычисление матриц
    auc_matrix, synergy_matrix, single_auc = compute_pairwise_importance(
        df_encoded, ALL_FEATURES, TARGET
    )
    
    # Сохранение матрицы ROC-AUC
    auc_csv_path = tables_dir / 'rf_pairwise_auc_matrix_11x11.csv'
    auc_matrix.to_csv(auc_csv_path, encoding='utf-8-sig')
    print(f"\nМатрица ROC-AUC сохранена: {auc_csv_path}")
    
    # Сохранение матрицы синергии
    synergy_csv_path = tables_dir / 'rf_pairwise_synergy_matrix_11x11.csv'
    synergy_matrix.to_csv(synergy_csv_path, encoding='utf-8-sig')
    print(f"Матрица синергии сохранена: {synergy_csv_path}")
    
    # Сохранение AUC отдельных признаков
    single_csv_path = tables_dir / 'rf_single_feature_auc.csv'
    single_auc.to_csv(single_csv_path, index=False, encoding='utf-8-sig')
    print(f"AUC отдельных признаков сохранены: {single_csv_path}\n")
    
    # Извлечение топ пар
    print("Извлечение топ-10 пар...\n")
    top_pairs = extract_top_pairs(auc_matrix, synergy_matrix, n=10)
    
    print("ТОП-10 ПАР ПО ROC-AUC:\n")
    for i, row in top_pairs.iterrows():
        print(f"{i+1:2d}. {row['Признак 1']:25} + {row['Признак 2']:25} | AUC = {row['ROC-AUC']:.4f} | Синергия = {row['Синергия']:+.4f}")
    print()
    
    # Сохранение топ пар
    top_pairs_csv_path = tables_dir / 'rf_pairwise_top_pairs.csv'
    top_pairs.to_csv(top_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Топ пары сохранены: {top_pairs_csv_path}\n")
    
    # Создание heatmap ROC-AUC
    print("Создание heatmap ROC-AUC...")
    auc_heatmap_path = figures_dir / 'rf_pairwise_auc_heatmap.png'
    create_auc_heatmap(auc_matrix, auc_heatmap_path)
    
    # Создание heatmap синергии
    print("Создание heatmap синергии...")
    synergy_heatmap_path = figures_dir / 'rf_pairwise_synergy_heatmap.png'
    create_synergy_heatmap(synergy_matrix, synergy_heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    # Убираем диагональ для статистики
    auc_values = auc_matrix.values.copy()
    np.fill_diagonal(auc_values, 0)
    
    # Только верхний треугольник (уникальные пары)
    upper_triangle = auc_values[np.triu_indices(len(auc_values), k=1)]
    
    print(f"\nСтатистика по парам (всего {len(upper_triangle)} пар):")
    print(f"   Максимальный ROC-AUC: {upper_triangle.max():.4f}")
    print(f"   Средний ROC-AUC:      {upper_triangle.mean():.4f}")
    print(f"   Минимальный ROC-AUC:  {upper_triangle.min():.4f}")
    
    # Синергия
    synergy_values = synergy_matrix.values.copy()
    np.fill_diagonal(synergy_values, 0)
    synergy_upper = synergy_values[np.triu_indices(len(synergy_values), k=1)]
    
    positive_synergy = (synergy_upper > 0.01).sum()
    negative_synergy = (synergy_upper < -0.01).sum()
    neutral = len(synergy_upper) - positive_synergy - negative_synergy
    
    print(f"\nСинергия между признаками:")
    print(f"   Положительная (>0.01):  {positive_synergy} пар (признаки усиливают друг друга)")
    print(f"   Нейтральная:            {neutral} пар")
    print(f"   Отрицательная (<-0.01): {negative_synergy} пар (признаки дублируют друг друга)")
    
    print("\n" + "="*80)
    print("METHOD 17 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   AUC матрица:     {auc_csv_path}")
    print(f"   Синергия матрица: {synergy_csv_path}")
    print(f"   Одиночные AUC:   {single_csv_path}")
    print(f"   Топ пары:        {top_pairs_csv_path}")
    print(f"   AUC heatmap:     {auc_heatmap_path}")
    print(f"   Синергия heatmap: {synergy_heatmap_path}")
    print()


def extract_top_pairs(auc_matrix, synergy_matrix, n=10):
    """Извлекает топ-N пар по ROC-AUC."""
    
    pairs = []
    features = auc_matrix.index.tolist()
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i < j:
                auc = auc_matrix.loc[feat1, feat2]
                synergy = synergy_matrix.loc[feat1, feat2]
                pairs.append({
                    'Признак 1': feat1,
                    'Признак 2': feat2,
                    'ROC-AUC': auc,
                    'Синергия': synergy
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('ROC-AUC', ascending=False).head(n).reset_index(drop=True)
    
    return pairs_df


def create_auc_heatmap(auc_matrix, save_path):
    """Создаёт heatmap ROC-AUC."""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    im = ax.imshow(auc_matrix, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=0.9)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(auc_matrix.columns)))
    ax.set_yticks(np.arange(len(auc_matrix.index)))
    ax.set_xticklabels(auc_matrix.columns, rotation=45, ha='right', fontsize=10, weight='bold')
    ax.set_yticklabels(auc_matrix.index, fontsize=10, weight='bold')
    
    # Ручное добавление текста
    for i in range(len(auc_matrix.index)):
        for j in range(len(auc_matrix.columns)):
            value = auc_matrix.iloc[i, j]
            text_color = 'white' if value > 0.75 or value < 0.55 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=8, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(auc_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(auc_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('ROC-AUC', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title('Random Forest Pairwise ROC-AUC\n(насколько хорошо пара признаков предсказывает target)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"AUC heatmap сохранен: {save_path}")


def create_synergy_heatmap(synergy_matrix, save_path):
    """Создаёт heatmap синергии."""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Симметричная шкала для синергии
    max_abs = max(abs(synergy_matrix.values.min()), abs(synergy_matrix.values.max()))
    max_abs = max(max_abs, 0.1)  # Минимум 0.1 для видимости
    
    im = ax.imshow(synergy_matrix, cmap='RdBu_r', aspect='auto', vmin=-max_abs, vmax=max_abs)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(synergy_matrix.columns)))
    ax.set_yticks(np.arange(len(synergy_matrix.index)))
    ax.set_xticklabels(synergy_matrix.columns, rotation=45, ha='right', fontsize=10, weight='bold')
    ax.set_yticklabels(synergy_matrix.index, fontsize=10, weight='bold')
    
    # Ручное добавление текста
    for i in range(len(synergy_matrix.index)):
        for j in range(len(synergy_matrix.columns)):
            value = synergy_matrix.iloc[i, j]
            text_color = 'white' if abs(value) > max_abs * 0.5 else 'black'
            ax.text(j, i, f'{value:+.3f}', ha='center', va='center',
                   color=text_color, fontsize=8, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(synergy_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(synergy_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Синергия (AUC пары - max AUC одного)', rotation=270, labelpad=20, fontsize=11, weight='bold')
    
    # Заголовок
    ax.set_title('Синергия пар признаков\n(красный = усиливают друг друга, синий = дублируют)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Синергия heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()
