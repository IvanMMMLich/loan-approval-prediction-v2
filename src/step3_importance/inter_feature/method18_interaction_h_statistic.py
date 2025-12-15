"""
Method 18: Feature Interaction H-statistic (Взаимодействие признаков)

ПОЧЕМУ H-STATISTIC ВМЕСТО SHAP INTERACTION:
============================================
Изначально планировалось использовать SHAP Interaction Values, но библиотека
`shap` не входит в зависимости проекта (requirements.txt).

Альтернатива - измерение взаимодействия через анализ предсказаний:
- Реализуется через sklearn (уже установлен)
- Измеряет силу взаимодействия между признаками
- Даёт матрицу 11×11

Интерпретация:
- H = 0: нет взаимодействия (эффекты аддитивны)
- H = 0.05-0.15: слабое взаимодействие
- H = 0.15-0.30: среднее взаимодействие
- H > 0.30: сильное взаимодействие
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'train.csv'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step3_importance' / 'inter_feature_correlations' / 'model_based' / 'method18_h_statistic'

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

# Размер выборки
SAMPLE_SIZE = 5000

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


def compute_interaction_strength(model, X, feature_i, feature_j, n_samples=500):
    """
    Вычисляет силу взаимодействия между двумя признаками.
    
    Метод: сравниваем эффект признака i при разных значениях признака j.
    Если эффект i зависит от j — есть взаимодействие.
    """
    
    X_sample = X.sample(n=min(n_samples, len(X)), random_state=42).copy()
    
    # Квантили для разделения на группы
    q_low_i, q_high_i = X_sample[feature_i].quantile([0.25, 0.75])
    q_low_j, q_high_j = X_sample[feature_j].quantile([0.25, 0.75])
    
    # Если квантили одинаковые (для категориальных), используем медиану
    if q_low_i == q_high_i:
        q_low_i = X_sample[feature_i].min()
        q_high_i = X_sample[feature_i].max()
    if q_low_j == q_high_j:
        q_low_j = X_sample[feature_j].min()
        q_high_j = X_sample[feature_j].max()
    
    # Создаём 4 версии данных
    X_low_low = X_sample.copy()
    X_low_low[feature_i] = q_low_i
    X_low_low[feature_j] = q_low_j
    
    X_low_high = X_sample.copy()
    X_low_high[feature_i] = q_low_i
    X_low_high[feature_j] = q_high_j
    
    X_high_low = X_sample.copy()
    X_high_low[feature_i] = q_high_i
    X_high_low[feature_j] = q_low_j
    
    X_high_high = X_sample.copy()
    X_high_high[feature_i] = q_high_i
    X_high_high[feature_j] = q_high_j
    
    # Предсказания для каждой комбинации
    pred_low_low = model.predict_proba(X_low_low)[:, 1].mean()
    pred_low_high = model.predict_proba(X_low_high)[:, 1].mean()
    pred_high_low = model.predict_proba(X_high_low)[:, 1].mean()
    pred_high_high = model.predict_proba(X_high_high)[:, 1].mean()
    
    # Эффект признака i при j=low
    effect_i_when_j_low = pred_high_low - pred_low_low
    
    # Эффект признака i при j=high
    effect_i_when_j_high = pred_high_high - pred_low_high
    
    # Взаимодействие = насколько эффект i зависит от j
    interaction = abs(effect_i_when_j_high - effect_i_when_j_low)
    
    # Нормализация: делим на средний абсолютный эффект
    avg_effect = (abs(effect_i_when_j_low) + abs(effect_i_when_j_high)) / 2
    
    if avg_effect < 0.001:
        # Если эффекта почти нет, смотрим на сырое взаимодействие
        # Нормализуем по разбросу предсказаний
        all_preds = [pred_low_low, pred_low_high, pred_high_low, pred_high_high]
        pred_range = max(all_preds) - min(all_preds)
        if pred_range < 0.001:
            return 0.0
        h_stat = interaction / pred_range
    else:
        h_stat = interaction / (avg_effect + 0.01)
    
    return min(h_stat, 1.0)


def compute_interaction_matrix(model, X, features):
    """Вычисляет матрицу взаимодействий для всех пар признаков."""
    
    n_features = len(features)
    h_matrix = np.zeros((n_features, n_features))
    
    total_pairs = n_features * (n_features - 1) // 2
    current_pair = 0
    
    for i in range(n_features):
        h_matrix[i, i] = 0  # Диагональ = 0
        
        for j in range(i + 1, n_features):
            current_pair += 1
            print(f"   Пара {current_pair}/{total_pairs}: {features[i]} + {features[j]}", end='\r')
            
            h_stat = compute_interaction_strength(model, X, features[i], features[j])
            
            h_matrix[i, j] = h_stat
            h_matrix[j, i] = h_stat  # Симметричная матрица
    
    print()
    
    # Создаём DataFrame с русскими названиями
    features_ru = [FEATURE_NAMES_RU.get(f, f) for f in features]
    h_df = pd.DataFrame(h_matrix, index=features_ru, columns=features_ru)
    
    return h_df


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*80)
    print("METHOD 18: H-STATISTIC (FEATURE INTERACTION)")
    print("="*80)
    print("\nПримечание: Используется H-statistic вместо SHAP Interaction,")
    print("т.к. библиотека shap не входит в зависимости проекта.\n")
    
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
        print(f"Сэмплирование до {SAMPLE_SIZE:,} строк...")
        df = df.sample(n=SAMPLE_SIZE, random_state=42)
        print(f"После сэмплирования: {len(df):,} строк\n")
    
    # Подготовка данных
    print("Кодирование категориальных признаков...")
    df_encoded, _ = prepare_data(df)
    print("Готово!\n")
    
    X = df_encoded[ALL_FEATURES]
    y = df_encoded[TARGET]
    
    print(f"Всего признаков: {len(ALL_FEATURES)}")
    print(f"Количество пар: {len(ALL_FEATURES) * (len(ALL_FEATURES) - 1) // 2}\n")
    
    # Обучение модели
    print("Обучение Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)
    print("Модель обучена!\n")
    
    # Вычисление матрицы взаимодействий
    print("Вычисление H-statistic для всех пар...")
    print("(Это может занять 2-3 минуты)\n")
    
    h_matrix = compute_interaction_matrix(model, X, ALL_FEATURES)
    
    print("\nГотово!\n")
    
    # Сохранение матрицы
    matrix_csv_path = tables_dir / 'h_statistic_matrix_11x11.csv'
    h_matrix.to_csv(matrix_csv_path, encoding='utf-8-sig')
    print(f"Матрица сохранена: {matrix_csv_path}")
    
    # Извлечение топ пар
    print("\nИзвлечение топ-15 пар...\n")
    top_pairs = extract_top_pairs(h_matrix, n=15)
    
    print("ТОП-15 ПАР ПО СИЛЕ ВЗАИМОДЕЙСТВИЯ:\n")
    for i, row in top_pairs.iterrows():
        strength = get_strength_label(row['H-statistic'])
        print(f"{i+1:2d}. {row['Признак 1']:25} × {row['Признак 2']:25} | H = {row['H-statistic']:.4f} ({strength})")
    print()
    
    # Сохранение топ пар
    top_pairs_csv_path = tables_dir / 'h_statistic_top_pairs.csv'
    top_pairs.to_csv(top_pairs_csv_path, index=False, encoding='utf-8-sig')
    print(f"Топ пары сохранены: {top_pairs_csv_path}\n")
    
    # Создание heatmap
    print("Создание heatmap...")
    heatmap_path = figures_dir / 'h_statistic_heatmap.png'
    create_heatmap(h_matrix, heatmap_path)
    
    # Статистика
    print("\n" + "="*80)
    print("СТАТИСТИКА:")
    print("="*80)
    
    h_values = h_matrix.values.copy()
    np.fill_diagonal(h_values, 0)
    upper_triangle = h_values[np.triu_indices(len(h_values), k=1)]
    
    strong = (upper_triangle > 0.30).sum()
    medium = ((upper_triangle > 0.15) & (upper_triangle <= 0.30)).sum()
    weak = ((upper_triangle > 0.05) & (upper_triangle <= 0.15)).sum()
    very_weak = (upper_triangle <= 0.05).sum()
    
    print(f"\nСильное взаимодействие (H > 0.30):     {strong} пар")
    print(f"Среднее взаимодействие (0.15-0.30):    {medium} пар")
    print(f"Слабое взаимодействие (0.05-0.15):     {weak} пар")
    print(f"Очень слабое взаимодействие (<=0.05):  {very_weak} пар")
    
    print(f"\nМаксимальное H: {upper_triangle.max():.4f}")
    print(f"Среднее H:      {upper_triangle.mean():.4f}")
    print(f"Минимальное H:  {upper_triangle.min():.4f}")
    
    print("\n" + "="*80)
    print("METHOD 18 ЗАВЕРШЕН")
    print("="*80 + "\n")
    
    print(f"Результаты:")
    print(f"   Матрица:  {matrix_csv_path}")
    print(f"   Топ пары: {top_pairs_csv_path}")
    print(f"   Heatmap:  {heatmap_path}")
    print()


def extract_top_pairs(h_matrix, n=15):
    """Извлекает топ-N пар по H-statistic."""
    
    pairs = []
    features = h_matrix.index.tolist()
    
    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i < j:
                h_stat = h_matrix.loc[feat1, feat2]
                pairs.append({
                    'Признак 1': feat1,
                    'Признак 2': feat2,
                    'H-statistic': h_stat,
                    'Сила взаимодействия': get_strength_label(h_stat)
                })
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('H-statistic', ascending=False).head(n).reset_index(drop=True)
    
    return pairs_df


def get_strength_label(h):
    """Возвращает текстовое описание силы взаимодействия."""
    if h > 0.30:
        return "сильное"
    elif h > 0.15:
        return "среднее"
    elif h > 0.05:
        return "слабое"
    else:
        return "очень слабое"


def create_heatmap(h_matrix, save_path):
    """Создаёт heatmap H-statistic."""
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Определяем максимум для шкалы
    max_val = max(h_matrix.values.max(), 0.3)
    
    im = ax.imshow(h_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=max_val)
    
    # Настройка осей
    ax.set_xticks(np.arange(len(h_matrix.columns)))
    ax.set_yticks(np.arange(len(h_matrix.index)))
    ax.set_xticklabels(h_matrix.columns, rotation=45, ha='right', fontsize=10, weight='bold')
    ax.set_yticklabels(h_matrix.index, fontsize=10, weight='bold')
    
    # Ручное добавление текста
    for i in range(len(h_matrix.index)):
        for j in range(len(h_matrix.columns)):
            value = h_matrix.iloc[i, j]
            text_color = 'white' if value > max_val * 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=8, weight='bold')
    
    # Сетка
    ax.set_xticks(np.arange(len(h_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(h_matrix.index)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('H-statistic (сила взаимодействия)', rotation=270, labelpad=20, fontsize=12, weight='bold')
    
    # Заголовок
    ax.set_title('H-statistic: Взаимодействие признаков\n(чем выше H — тем сильнее признаки влияют друг на друга в модели)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Heatmap сохранен: {save_path}")


if __name__ == '__main__':
    main()