"""
Step 10: Final Training + Submission

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

1. Обучаем Pipeline на ВСЕХ train данных
2. Метрики и визуализация на Train (ROC-AUC ~0.97)
3. Feature Importance анализ
4. Предсказания на test
5. Создаём submission.csv для Kaggle

============================================================================
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer

from src.utils.analysis import run_full_analysis
from src.utils.metrics import calculate_all_metrics, print_metrics, save_metrics
from src.utils.plotting import plot_all_model_visualizations

warnings.filterwarnings("ignore")

# Добавляем путь к utils
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "step7_final_pipeline"
SUBMISSION_DIR = PROJECT_ROOT / "data" / "submissions"

TRAIN_FILE = DATA_DIR / "train_step4_2.csv"
TEST_FILE = DATA_DIR / "test_step4_2.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Лучшие параметры из Optuna
BEST_PARAMS = {
    "n_estimators": 836,
    "max_depth": 12,
    "learning_rate": 0.033018,
    "num_leaves": 23,
    "min_child_samples": 40,
    "subsample": 0.647898,
    "colsample_bytree": 0.574083,
    "reg_alpha": 4.743694,
    "reg_lambda": 0.408280,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

# Признаки по типам
NUMERIC_FEATURES = [
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

# Ordinal Encoding категории
ORDINAL_CATEGORIES = [
    ["RENT", "OTHER", "MORTGAGE", "OWN"],
    ["DEBTCONSOLIDATION", "MEDICAL", "HOMEIMPROVEMENT", "PERSONAL", "EDUCATION", "VENTURE"],
    ["G", "F", "E", "D", "C", "B", "A"],
    ["Y", "N"],
]

TARGET = "loan_status"


# ============================================================================
# СОЗДАНИЕ PIPELINE
# ============================================================================


def create_pipeline():
    """Создаёт Pipeline."""

    numeric_transformer = ColumnTransformer(
        transformers=[
            (
                "boxcox_income",
                PowerTransformer(method="box-cox", standardize=True),
                ["person_income"],
            ),
            (
                "passthrough_numeric",
                "passthrough",
                ["person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income"],
            ),
        ],
        remainder="drop",
    )

    categorical_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES, handle_unknown="use_encoded_value", unknown_value=-1
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", LGBMClassifier(**BEST_PARAMS))]
    )

    return pipeline


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print("STEP 10: FINAL TRAINING + SUBMISSION")
    print("=" * 70)

    # ========================================================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 70)

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    print(f"\nTrain: {train.shape[0]:,} строк, {train.shape[1]} столбцов")
    print(f"Test:  {test.shape[0]:,} строк, {test.shape[1]} столбцов")

    X_full = train.drop(columns=[TARGET])
    y_full = train[TARGET]
    X_test = test.copy()

    # Проверяем наличие id в test
    if "id" in X_test.columns:
        test_ids = X_test["id"].copy()
        X_test = X_test.drop(columns=["id"])
    else:
        test_ids = pd.Series(range(len(X_test)), name="id")

    print(f"\nПризнаки для обучения: {list(X_full.columns)}")

    # ========================================================================
    # 2. ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ")
    print("-" * 70)

    pipeline = create_pipeline()

    print("\n   Обучение Pipeline на ВСЕХ train данных...")
    pipeline.fit(X_full, y_full)
    print("   ✓ Обучение завершено!")

    # ========================================================================
    # 3. МЕТРИКИ И ВИЗУАЛИЗАЦИЯ (на Train)
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. МЕТРИКИ И ВИЗУАЛИЗАЦИЯ (Train)")
    print("-" * 70)

    y_pred = pipeline.predict(X_full)
    y_pred_proba = pipeline.predict_proba(X_full)[:, 1]

    metrics = calculate_all_metrics(y_full, y_pred, y_pred_proba)
    print_metrics(metrics, title="LightGBM Final (Train)")
    save_metrics(metrics, RESULTS_DIR / "tables" / "final_metrics.csv")

    # Визуализация
    plot_all_model_visualizations(
        y_true=y_full,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        metrics_dict=metrics,
        model_name="LightGBM_Final",
        save_dir=RESULTS_DIR / "figures",
    )

    # ========================================================================
    # 4. FEATURE IMPORTANCE ANALYSIS
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 70)

    # Получаем трансформированные данные для анализа
    X_transformed = pipeline.named_steps["preprocessor"].transform(X_full)

    # Имена признаков после трансформации
    feature_names = (
        ["person_income_boxcox"]
        + ["person_emp_length", "loan_amnt", "loan_int_rate", "loan_percent_income"]
        + CATEGORICAL_FEATURES
    )

    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    # Анализ с Permutation Importance
    run_full_analysis(
        X_train=X_df,
        y_train=y_full.reset_index(drop=True),
        X_val=X_df,
        y_val=y_full.reset_index(drop=True),
        model=pipeline.named_steps["classifier"],
        new_features=["person_income_boxcox"],
        save_dir=RESULTS_DIR,
        top_n=9,
    )

    # ========================================================================
    # 5. ПРЕДСКАЗАНИЯ НА TEST
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. ПРЕДСКАЗАНИЯ НА TEST")
    print("-" * 70)

    # Предсказания
    test_pred = pipeline.predict(X_test)
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n   Предсказания сделаны!")
    print("\n   Распределение предсказаний:")
    print(
        f"      Класс 0 (отказ):    {(test_pred == 0).sum():,} ({(test_pred == 0).mean() * 100:.1f}%)"
    )
    print(
        f"      Класс 1 (одобрен):  {(test_pred == 1).sum():,} ({(test_pred == 1).mean() * 100:.1f}%)"
    )

    print("\n   Статистика вероятностей:")
    print(f"      Min:    {test_proba.min():.4f}")
    print(f"      Max:    {test_proba.max():.4f}")
    print(f"      Mean:   {test_proba.mean():.4f}")
    print(f"      Median: {np.median(test_proba):.4f}")

    # ========================================================================
    # 6. СОЗДАНИЕ SUBMISSION
    # ========================================================================
    print("\n" + "-" * 70)
    print("6. СОЗДАНИЕ SUBMISSION")
    print("-" * 70)

    # Submission с вероятностями (обычно лучше для Kaggle)
    submission_proba = pd.DataFrame({"id": test_ids, "loan_status": test_proba})

    # Submission с классами
    submission_class = pd.DataFrame({"id": test_ids, "loan_status": test_pred})

    # Сохранение
    submission_proba_path = SUBMISSION_DIR / "submission_proba.csv"
    submission_class_path = SUBMISSION_DIR / "submission_class.csv"

    submission_proba.to_csv(submission_proba_path, index=False)
    submission_class.to_csv(submission_class_path, index=False)

    print("\n   ✓ Submission файлы созданы:")
    print(f"      • {submission_proba_path} (вероятности)")
    print(f"      • {submission_class_path} (классы)")

    # Показываем первые строки
    print("\n   Превью submission_proba.csv:")
    print(submission_proba.head(10).to_string(index=False))

    # ========================================================================
    # 7. СОХРАНЕНИЕ МОДЕЛИ
    # ========================================================================
    print("\n" + "-" * 70)
    print("7. СОХРАНЕНИЕ МОДЕЛИ")
    print("-" * 70)

    model_path = RESULTS_DIR / "final_pipeline.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\n   ✓ Pipeline сохранён: {model_path}")

    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "=" * 70)
    print("ГОТОВО! 🎉")
    print("=" * 70)

    print(f"""
    Pipeline обучен на {len(X_full):,} примерах
    Предсказания сделаны для {len(X_test):,} примеров
    Визуализация сохранена
    Submission файлы созданы

    Train метрики:
       ROC-AUC:  {metrics["ROC-AUC"]:.4f}
       Accuracy: {metrics["Accuracy"]:.4f}
       Recall:   {metrics["Recall"]:.4f}
       F1-Score: {metrics["F1-Score"]:.4f}

    Файлы:
       • {RESULTS_DIR / "figures"} (графики)
       • {RESULTS_DIR / "tables"} (таблицы)
       • {submission_proba_path}
       • {submission_class_path}
       • {model_path}



    Ожидаемый ROC-AUC на Kaggle: ~0.955-0.960
    """)


if __name__ == "__main__":
    main()
