"""
Step 6: Hyperparameter Tuning — LightGBM + Optuna

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

Оптимизируем гиперпараметры LightGBM через Optuna.
Целевая метрика: ROC-AUC

Параметры для оптимизации:
- n_estimators (100-1000)
- max_depth (3-12)
- learning_rate (0.01-0.3)
- num_leaves (20-150)
- min_child_samples (5-100)
- subsample (0.5-1.0)
- colsample_bytree (0.5-1.0)
- reg_alpha (0-10)
- reg_lambda (0-10)

============================================================================
"""

import warnings
from pathlib import Path

import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score, train_test_split

from src.utils.metrics import calculate_all_metrics, print_metrics, save_metrics
from src.utils.plotting import plot_all_model_visualizations

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "step5" / "1_transformations"
RESULTS_DIR = PROJECT_ROOT / "results" / "step6_model_selection"

TRAIN_FILE = DATA_DIR / "train.csv"

(RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)


# ============================================================================
# ORDINAL ENCODING
# ============================================================================

ORDINAL_MAPPINGS = {
    "loan_grade": {"A": 7, "B": 6, "C": 5, "D": 4, "E": 3, "F": 2, "G": 1},
    "person_home_ownership": {"OWN": 4, "MORTGAGE": 3, "OTHER": 2, "RENT": 1},
    "loan_intent": {
        "VENTURE": 6,
        "EDUCATION": 5,
        "PERSONAL": 4,
        "HOMEIMPROVEMENT": 3,
        "MEDICAL": 2,
        "DEBTCONSOLIDATION": 1,
    },
    "cb_person_default_on_file": {"N": 1, "Y": 0},
}


def encode_categorical(df):
    """Применяет Ordinal Encoding к категориальным признакам."""
    df_encoded = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    return df_encoded


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================


def create_objective(X_train, y_train):
    """Создаёт objective функцию для Optuna."""

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        model = LGBMClassifier(**params)

        # 5-fold CV для ROC-AUC
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)

        return scores.mean()

    return objective


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================


def main():
    print("\n" + "=" * 70)
    print("STEP 6: HYPERPARAMETER TUNING — LightGBM + Optuna")
    print("=" * 70)

    # ========================================================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 70)

    train = pd.read_csv(TRAIN_FILE)
    print(f"\nЗагружено: {train.shape[0]:,} строк, {train.shape[1]} столбцов")

    TARGET = "loan_status"
    X = train.drop(columns=[TARGET])
    y = train[TARGET]

    # Ordinal Encoding
    X_encoded = encode_categorical(X)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {X_train.shape[0]:,} строк")
    print(f"Val:   {X_val.shape[0]:,} строк")

    # ========================================================================
    # 2. BASELINE (до тюнинга)
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. BASELINE LightGBM (до тюнинга)")
    print("-" * 70)

    baseline_model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_val)
    y_proba_baseline = baseline_model.predict_proba(X_val)[:, 1]

    baseline_metrics = calculate_all_metrics(y_val, y_pred_baseline, y_proba_baseline)
    print(f"\n   ROC-AUC (baseline): {baseline_metrics['ROC-AUC']:.4f}")

    # ========================================================================
    # 3. OPTUNA OPTIMIZATION
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. OPTUNA OPTIMIZATION")
    print("-" * 70)

    N_TRIALS = 100
    print(f"\n   Запускаем {N_TRIALS} trials...")
    print("   Целевая метрика: ROC-AUC (5-fold CV)")
    print("   Это займёт несколько минут...\n")

    # Создаём study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Оптимизация
    objective = create_objective(X_train, y_train)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n   ✓ Оптимизация завершена!")
    print(f"   Лучший ROC-AUC (CV): {study.best_value:.4f}")

    # ========================================================================
    # 4. ЛУЧШИЕ ПАРАМЕТРЫ
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. ЛУЧШИЕ ПАРАМЕТРЫ")
    print("-" * 70)

    best_params = study.best_params
    print("\n   Лучшие гиперпараметры:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"      {param}: {value:.6f}")
        else:
            print(f"      {param}: {value}")

    # Сохраняем параметры
    params_df = pd.DataFrame([best_params])
    params_df.to_csv(RESULTS_DIR / "tables" / "best_params_lightgbm.csv", index=False)

    # ========================================================================
    # 5. ФИНАЛЬНАЯ МОДЕЛЬ
    # ========================================================================
    print("\n" + "-" * 70)
    print("5. ФИНАЛЬНАЯ МОДЕЛЬ")
    print("-" * 70)

    # Добавляем фиксированные параметры
    final_params = best_params.copy()
    final_params["class_weight"] = "balanced"
    final_params["random_state"] = 42
    final_params["n_jobs"] = -1
    final_params["verbose"] = -1

    final_model = LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_val)
    y_pred_proba = final_model.predict_proba(X_val)[:, 1]

    final_metrics = calculate_all_metrics(y_val, y_pred, y_pred_proba)
    print_metrics(final_metrics, title="LightGBM (tuned)")

    # Сохраняем метрики
    save_metrics(final_metrics, RESULTS_DIR / "tables" / "metrics_lightgbm_tuned.csv")

    # ========================================================================
    # 6. ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    print("\n" + "-" * 70)
    print("6. ВИЗУАЛИЗАЦИЯ")
    print("-" * 70)

    plot_all_model_visualizations(
        y_true=y_val,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        metrics_dict=final_metrics,
        model_name="LightGBM_Tuned",
        save_dir=RESULTS_DIR / "figures",
    )

    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)

    improvement = final_metrics["ROC-AUC"] - baseline_metrics["ROC-AUC"]

    print("\n📊 СРАВНЕНИЕ:")
    print(f"   {'Метрика':<15} {'Baseline':>10} {'Tuned':>10} {'Δ':>10}")
    print(f"   {'-' * 45}")
    print(
        f"   {'ROC-AUC':<15} {baseline_metrics['ROC-AUC']:>10.4f} {final_metrics['ROC-AUC']:>10.4f} {improvement:>+10.4f}"
    )
    print(
        f"   {'Accuracy':<15} {baseline_metrics['Accuracy']:>10.4f} {final_metrics['Accuracy']:>10.4f} {final_metrics['Accuracy'] - baseline_metrics['Accuracy']:>+10.4f}"
    )
    print(
        f"   {'Recall':<15} {baseline_metrics['Recall']:>10.4f} {final_metrics['Recall']:>10.4f} {final_metrics['Recall'] - baseline_metrics['Recall']:>+10.4f}"
    )
    print(
        f"   {'F1-Score':<15} {baseline_metrics['F1-Score']:>10.4f} {final_metrics['F1-Score']:>10.4f} {final_metrics['F1-Score'] - baseline_metrics['F1-Score']:>+10.4f}"
    )

    if improvement > 0:
        print(f"\n🚀 ROC-AUC улучшился на {improvement:.4f}!")
    else:
        print("\n⚠️ ROC-AUC не улучшился (возможно, baseline уже оптимален)")

    print(f"\n📁 Результаты сохранены в: {RESULTS_DIR}")
    print("   • best_params_lightgbm.csv")
    print("   • metrics_lightgbm_tuned.csv")
    print("   • figures/")


if __name__ == "__main__":
    main()
