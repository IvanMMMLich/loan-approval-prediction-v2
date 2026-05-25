"""
Мониторинг ML-пайплайна через MLflow.
Логирует метрики модели и инфраструктурные показатели.
"""

import mlflow
import time
import psutil
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================================================
# МЕТРИКИ ПРОГРЕССА ПО ЭТАПАМ (из README)
# ============================================================================

PROGRESS_METRICS = [
    {"step": "Step1_LogisticRegression", "roc_auc": 0.87},
    {"step": "Step3_RF_OneHot",          "roc_auc": 0.92},
    {"step": "Step3_RF_Ordinal",         "roc_auc": 0.93},
    {"step": "Step4_after_cleaning",     "roc_auc": 0.935},
    {"step": "Step5_BoxCox",             "roc_auc": 0.937},
    {"step": "Step6_LightGBM_baseline",  "roc_auc": 0.9552},
    {"step": "Step6_LightGBM_Optuna",    "roc_auc": 0.9602},
    {"step": "Step7_Final_CV",           "roc_auc": 0.9587},
]

# Лучшие параметры из Optuna
BEST_PARAMS = {
    "n_estimators":      836,
    "max_depth":         12,
    "learning_rate":     0.033018,
    "num_leaves":        23,
    "min_child_samples": 40,
    "subsample":         0.647898,
    "colsample_bytree":  0.574083,
    "reg_alpha":         4.743694,
    "reg_lambda":        0.408280,
}

# Финальные метрики модели
FINAL_METRICS = {
    "roc_auc":     0.9587,
    "accuracy":    0.9235,
    "precision":   0.6887,
    "recall":      0.8443,
    "f1_score":    0.7585,
    "specificity": 0.9408,
}


def log_infrastructure():
    """Логирует показатели инфраструктуры."""
    process = psutil.Process(os.getpid())

    ram_used_mb = process.memory_info().rss / 1024 / 1024
    ram_total_mb = psutil.virtual_memory().total / 1024 / 1024
    ram_percent = psutil.virtual_memory().percent
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()

    mlflow.log_metric("ram_used_mb", round(ram_used_mb, 2))
    mlflow.log_metric("ram_total_mb", round(ram_total_mb, 2))
    mlflow.log_metric("ram_used_percent", round(ram_percent, 2))
    mlflow.log_metric("cpu_used_percent", round(cpu_percent, 2))
    mlflow.log_metric("cpu_count", cpu_count)

    print(f"   RAM использовано: {ram_used_mb:.1f} MB ({ram_percent:.1f}%)")
    print(f"   RAM всего:        {ram_total_mb:.1f} MB")
    print(f"   CPU загрузка:     {cpu_percent:.1f}% ({cpu_count} ядер)")


def main():
    print("=" * 60)
    print("МОНИТОРИНГ ML-ПАЙПЛАЙНА (MLflow)")
    print("=" * 60)

    mlflow.set_experiment("loan-approval-prediction")

    # ========================================================================
    # 1. ПРОГРЕСС МЕТРИК ПО ЭТАПАМ
    # ========================================================================
    print("\n1. Логируем прогресс ROC-AUC по этапам...")

    with mlflow.start_run(run_name="model_quality_monitoring"):
        for i, entry in enumerate(PROGRESS_METRICS):
            mlflow.log_metric("roc_auc_progress", entry["roc_auc"], step=i)
            print(f"   {entry['step']}: ROC-AUC = {entry['roc_auc']}")

    # ========================================================================
    # 2. ФИНАЛЬНАЯ МОДЕЛЬ — ПАРАМЕТРЫ И МЕТРИКИ
    # ========================================================================
    print("\n2. Логируем финальную модель...")

    with mlflow.start_run(run_name="final_model_lightgbm"):
        mlflow.log_params(BEST_PARAMS)
        mlflow.log_metrics(FINAL_METRICS)
        print(f"   Параметры Optuna залогированы")
        print(f"   ROC-AUC (CV): {FINAL_METRICS['roc_auc']}")

    # ========================================================================
    # 3. МОНИТОРИНГ ИНФРАСТРУКТУРЫ
    # ========================================================================
    print("\n3. Мониторинг инфраструктуры...")

    with mlflow.start_run(run_name="infrastructure_monitoring"):
        start_time = time.time()
        log_infrastructure()
        elapsed = time.time() - start_time
        mlflow.log_metric("monitoring_time_sec", round(elapsed, 3))

    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print("\nЗапусти UI командой:")
    print("   mlflow ui")
    print("Открой браузер: http://127.0.0.1:5000")


if __name__ == "__main__":
    main()