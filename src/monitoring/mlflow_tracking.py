"""
Логирование результатов проекта в MLflow.

Принцип: results/ — единственный источник правды (лежит в Git). Этот скрипт
ничего не вычисляет и не придумывает — он только читает готовые CSV и графики
из results/ и показывает их в MLflow (метрики, параметры, артефакты).

Один MLflow-запуск на каждый шаг проекта, теги: step=<номер>, source=results.
Финальная модель (шаг 7) логируется отдельным скриптом log_final_model.py,
который сам прогоняет кросс-валидацию.
"""

import warnings
from pathlib import Path

import mlflow
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS = PROJECT_ROOT / "results"

# Метрики, которые показываем в UI (без счётчиков TP/TN/FP/FN).
STANDARD_METRICS = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Specificity"]


def log_metric_csv(path, prefix=""):
    """Читает CSV с метриками и логирует их в MLflow.

    Поддерживаются три формата, которые встречаются в results/:
    - широкий (одна строка: колонки-метрики);
    - длинный (колонки Metric, Value);
    - сравнение (колонка Model + колонки-метрики, по строке на модель).
    """
    df = pd.read_csv(path)

    if "Metric" in df.columns and "Value" in df.columns:
        for _, row in df.iterrows():
            name = str(row["Metric"])
            if name in STANDARD_METRICS:
                mlflow.log_metric(f"{prefix}{name}", float(row["Value"]))

    elif len(df) == 1:
        for col in df.columns:
            if col in STANDARD_METRICS:
                mlflow.log_metric(f"{prefix}{col}", float(df.iloc[0][col]))

    else:
        for _, row in df.iterrows():
            model = str(row["Model"]).replace(" ", "_")
            for col in df.columns:
                if col in STANDARD_METRICS:
                    mlflow.log_metric(f"{prefix}{model}_{col}", float(row[col]))


def log_param_csv(path):
    """Читает CSV с одной строкой параметров и логирует их как параметры запуска."""
    df = pd.read_csv(path)
    row = df.iloc[0]
    for col in df.columns:
        mlflow.log_param(col, row[col])


def log_dir(path, base):
    """Логирует файлы каталога как артефакты, пропуская скрытые файлы (.DS_Store и т.п.)."""
    path = Path(path)
    if not path.is_dir():
        return
    for p in sorted(path.rglob("*")):
        if p.is_file() and not any(part.startswith(".") for part in p.parts):
            sub = str(p.relative_to(path).parent)
            artifact_path = base if sub == "." else f"{base}/{sub}"
            mlflow.log_artifact(str(p), artifact_path=artifact_path)


# Описание шагов проекта: какие CSV с метриками/параметрами читать и какие
# каталоги figures/tables прикладывать артефактами. Шаг 7 — отдельным скриптом.
STEPS = [
    {
        "step": "step1",
        "name": "baseline_logistic_regression",
        "metric_csvs": ["results/step1_first_look/tables/01_baseline_0_metrics.csv"],
        "artifacts": ["results/step1_first_look/figures", "results/step1_first_look/tables"],
    },
    {
        "step": "step2",
        "name": "eda",
        "metric_csvs": [],  # метрик модели нет, только описательные таблицы и графики
        "artifacts": [
            "results/step2_eda/categorical_features",
            "results/step2_eda/numeric_features",
        ],
    },
    {
        "step": "step3",
        "name": "feature_importance_and_baselines",
        "metric_csvs": [
            "results/step3_importance/baseline_v1/tables/metrics.csv",
            "results/step3_importance/baseline_v1_onehot/tables/metrics.csv",
        ],
        "artifacts": [
            "results/step3_importance/baseline_v1",
            "results/step3_importance/baseline_v1_onehot",
        ],
    },
    {
        "step": "step4",
        "name": "after_cleaning",
        "metric_csvs": ["results/step4_cleaning/baseline_after_cleaning/tables/metrics.csv"],
        "artifacts": [
            "results/step4_cleaning/figures",
            "results/step4_cleaning/tables",
            "results/step4_cleaning/baseline_after_cleaning",
        ],
    },
    {
        "step": "step5",
        "name": "feature_engineering_boxcox",
        "metric_csvs": ["results/step5_feature_engineering/1_transformations/tables/metrics.csv"],
        "artifacts": ["results/step5_feature_engineering/1_transformations"],
    },
    {
        "step": "step6",
        "name": "model_selection_optuna",
        "metric_csvs": [
            "results/step6_model_selection/tables/metrics_lightgbm_tuned.csv",
            "results/step6_model_selection/tables/model_comparison.csv",
        ],
        "param_csv": "results/step6_model_selection/tables/best_params_lightgbm.csv",
        "artifacts": [
            "results/step6_model_selection/figures",
            "results/step6_model_selection/tables",
        ],
    },
]


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("loan-approval-prediction")

    for cfg in STEPS:
        with mlflow.start_run(
            run_name=cfg["name"], tags={"step": cfg["step"], "source": "results"}
        ):
            for csv_path in cfg.get("metric_csvs", []):
                log_metric_csv(RESULTS.parent / csv_path)
            if "param_csv" in cfg:
                log_param_csv(RESULTS.parent / cfg["param_csv"])
            for art_dir in cfg.get("artifacts", []):
                log_dir(RESULTS.parent / art_dir, base="artifacts")

    print("Готово: шаги 1-6 залогированы в mlflow.db.")
    print("Шаг 7 (финальная модель + Registry) логируется скриптом log_final_model.py.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
