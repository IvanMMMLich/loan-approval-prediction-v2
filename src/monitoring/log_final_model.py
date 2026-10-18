"""
Логирование финальной модели (шаг 7) в MLflow и регистрация в Model Registry.

Что делает:
1. Прогоняет кросс-валидацию существующей функцией run_cross_validation()
   из pipeline_cv.py (те же признаки и параметры).
2. Сравнивает ROC-AUC по фолдам и среднее с эталоном в
   results/step7_final_pipeline/tables/cv_results.csv. При расхождении
   останавливается и сообщает — содержимое results/ не перезаписывается.
3. Обучает модель на всех данных и логирует в MLflow: параметры, метрики CV
   (среднее, std, по фолдам), важность признаков, саму модель (sklearn Pipeline).
4. Регистрирует модель в Registry и ставит алиас champion.
"""

import contextlib
import sys
import warnings
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.step7_pipeline.pipeline_cv import (
    BEST_PARAMS,
    CATEGORICAL_FEATURES,
    create_pipeline,
    run_cross_validation,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_NAME = "loan-approval-lgbm"
TARGET = "loan_status"


def main():
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment("loan-approval-prediction")

    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train_step4_2.csv")
    X = train.drop(columns=[TARGET])
    y = train[TARGET]

    # 1. Реальная кросс-валидация (считаем сами, не читаем из файла).
    cv = run_cross_validation(X, y, n_splits=5)

    # 2. Сверка с эталоном из results/step7_final_pipeline/tables/cv_results.csv.
    ref = pd.read_csv(
        PROJECT_ROOT / "results" / "step7_final_pipeline" / "tables" / "cv_results.csv"
    )
    for _, row in ref.iterrows():
        metric = row["Metric"]
        mean_ref, std_ref = float(row["Mean"]), float(row["Std"])
        mean_calc, std_calc = float(cv[metric]["mean"]), float(cv[metric]["std"])
        if not (
            np.isclose(mean_calc, mean_ref, atol=1e-9) and np.isclose(std_calc, std_ref, atol=1e-9)
        ):
            print(
                f"РАСХОЖДЕНИЕ по метрике {metric}: "
                f"вычислено {mean_calc} ± {std_calc}, в файле {mean_ref} ± {std_ref}"
            )
            sys.exit(1)

    # 3. Обучение на всех данных и логирование.
    pipeline = create_pipeline()
    pipeline.fit(X, y)

    with mlflow.start_run(
        run_name="step7_final_model", tags={"step": "step7", "source": "results"}
    ):
        mlflow.log_params(BEST_PARAMS)

        for metric in ("roc_auc", "accuracy", "precision", "recall", "f1"):
            mlflow.log_metric(f"{metric}_mean", float(cv[metric]["mean"]))
            mlflow.log_metric(f"{metric}_std", float(cv[metric]["std"]))
        for i, score in enumerate(cv["roc_auc"]["scores"], 1):
            mlflow.log_metric(f"roc_auc_fold_{i}", float(score))

        feature_names = [
            "person_income_boxcox",
            "person_emp_length",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
        ] + CATEGORICAL_FEATURES
        for name, value in zip(
            feature_names, pipeline.named_steps["classifier"].feature_importances_, strict=False
        ):
            mlflow.log_metric(f"importance_{name}", float(value))

        mlflow.sklearn.log_model(pipeline, "model")
        run_id = mlflow.active_run().info.run_id

    # 4. Регистрация модели и алиас champion.
    client = MlflowClient()
    with contextlib.suppress(mlflow.exceptions.MlflowException):
        # Модель уже может быть зарегистрирована — тогда используем существующую запись.
        client.create_registered_model(MODEL_NAME)
    version = client.create_model_version(MODEL_NAME, source=f"runs:/{run_id}/model", run_id=run_id)
    client.set_registered_model_alias(MODEL_NAME, "champion", version=version.version)

    print(f"Модель зарегистрирована: {MODEL_NAME}, версия {version.version}, алиас champion.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
