"""
Шаг 8: расчёт границ зон решения по out-of-fold вероятностям.

Берёт существующий пайплайн (create_pipeline из step7) и те же данные
(data/processed/train_step4_2.csv). Считает out-of-fold вероятности класса 1
(cross_val_predict с тем же разбиением: StratifiedKFold, 5 фолдов, shuffle,
random_state=42) — каждая вероятность получена моделью, не видевшей эту заявку.

Смысл классов из README: 1 = одобрение, 0 = отказ. Для уровней ошибки 5, 10 и 15
процентов подбирает границы на сетке порогов с шагом 0.01:

- верхняя граница — минимальный порог, при котором доля фактических отказов
  (класс 0) среди автоматически одобренных не больше уровня;
- нижняя граница — максимальный порог, при котором доля фактических одобрений
  (класс 1) среди автоматически отклонённых не больше уровня.

Если нижняя граница оказалась не меньше верхней (зоны перекрываются), средняя
зона считается нулевой, в таблице это помечается.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.step7_pipeline.pipeline_cv import create_pipeline

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "train_step4_2.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "step8_decision_zones"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TARGET = "loan_status"
REF_ROC_AUC = 0.9587
MAX_ROC_AUC_DIFF = 0.005
ERROR_LEVELS = [0.05, 0.10, 0.15]
GRID_STEP = 0.01


def approve_error_rate(oof_proba, y, t):
    """Доля класса 0 среди заявок с proba >= t (ошибка автоодобрения)."""
    above = oof_proba >= t
    n = int(above.sum())
    if n == 0:
        return np.inf
    return float((y[above] == 0).sum() / n)


def reject_error_rate(oof_proba, y, t):
    """Доля класса 1 среди заявок с proba < t (ошибка автоотказа)."""
    below = oof_proba < t
    n = int(below.sum())
    if n == 0:
        return np.inf
    return float((y[below] == 1).sum() / n)


def find_bounds(oof_proba, y, thresholds, level):
    """Подбирает верхнюю и нижнюю границы на заданной сетке порогов."""
    valid_upper = [t for t in thresholds if approve_error_rate(oof_proba, y, t) <= level]
    valid_lower = [t for t in thresholds if reject_error_rate(oof_proba, y, t) <= level]
    upper = min(valid_upper) if valid_upper else None
    lower = max(valid_lower) if valid_lower else None
    return lower, upper


def zone_sizes(oof_proba, y, lower, upper):
    """Считает число заявок в трёх зонах и фактические доли ошибок в авто-зонах."""
    n = len(y)
    auto_reject = oof_proba < lower if lower is not None else np.zeros(n, dtype=bool)
    auto_approve = oof_proba >= upper if upper is not None else np.zeros(n, dtype=bool)
    manual = ~auto_reject & ~auto_approve

    approve_err = (y[auto_approve] == 0).mean() if auto_approve.sum() else 0.0
    reject_err = (y[auto_reject] == 1).mean() if auto_reject.sum() else 0.0

    return {
        "auto_approve_count": int(auto_approve.sum()),
        "auto_approve_share": float(auto_approve.mean()),
        "manual_count": int(manual.sum()),
        "manual_share": float(manual.mean()),
        "auto_reject_count": int(auto_reject.sum()),
        "auto_reject_share": float(auto_reject.mean()),
        "approve_error_rate": float(approve_err),
        "reject_error_rate": float(reject_err),
        "auto_decided_share": float((auto_reject | auto_approve).mean()),
    }


def build_threshold_grid(oof_proba, y, thresholds):
    rows = []
    for t in thresholds:
        above = oof_proba >= t
        below = ~above
        rows.append(
            {
                "threshold": round(float(t), 2),
                "share_above": float(above.mean()),
                "share_below": float(below.mean()),
                "error_rate_approve": approve_error_rate(oof_proba, y, t),
                "error_rate_reject": reject_error_rate(oof_proba, y, t),
            }
        )
    return pd.DataFrame(rows)


def plot_probability_distribution(oof_proba, y, levels, bounds):
    fig, axes = plt.subplots(1, len(levels), figsize=(6 * len(levels), 5), sharey=True)
    for ax, level in zip(axes, levels, strict=False):
        lower, upper = bounds[level]
        ax.hist(oof_proba[y == 0], bins=50, alpha=0.6, label="класс 0 (отказ)", color="#3498db")
        ax.hist(oof_proba[y == 1], bins=50, alpha=0.6, label="класс 1 (одобрение)", color="#e74c3c")
        if lower is not None:
            ax.axvline(lower, color="green", linestyle="--", label=f"нижняя {lower:.2f}")
        if upper is not None:
            ax.axvline(upper, color="purple", linestyle="--", label=f"верхняя {upper:.2f}")
        ax.set_title(f"Уровень ошибки {level:.0%}")
        ax.set_xlabel("Вероятность класса 1 (out-of-fold)")
        ax.set_ylabel("Число заявок")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "probability_distribution_by_error_level.png", dpi=150)
    plt.close(fig)


def plot_zone_shares(summary):
    labels = [f"{row['error_level']:.0%}" for row in summary]
    reject = [row["auto_reject_share"] for row in summary]
    manual = [row["manual_share"] for row in summary]
    approve = [row["auto_approve_share"] for row in summary]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, reject, width, label="Автоотказ", color="#e74c3c")
    ax.bar(x, manual, width, label="Ручная проверка", color="#f1c40f")
    ax.bar(x + width, approve, width, label="Автоодобрение", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Допустимая доля ошибок")
    ax.set_ylabel("Доля заявок")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "zone_shares_by_error_level.png", dpi=150)
    plt.close(fig)


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(DATA_FILE)
    X = train.drop(columns=[TARGET])
    y = train[TARGET].to_numpy()

    pipeline = create_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]

    oof_roc_auc = float(roc_auc_score(y, oof_proba))
    print(f"ROC-AUC (out-of-fold): {oof_roc_auc:.6f}")
    if abs(oof_roc_auc - REF_ROC_AUC) > MAX_ROC_AUC_DIFF:
        print(
            f"РАСХОЖДЕНИЕ: out-of-fold {oof_roc_auc:.6f} отличается от эталона "
            f"{REF_ROC_AUC:.6f} более чем на {MAX_ROC_AUC_DIFF}."
        )
        sys.exit(1)

    thresholds = np.round(np.arange(0.0, 1.0 + GRID_STEP, GRID_STEP), 2)

    grid = build_threshold_grid(oof_proba, y, thresholds)
    grid.to_csv(TABLES_DIR / "threshold_grid.csv", index=False)

    bounds = {}
    summary_rows = []
    for level in ERROR_LEVELS:
        lower, upper = find_bounds(oof_proba, y, thresholds, level)
        bounds[level] = (lower, upper)
        sizes = zone_sizes(oof_proba, y, lower, upper)
        overlap = lower is None or upper is None or lower >= upper
        if overlap:
            sizes["manual_count"] = 0
            sizes["manual_share"] = 0.0
        summary_rows.append(
            {
                "error_level": level,
                "lower_bound": lower,
                "upper_bound": upper,
                "zones_overlap": bool(overlap),
                **sizes,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "zones_by_error_level.csv", index=False)

    plot_probability_distribution(oof_proba, y, ERROR_LEVELS, bounds)
    plot_zone_shares(summary_rows)

    print(summary.round(4).to_string(index=False))
    print(f"\nРезультаты сохранены в {RESULTS_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
