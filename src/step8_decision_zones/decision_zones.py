"""
Шаг 8: расчёт границ зон решения по out-of-fold вероятностям.

Берёт существующий пайплайн (create_pipeline из step7) и те же данные
(data/processed/train_step4_2.csv). Считает out-of-fold вероятности класса 1
(cross_val_predict с тем же разбиением: StratifiedKFold, 5 фолдов, shuffle,
random_state=42) — каждая вероятность получена моделью, не видевшей эту заявку.

Смысл классов из README: 1 = одобрение, 0 = отказ. Границы подбираются на сетке
порогов с шагом 0.01 по двум независимым критериям:

- A (автоодобрение): верхняя граница — минимальный порог, при котором доля
  класса 0 среди автоматически одобренных <= A;
- B (автоотказ): нижняя граница — максимальный порог, при котором доля
  потерянных хороших заявок (класс 1, попавший ниже порога) от всех заявок
  класса 1 <= B. То есть «теряем не больше B процентов хороших заявок».

Сетка вариантов: A в {5%, 10%, 15%} x B в {2%, 5%, 10%} — 9 вариантов.
Если нижняя граница не меньше верхней (зоны перекрываются), средняя зона
считается нулевой, в таблице это помечается.
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
APPROVE_LEVELS = [0.05, 0.10, 0.15]  # A: допустимая доля плохих среди автоодобренных
LOST_LEVELS = [0.02, 0.05, 0.10]  # B: допустимая доля потерянных хороших от всех хороших
GRID_STEP = 0.01


def approve_error_rate(oof_proba, y, t):
    """Доля класса 0 среди заявок с proba >= t (ошибка автоодобрения)."""
    above = oof_proba >= t
    n = int(above.sum())
    if n == 0:
        return np.inf
    return float((y[above] == 0).sum() / n)


def lost_good_rate(oof_proba, y, t):
    """Доля класса 1, попавшего ниже порога t, от всех заявок класса 1."""
    n_good = int((y == 1).sum())
    if n_good == 0:
        return np.inf
    return float((y[oof_proba < t] == 1).sum() / n_good)


def find_bounds(oof_proba, y, thresholds, approve_level, lost_level):
    """Подбирает верхнюю (A) и нижнюю (B) границы на заданной сетке порогов."""
    valid_upper = [t for t in thresholds if approve_error_rate(oof_proba, y, t) <= approve_level]
    valid_lower = [t for t in thresholds if lost_good_rate(oof_proba, y, t) <= lost_level]
    upper = min(valid_upper) if valid_upper else None
    lower = max(valid_lower) if valid_lower else None
    return lower, upper


def zone_stats(oof_proba, y, lower, upper):
    """Считает число заявок в трёх зонах и метрики ошибок."""
    n = len(y)
    auto_reject = oof_proba < lower if lower is not None else np.zeros(n, dtype=bool)
    auto_approve = oof_proba >= upper if upper is not None else np.zeros(n, dtype=bool)
    manual = ~auto_reject & ~auto_approve

    approved_bad_count = int((y[auto_approve] == 0).sum())
    lost_good_count = int((y[auto_reject] == 1).sum())
    n_good = int((y == 1).sum())

    return {
        "auto_approve_count": int(auto_approve.sum()),
        "auto_approve_share": float(auto_approve.mean()),
        "manual_count": int(manual.sum()),
        "manual_share": float(manual.mean()),
        "auto_reject_count": int(auto_reject.sum()),
        "auto_reject_share": float(auto_reject.mean()),
        "approve_error_rate": float((y[auto_approve] == 0).mean()) if auto_approve.sum() else 0.0,
        "lost_good_rate": float(lost_good_count / n_good) if n_good else 0.0,
        "lost_good_count": lost_good_count,
        "approved_bad_count": approved_bad_count,
        "auto_decided_share": float((auto_reject | auto_approve).mean()),
    }


def build_threshold_grid(oof_proba, y, thresholds):
    rows = []
    for t in thresholds:
        above = oof_proba >= t
        rows.append(
            {
                "threshold": round(float(t), 2),
                "share_above": float(above.mean()),
                "share_below": float((~above).mean()),
                "error_rate_approve": approve_error_rate(oof_proba, y, t),
                "lost_good_rate": lost_good_rate(oof_proba, y, t),
            }
        )
    return pd.DataFrame(rows)


def variant_label(approve_level, lost_level):
    return f"A={approve_level:.0%}, B={lost_level:.0%}"


def plot_probability_distribution(oof_proba, y, variants, bounds):
    n_rows = len(APPROVE_LEVELS)
    n_cols = len(LOST_LEVELS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True
    )
    for ax, (approve_level, lost_level) in zip(axes.flat, variants, strict=False):
        lower, upper = bounds[(approve_level, lost_level)]
        ax.hist(oof_proba[y == 0], bins=50, alpha=0.6, color="#3498db")
        ax.hist(oof_proba[y == 1], bins=50, alpha=0.6, color="#e74c3c")
        if lower is not None:
            ax.axvline(lower, color="green", linestyle="--")
        if upper is not None:
            ax.axvline(upper, color="purple", linestyle="--")
        ax.set_title(variant_label(approve_level, lost_level), fontsize=9)
    axes[0, 0].legend(
        ["класс 0 (отказ)", "класс 1 (одобрение)", "нижняя", "верхняя"],
        fontsize=8,
        loc="upper left",
    )
    fig.suptitle("Распределение out-of-fold вероятностей и границы зон", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "probability_distribution_by_error_level.png", dpi=150)
    plt.close(fig)


def plot_zone_shares(variants, stats):
    labels = [variant_label(a, b) for a, b in variants]
    reject = [stats[(a, b)]["auto_reject_share"] for a, b in variants]
    manual = [stats[(a, b)]["manual_share"] for a, b in variants]
    approve = [stats[(a, b)]["auto_approve_share"] for a, b in variants]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, reject, width, label="Автоотказ", color="#e74c3c")
    ax.bar(x, manual, width, label="Ручная проверка", color="#f1c40f")
    ax.bar(x + width, approve, width, label="Автоодобрение", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Вариант (A — доля плохих среди автоодобренных, B — доля потерянных хороших)")
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

    variants = [(a, b) for a in APPROVE_LEVELS for b in LOST_LEVELS]
    bounds = {}
    stats = {}
    summary_rows = []
    for approve_level, lost_level in variants:
        lower, upper = find_bounds(oof_proba, y, thresholds, approve_level, lost_level)
        bounds[(approve_level, lost_level)] = (lower, upper)
        row = zone_stats(oof_proba, y, lower, upper)
        stats[(approve_level, lost_level)] = row
        overlap = lower is None or upper is None or lower >= upper
        if overlap:
            row["manual_count"] = 0
            row["manual_share"] = 0.0
        summary_rows.append(
            {
                "approve_error_level": approve_level,
                "lost_good_level": lost_level,
                "lower_bound": lower,
                "upper_bound": upper,
                "zones_overlap": bool(overlap),
                **row,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES_DIR / "zones_by_error_level.csv", index=False)

    plot_probability_distribution(oof_proba, y, variants, bounds)
    plot_zone_shares(variants, stats)

    print(summary.round(4).to_string(index=False))
    print(f"\nРезультаты сохранены в {RESULTS_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
