"""
Функции для анализа признаков:
- Pearson correlation matrix + heatmap
- Spearman correlation matrix + heatmap
- Permutation Importance bar chart

Использование:
    from utils.analysis import run_full_analysis

    run_full_analysis(
        X_train, y_train, X_val, y_val,
        model=trained_model,
        new_features=['feature1', 'feature2'],  # для выделения цветом
        save_dir='results/step5/.../figures'
    )
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

# ============================================================================
# ORDINAL ENCODING ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ (Subprime Logic)
# ============================================================================

# Логика: выше число = лучше клиент = РЕЖЕ одобряют
ORDINAL_MAPPINGS = {
    "loan_grade": {
        "A": 7,  # Лучший грейд: 4.92% approval
        "B": 6,
        "C": 5,
        "D": 4,
        "E": 3,
        "F": 2,
        "G": 1,  # Худший грейд: 81.82% approval
    },
    "person_home_ownership": {
        "OWN": 4,  # 1.37% approval
        "MORTGAGE": 3,  # 5.97% approval
        "OTHER": 2,  # 16.85% approval
        "RENT": 1,  # 22.26% approval
    },
    "loan_intent": {
        "VENTURE": 6,  # 9.28% approval
        "EDUCATION": 5,  # 10.48% approval
        "PERSONAL": 4,  # 12.35% approval
        "HOMEIMPROVEMENT": 3,  # 13.55% approval
        "MEDICAL": 2,  # 16.67% approval
        "DEBTCONSOLIDATION": 1,  # 18.93% approval
    },
    "cb_person_default_on_file": {
        "N": 1,  # Не было дефолта (лучше)
        "Y": 0,  # Был дефолт (хуже)
    },
}

# Перевод названий признаков на русский
FEATURE_NAMES_RU = {
    "person_income": "Доход",
    "person_emp_length": "Стаж работы",
    "loan_amnt": "Сумма кредита",
    "loan_int_rate": "Процентная ставка",
    "loan_percent_income": "% дохода на кредит",
    "person_home_ownership": "Владение жильём",
    "loan_intent": "Цель кредита",
    "loan_grade": "Грейд кредита",
    "cb_person_default_on_file": "Наличие дефолта",
    "loan_status": "Статус кредита",
}


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================


def apply_ordinal_encoding(df):
    """
    Применяет Ordinal Encoding к категориальным признакам.
    Возвращает DataFrame с числовыми значениями для корреляций.
    """
    df_encoded = df.copy()

    for col, mapping in ORDINAL_MAPPINGS.items():
        # Проверяем, что колонка ещё не закодирована (содержит строки)
        if col in df_encoded.columns and df_encoded[col].dtype == "object":
            df_encoded[col] = df_encoded[col].map(mapping)

    return df_encoded


def get_feature_name_ru(feature):
    """Возвращает русское название признака или оригинальное."""
    return FEATURE_NAMES_RU.get(feature, feature)


def rename_columns_to_russian(df):
    """Переименовывает колонки на русский язык."""
    new_columns = [get_feature_name_ru(col) for col in df.columns]
    df_renamed = df.copy()
    df_renamed.columns = new_columns
    return df_renamed


# ============================================================================
# PEARSON CORRELATION
# ============================================================================


def compute_pearson_correlation(df, save_dir=None):
    """
    Вычисляет матрицу корреляций Пирсона.

    Parameters
    ----------
    df : DataFrame
        Данные (уже с ordinal encoding для категориальных)
    save_dir : str or Path, optional
        Путь для сохранения результатов

    Returns
    -------
    corr_matrix : DataFrame
        Матрица корреляций
    """
    # Применяем ordinal encoding если есть категориальные
    df_encoded = apply_ordinal_encoding(df)

    # Переименовываем на русский
    df_ru = rename_columns_to_russian(df_encoded)

    # Вычисляем корреляцию
    corr_matrix = df_ru.corr(method="pearson")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем CSV
        csv_path = save_dir / "pearson_correlation_matrix.csv"
        corr_matrix.to_csv(csv_path, encoding="utf-8-sig")
        print(f"   Pearson матрица: {csv_path}")

    return corr_matrix


def plot_pearson_heatmap(corr_matrix, save_path=None):
    """
    Создаёт heatmap корреляционной матрицы Пирсона.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Heatmap
    im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    # Настройка осей
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=10, weight="bold")
    ax.set_yticklabels(corr_matrix.index, fontsize=10, weight="bold")

    # Добавляем значения в ячейки
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                weight="bold",
            )

    # Сетка
    ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("Корреляция Пирсона", rotation=270, labelpad=20, fontsize=12, weight="bold")

    # Заголовок
    ax.set_title("Матрица корреляций Пирсона", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"   Pearson heatmap: {save_path}")

    plt.close()


# ============================================================================
# SPEARMAN CORRELATION
# ============================================================================


def compute_spearman_correlation(df, save_dir=None):
    """
    Вычисляет матрицу ранговых корреляций Спирмена.

    Parameters
    ----------
    df : DataFrame
        Данные (уже с ordinal encoding для категориальных)
    save_dir : str or Path, optional
        Путь для сохранения результатов

    Returns
    -------
    corr_matrix : DataFrame
        Матрица корреляций
    """
    # Применяем ordinal encoding если есть категориальные
    df_encoded = apply_ordinal_encoding(df)

    # Переименовываем на русский
    df_ru = rename_columns_to_russian(df_encoded)

    # Вычисляем корреляцию Спирмена
    corr_matrix = df_ru.corr(method="spearman")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Сохраняем CSV
        csv_path = save_dir / "spearman_correlation_matrix.csv"
        corr_matrix.to_csv(csv_path, encoding="utf-8-sig")
        print(f"   Spearman матрица: {csv_path}")

    return corr_matrix


def plot_spearman_heatmap(corr_matrix, save_path=None):
    """
    Создаёт heatmap корреляционной матрицы Спирмена.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Heatmap
    im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)

    # Настройка осей
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.index)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=10, weight="bold")
    ax.set_yticklabels(corr_matrix.index, fontsize=10, weight="bold")

    # Добавляем значения в ячейки
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                weight="bold",
            )

    # Сетка
    ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix.index)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", size=0)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label("Корреляция Спирмена", rotation=270, labelpad=20, fontsize=12, weight="bold")

    # Заголовок
    ax.set_title("Матрица корреляций Спирмена", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"   Spearman heatmap: {save_path}")

    plt.close()


# ============================================================================
# PERMUTATION IMPORTANCE
# ============================================================================


def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """
    Вычисляет Permutation Importance.

    Parameters
    ----------
    model : fitted model
        Обученная модель
    X : DataFrame
        Признаки (validation set)
    y : Series
        Целевая переменная
    n_repeats : int
        Количество повторений перемешивания
    random_state : int
        Seed для воспроизводимости

    Returns
    -------
    importance_df : DataFrame
        Таблица с важностью признаков
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1, scoring="roc_auc"
    )

    importance_df = (
        pd.DataFrame(
            {
                "Признак": X.columns,
                "Importance_mean": result.importances_mean,
                "Importance_std": result.importances_std,
            }
        )
        .sort_values("Importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    return importance_df


def plot_permutation_importance(importance_df, new_features=None, top_n=15, save_path=None):
    """
    Создаёт bar chart для Permutation Importance с выделением новых признаков.

    Parameters
    ----------
    importance_df : DataFrame
        Таблица с колонками ['Признак', 'Importance_mean', 'Importance_std']
    new_features : list, optional
        Список новых признаков для выделения красным цветом
    top_n : int
        Сколько топ признаков показать
    save_path : str or Path, optional
        Путь для сохранения графика
    """
    if new_features is None:
        new_features = []

    fig, ax = plt.subplots(figsize=(12, 10))

    # Берём топ признаков
    df_plot = importance_df.head(top_n).copy()
    df_plot = df_plot.sort_values("Importance_mean", ascending=True)

    # Цвета: новые признаки — красный, старые — синий
    colors = ["#e74c3c" if f in new_features else "#3498db" for f in df_plot["Признак"]]

    # Bar chart
    bars = ax.barh(
        range(len(df_plot)),
        df_plot["Importance_mean"],
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Error bars
    if "Importance_std" in df_plot.columns:
        ax.errorbar(
            df_plot["Importance_mean"],
            range(len(df_plot)),
            xerr=df_plot["Importance_std"],
            fmt="none",
            color="black",
            capsize=3,
            alpha=0.5,
        )

    # Подписи по оси Y
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["Признак"], fontsize=11)

    # Заголовок
    if new_features:
        title = f"Permutation Importance (Топ-{top_n})\nСиний = исходные, Красный = новые признаки"
    else:
        title = f"Permutation Importance (Топ-{top_n})"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Importance (падение ROC-AUC при перемешивании)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="x")

    # Добавляем значения на столбцах
    max_val = df_plot["Importance_mean"].max()
    for i, (_, value) in enumerate(zip(bars, df_plot["Importance_mean"], strict=False)):
        ax.text(
            value + max_val * 0.02,
            i,
            f"{value:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    # Легенда
    if new_features:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#3498db", edgecolor="black", label="Исходные признаки"),
            Patch(facecolor="#e74c3c", edgecolor="black", label="Новые признаки"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"   Permutation Importance: {save_path}")

    plt.close()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ — ЗАПУСК ВСЕГО АНАЛИЗА
# ============================================================================


def run_full_analysis(
    X_train, y_train, X_val, y_val, model, new_features=None, save_dir=None, top_n=15
):
    """
    Запускает полный анализ:
    1. Pearson correlation + heatmap
    2. Spearman correlation + heatmap
    3. Permutation Importance + bar chart

    Parameters
    ----------
    X_train : DataFrame
        Признаки обучающей выборки
    y_train : Series
        Таргет обучающей выборки
    X_val : DataFrame
        Признаки валидационной выборки
    y_val : Series
        Таргет валидационной выборки
    model : fitted model
        Обученная модель
    new_features : list, optional
        Список новых признаков для выделения
    save_dir : str or Path, optional
        Директория для сохранения результатов
    top_n : int
        Сколько топ признаков показать в Permutation Importance

    Returns
    -------
    results : dict
        Словарь с результатами анализа
    """

    print("\n" + "=" * 60)
    print("АНАЛИЗ ПРИЗНАКОВ")
    print("=" * 60)

    if save_dir:
        save_dir = Path(save_dir)
        figures_dir = save_dir / "figures"
        tables_dir = save_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = None
        tables_dir = None

    results = {}

    # ========================================================================
    # 1. PEARSON CORRELATION
    # ========================================================================
    print("\n1. Pearson Correlation...")

    # Объединяем X и y для корреляции с таргетом
    df_for_corr = X_train.copy()
    df_for_corr["loan_status"] = y_train.values

    pearson_matrix = compute_pearson_correlation(df_for_corr, save_dir=tables_dir)
    results["pearson"] = pearson_matrix

    if figures_dir:
        plot_pearson_heatmap(pearson_matrix, save_path=figures_dir / "pearson_heatmap.png")

    # ========================================================================
    # 2. SPEARMAN CORRELATION
    # ========================================================================
    print("\n2. Spearman Correlation...")

    spearman_matrix = compute_spearman_correlation(df_for_corr, save_dir=tables_dir)
    results["spearman"] = spearman_matrix

    if figures_dir:
        plot_spearman_heatmap(spearman_matrix, save_path=figures_dir / "spearman_heatmap.png")

    # ========================================================================
    # 3. PERMUTATION IMPORTANCE
    # ========================================================================
    print("\n3. Permutation Importance...")

    importance_df = compute_permutation_importance(model, X_val, y_val)
    results["permutation_importance"] = importance_df

    if tables_dir:
        csv_path = tables_dir / "permutation_importance.csv"
        importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"   Permutation Importance CSV: {csv_path}")

    if figures_dir:
        plot_permutation_importance(
            importance_df,
            new_features=new_features,
            top_n=top_n,
            save_path=figures_dir / "permutation_importance.png",
        )

    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "=" * 60)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("=" * 60)

    if save_dir:
        print(f"\nРезультаты сохранены в: {save_dir}")
        print("   figures/: heatmaps, bar charts")
        print("   tables/:  CSV с матрицами и importance")

    # Выводим топ-5 по важности
    print("\nТоп-5 признаков по Permutation Importance:")
    for _, row in importance_df.head(5).iterrows():
        marker = "🔴" if row["Признак"] in (new_features or []) else "🔵"
        print(f"   {marker} {row['Признак']:30} | {row['Importance_mean']:.4f}")

    return results


# ============================================================================
# ОТДЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВЫЗОВА ПО ЧАСТЯМ
# ============================================================================


def run_correlation_analysis(X, y, save_dir=None):
    """
    Запускает только корреляционный анализ (Pearson + Spearman).
    Без модели — только данные.
    """
    print("\n" + "=" * 60)
    print("КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
    print("=" * 60)

    if save_dir:
        save_dir = Path(save_dir)
        figures_dir = save_dir / "figures"
        tables_dir = save_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = None
        tables_dir = None

    # Объединяем X и y
    df_for_corr = X.copy()
    df_for_corr["loan_status"] = y.values

    # Pearson
    print("\n1. Pearson Correlation...")
    pearson_matrix = compute_pearson_correlation(df_for_corr, save_dir=tables_dir)
    if figures_dir:
        plot_pearson_heatmap(pearson_matrix, save_path=figures_dir / "pearson_heatmap.png")

    # Spearman
    print("\n2. Spearman Correlation...")
    spearman_matrix = compute_spearman_correlation(df_for_corr, save_dir=tables_dir)
    if figures_dir:
        plot_spearman_heatmap(spearman_matrix, save_path=figures_dir / "spearman_heatmap.png")

    print("\nКорреляционный анализ завершён!")

    return {"pearson": pearson_matrix, "spearman": spearman_matrix}


def run_importance_analysis(model, X_val, y_val, new_features=None, save_dir=None, top_n=15):
    """
    Запускает только Permutation Importance анализ.
    Требует обученную модель.
    """
    print("\n" + "=" * 60)
    print("PERMUTATION IMPORTANCE")
    print("=" * 60)

    if save_dir:
        save_dir = Path(save_dir)
        figures_dir = save_dir / "figures"
        tables_dir = save_dir / "tables"
        figures_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)
    else:
        figures_dir = None
        tables_dir = None

    importance_df = compute_permutation_importance(model, X_val, y_val)

    if tables_dir:
        csv_path = tables_dir / "permutation_importance.csv"
        importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"   CSV: {csv_path}")

    if figures_dir:
        plot_permutation_importance(
            importance_df,
            new_features=new_features,
            top_n=top_n,
            save_path=figures_dir / "permutation_importance.png",
        )

    print("\nPermutation Importance анализ завершён!")

    return importance_df
