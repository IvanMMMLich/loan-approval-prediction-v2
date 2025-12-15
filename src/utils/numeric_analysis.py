"""
Утилиты для анализа числовых признаков (Step 2.1 EDA)

Этот модуль содержит функции для комплексного анализа числовых признаков:
- Описательная статистика
- Асимметрия и эксцесс
- Визуализация распределений
- Поиск выбросов
- Сравнение распределений по классам

Автор: NeIvan
Проект: Loan Approval Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Union


# ============================================================================
# НАСТРОЙКИ ВИЗУАЛИЗАЦИИ
# ============================================================================

# Настройка стиля графиков
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Цветовая палитра для классов (loan_status)
CLASS_COLORS = {
    0: '#e74c3c',  # Красный для отклонённых (0)
    1: '#2ecc71'   # Зелёный для одобренных (1)
}

CLASS_LABELS = {
    0: 'Отклонено',
    1: 'Одобрено'
}


# ============================================================================
# ФУНКЦИЯ 1: ОПИСАТЕЛЬНАЯ СТАТИСТИКА
# ============================================================================

def calculate_descriptive_stats(
    df: pd.DataFrame,
    features: List[str],
    save_table_image_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Вычисляет описательную статистику для числовых признаков.
    
    Рассчитывает: среднее, медиану, стандартное отклонение, 
    минимум, максимум, квартили (Q1, Q3).
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    features : List[str]
        Список названий числовых признаков для анализа
    save_table_image_path : Optional[Union[str, Path]], optional
        Путь для сохранения PNG картинки таблицы. 
        Если None, картинка не создаётся.
    
    Returns
    -------
    pd.DataFrame
        Таблица со статистикой (строки = признаки, колонки = метрики)
        
    Examples
    --------
    >>> stats_df = calculate_descriptive_stats(
    ...     df, 
    ...     ['person_age', 'person_income'],
    ...     save_table_image_path='results/stats_table.png'
    ... )
    >>> stats_df.to_csv('results/stats.csv', index=False)
    """
    
    # Вычисляем статистики для каждого признака
    stats_list = []
    
    for feature in features:
        stats = {
            'Признак': feature,
            'Среднее': df[feature].mean(),
            'Медиана': df[feature].median(),
            'Станд. откл.': df[feature].std(),
            'Минимум': df[feature].min(),
            'Максимум': df[feature].max(),
            'Q1 (25%)': df[feature].quantile(0.25),
            'Q3 (75%)': df[feature].quantile(0.75)
        }
        stats_list.append(stats)
    
    # Создаём DataFrame
    stats_df = pd.DataFrame(stats_list)
    
    # Округляем до 2 знаков для читаемости
    numeric_cols = stats_df.columns[1:]  # Все кроме 'Признак'
    stats_df[numeric_cols] = stats_df[numeric_cols].round(2)
    
    # Если указан путь для картинки - создаём визуализацию таблицы
    if save_table_image_path is not None:
        _save_table_as_image(
            stats_df,
            title='Описательная статистика',
            save_path=save_table_image_path
        )
    
    return stats_df


# ============================================================================
# ФУНКЦИЯ 2: АСИММЕТРИЯ И ЭКСЦЕСС
# ============================================================================

def calculate_skewness_kurtosis(
    df: pd.DataFrame,
    features: List[str],
    save_table_image_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Вычисляет асимметрию (skewness) и эксцесс (kurtosis) для признаков.
    
    Skewness показывает асимметрию распределения:
    - < -0.5: левосторонняя асимметрия
    - от -0.5 до 0.5: симметричное
    - > 0.5: правосторонняя асимметрия
    
    Kurtosis показывает "тяжесть" хвостов:
    - < 0: плоское распределение
    - ~ 0: нормальное
    - > 0: острое, много выбросов
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    features : List[str]
        Список названий числовых признаков
    save_table_image_path : Optional[Union[str, Path]], optional
        Путь для сохранения PNG картинки таблицы
    
    Returns
    -------
    pd.DataFrame
        Таблица с колонками: Признак, Асимметрия, Эксцесс, Интерпретация
    """
    
    results = []
    
    for feature in features:
        skew = df[feature].skew()
        kurt = df[feature].kurt()
        
        # Интерпретация асимметрии
        if skew < -0.5:
            skew_interp = 'Левосторонняя'
        elif skew > 0.5:
            skew_interp = 'Правосторонняя'
        else:
            skew_interp = 'Симметричное'
        
        # Интерпретация эксцесса
        if kurt > 1:
            kurt_interp = 'Тяжёлые хвосты'
        elif kurt < -1:
            kurt_interp = 'Лёгкие хвосты'
        else:
            kurt_interp = 'Нормальное'
        
        results.append({
            'Признак': feature,
            'Асимметрия (Skewness)': round(skew, 3),
            'Эксцесс (Kurtosis)': round(kurt, 3),
            'Интерпретация': f'{skew_interp}, {kurt_interp}'
        })
    
    result_df = pd.DataFrame(results)
    
    # Сохраняем картинку таблицы если указан путь
    if save_table_image_path is not None:
        _save_table_as_image(
            result_df,
            title='Асимметрия и Эксцесс',
            save_path=save_table_image_path
        )
    
    return result_df


# ============================================================================
# ФУНКЦИЯ 3: ПОИСК ВЫБРОСОВ (IQR МЕТОД)
# ============================================================================

def detect_outliers_iqr(
    df: pd.DataFrame,
    features: List[str],
    save_table_image_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Находит выбросы по IQR (Interquartile Range) методу.
    
    IQR метод:
    1. Вычисляем Q1 (25-й процентиль) и Q3 (75-й процентиль)
    2. IQR = Q3 - Q1
    3. Нижняя граница = Q1 - 1.5 × IQR
    4. Верхняя граница = Q3 + 1.5 × IQR
    5. Выбросы = значения вне [нижняя граница, верхняя граница]
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    features : List[str]
        Список названий числовых признаков
    save_table_image_path : Optional[Union[str, Path]], optional
        Путь для сохранения PNG картинки таблицы
    
    Returns
    -------
    pd.DataFrame
        Таблица с колонками: Признак, Кол-во выбросов, Процент выбросов,
        Нижняя граница, Верхняя граница
    """
    
    results = []
    
    for feature in features:
        # Вычисляем квартили
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        
        # Границы для выбросов
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Находим выбросы
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        results.append({
            'Признак': feature,
            'Кол-во выбросов': outlier_count,
            'Процент выбросов (%)': round(outlier_percentage, 2),
            'Нижняя граница': round(lower_bound, 2),
            'Верхняя граница': round(upper_bound, 2)
        })
    
    result_df = pd.DataFrame(results)
    
    # Сохраняем картинку таблицы если указан путь
    if save_table_image_path is not None:
        _save_table_as_image(
            result_df,
            title='Выбросы (IQR метод)',
            save_path=save_table_image_path,
            figsize=(14, 4)  # Шире, т.к. 5 колонок
        )
    
    return result_df


# ============================================================================
# ФУНКЦИЯ 4: ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ (HISTOGRAM + KDE)
# ============================================================================

def plot_distribution_single(
    df: pd.DataFrame,
    feature: str,
    save_path: Union[str, Path]
) -> None:
    """
    Создаёт график распределения для одного признака (Histogram + KDE).
    
    График показывает:
    - Гистограмму (bars) с 50 интервалами
    - KDE кривую (сглаженная плотность)
    - Вертикальные линии для mean и median
    - Текстовую аннотацию со статистиками
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название числового признака
    save_path : Union[str, Path]
        Путь для сохранения PNG файла
    
    Returns
    -------
    None
        График сохраняется в файл
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram + KDE
    sns.histplot(
        data=df,
        x=feature,
        kde=True,
        bins=50,
        color='steelblue',
        alpha=0.6,
        edgecolor='black',
        linewidth=0.5,
        ax=ax
    )
    
    # Вертикальные линии для mean и median
    mean_val = df[feature].mean()
    median_val = df[feature].median()
    
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Среднее: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
               label=f'Медиана: {median_val:.2f}')
    
    # Заголовок и подписи
    ax.set_title(f'Распределение: {feature}', fontsize=16, fontweight='bold')
    ax.set_xlabel(feature, fontsize=14)
    ax.set_ylabel('Частота', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Добавляем текстовую аннотацию со статистиками
    skew = df[feature].skew()
    kurt = df[feature].kurt()
    stats_text = f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" График распределения сохранён: {save_path}")


# ============================================================================
# ФУНКЦИЯ 5: BOXPLOT ДЛЯ ОДНОГО ПРИЗНАКА
# ============================================================================

def plot_boxplot_single(
    df: pd.DataFrame,
    feature: str,
    save_path: Union[str, Path]
) -> None:
    """
    Создаёт boxplot для одного признака (поиск выбросов).
    
    Boxplot показывает:
    - Медиану (линия внутри коробки)
    - Q1 и Q3 (границы коробки)
    - Усы (нормальный диапазон)
    - Точки-выбросы за усами
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название числового признака
    save_path : Union[str, Path]
        Путь для сохранения PNG файла
    
    Returns
    -------
    None
        График сохраняется в файл
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Boxplot
    sns.boxplot(
        data=df,
        y=feature,
        color='lightblue',
        width=0.5,
        linewidth=2,
        fliersize=5,
        ax=ax
    )
    
    # Заголовок и подписи
    ax.set_title(f'Boxplot: {feature} (Поиск выбросов)', 
                 fontsize=16, fontweight='bold')
    ax.set_ylabel(feature, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем текст с количеством выбросов
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_count = len(df[(df[feature] < lower_bound) | (df[feature] > upper_bound)])
    outliers_pct = (outliers_count / len(df)) * 100
    
    stats_text = f'Выбросов: {outliers_count} ({outliers_pct:.2f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Boxplot сохранён: {save_path}")


# ============================================================================
# ФУНКЦИЯ 6: СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ПО КЛАССАМ (KDE)
# ============================================================================

def plot_class_comparison_single(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    save_path: Union[str, Path]
) -> None:
    """
    Создаёт KDE сравнение распределений для loan_status=0 vs loan_status=1.
    
    ЭТО КЛЮЧЕВОЙ ГРАФИК!
    Показывает разделяет ли признак классы (одобренные vs отклонённые).
    
    Если кривые не пересекаются → признак ОЧЕНЬ важен для модели.
    Если кривые совпадают → признак бесполезен.
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название числового признака
    target_col : str
        Название целевой переменной (обычно 'loan_status')
    save_path : Union[str, Path]
        Путь для сохранения PNG файла
    
    Returns
    -------
    None
        График сохраняется в файл
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # KDE для каждого класса
    for class_value in sorted(df[target_col].unique()):
        subset = df[df[target_col] == class_value]
        
        sns.kdeplot(
            data=subset,
            x=feature,
            fill=True,
            alpha=0.5,
            linewidth=2.5,
            color=CLASS_COLORS[class_value],
            label=CLASS_LABELS[class_value],
            ax=ax
        )
    
    # Заголовок и подписи
    ax.set_title(
        f'Сравнение распределений по классам: {feature}',
        fontsize=16,
        fontweight='bold'
    )
    ax.set_xlabel(feature, fontsize=14)
    ax.set_ylabel('Плотность вероятности', fontsize=14)
    ax.legend(title='Статус заявки', fontsize=12, title_fontsize=13)
    ax.grid(True, alpha=0.3)
    
    # Вычисляем средние для каждого класса и добавляем вертикальные линии
    for class_value in sorted(df[target_col].unique()):
        subset = df[df[target_col] == class_value]
        mean_val = subset[feature].mean()
        ax.axvline(
            mean_val,
            color=CLASS_COLORS[class_value],
            linestyle='--',
            linewidth=1.5,
            alpha=0.7
        )
    
    # Добавляем текст с разницей средних
    mean_0 = df[df[target_col] == 0][feature].mean()
    mean_1 = df[df[target_col] == 1][feature].mean()
    diff = abs(mean_1 - mean_0)
    diff_pct = (diff / mean_0) * 100 if mean_0 != 0 else 0
    
    stats_text = (f'Среднее (Отклонено): {mean_0:.2f}\n'
                  f'Среднее (Одобрено): {mean_1:.2f}\n'
                  f'Разница: {diff:.2f} ({diff_pct:.1f}%)')
    
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" График сравнения по классам сохранён: {save_path}")


# ============================================================================
# ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ: СОХРАНЕНИЕ ТАБЛИЦЫ КАК PNG
# ============================================================================

def _save_table_as_image(
    df: pd.DataFrame,
    title: str,
    save_path: Union[str, Path],
    figsize: tuple = (12, 6)
) -> None:
    """
    Сохраняет pandas DataFrame как красиво оформленную PNG картинку.
    
    Внутренняя функция, используется другими функциями модуля.
    
    Parameters
    ----------
    df : pd.DataFrame
        Таблица для сохранения
    title : str
        Заголовок таблицы
    save_path : Union[str, Path]
        Путь для сохранения PNG
    figsize : tuple, optional
        Размер фигуры (ширина, высота)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    
    # Создаём таблицу
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)  # Увеличиваем высоту строк
    
    # Заголовки колонок - жирным шрифтом и с фоном
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Чередующиеся цвета строк для читаемости
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')
            else:
                cell.set_facecolor('white')
    
    # Заголовок
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Таблица (PNG) сохранена: {save_path}")


# ============================================================================
# КОНЕЦ МОДУЛЯ
# ============================================================================