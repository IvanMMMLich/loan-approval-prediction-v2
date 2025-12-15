"""
Утилиты для анализа категориальных признаков (Step 2.2 EDA)

Этот модуль содержит функции для комплексного анализа категориальных признаков:
- Частотное распределение категорий
- Approval rate по категориям (КЛЮЧЕВОЙ АНАЛИЗ!)
- Визуализация распределений
- Сравнение по классам

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
# ФУНКЦИЯ 1: ЧАСТОТНОЕ РАСПРЕДЕЛЕНИЕ
# ============================================================================

def calculate_frequency_distribution(
    df: pd.DataFrame,
    feature: str,
    save_table_image_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Вычисляет частотное распределение для категориального признака.
    
    Показывает сколько раз встречается каждая категория (абсолютное и в %).
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название категориального признака
    save_table_image_path : Optional[Union[str, Path]], optional
        Путь для сохранения PNG картинки таблицы
    
    Returns
    -------
    pd.DataFrame
        Таблица с колонками: Категория, Количество, Процент (%)
    """
    
    # Подсчёт частот
    value_counts = df[feature].value_counts()
    total = len(df)
    
    # Создаём DataFrame
    freq_df = pd.DataFrame({
        'Категория': value_counts.index.astype(str),
        'Количество': value_counts.values,
        'Процент (%)': (value_counts.values / total * 100).round(2)
    })
    
    # Сортируем по количеству (по убыванию)
    freq_df = freq_df.sort_values('Количество', ascending=False).reset_index(drop=True)
    
    # Сохраняем картинку таблицы если указан путь
    if save_table_image_path is not None:
        _save_table_as_image(
            freq_df,
            title=f'Частотное распределение: {feature}',
            save_path=save_table_image_path
        )
    
    return freq_df


# ============================================================================
# ФУНКЦИЯ 2: APPROVAL RATE ПО КАТЕГОРИЯМ (КЛЮЧЕВАЯ!)
# ============================================================================

def calculate_approval_rate_by_category(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    save_table_image_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Вычисляет approval rate (% одобренных) для каждой категории.
    
    ЭТО КЛЮЧЕВОЙ АНАЛИЗ ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ!
    Показывает какие категории "опасные" (низкий approval rate).
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название категориального признака
    target_col : str
        Название целевой переменной (обычно 'loan_status')
    save_table_image_path : Optional[Union[str, Path]], optional
        Путь для сохранения PNG картинки таблицы
    
    Returns
    -------
    pd.DataFrame
        Таблица с колонками: Категория, Всего заявок, Одобрено, Отклонено,
        Approval Rate (%)
    """
    
    # Группируем по категориям
    grouped = df.groupby(feature)[target_col].agg([
        ('Всего заявок', 'count'),
        ('Одобрено', lambda x: (x == 1).sum()),
        ('Отклонено', lambda x: (x == 0).sum())
    ]).reset_index()
    
    # Переименовываем колонку признака
    grouped = grouped.rename(columns={feature: 'Категория'})
    
    # Вычисляем approval rate
    grouped['Approval Rate (%)'] = (
        grouped['Одобрено'] / grouped['Всего заявок'] * 100
    ).round(2)
    
    # Сортируем по approval rate (по убыванию)
    grouped = grouped.sort_values('Approval Rate (%)', ascending=False).reset_index(drop=True)
    
    # Сохраняем картинку таблицы если указан путь
    if save_table_image_path is not None:
        _save_table_as_image(
            grouped,
            title=f'Approval Rate по категориям: {feature}',
            save_path=save_table_image_path,
            figsize=(14, 6)
        )
    
    return grouped


# ============================================================================
# ФУНКЦИЯ 3: BAR CHART ЧАСТОТНОГО РАСПРЕДЕЛЕНИЯ
# ============================================================================

def plot_frequency_bar(
    df: pd.DataFrame,
    feature: str,
    save_path: Union[str, Path]
) -> None:
    """
    Создаёт bar chart частотного распределения категорий.
    
    График показывает сколько раз встречается каждая категория.
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название категориального признака
    save_path : Union[str, Path]
        Путь для сохранения PNG файла
    
    Returns
    -------
    None
        График сохраняется в файл
    """
    
    # Подсчёт частот
    value_counts = df[feature].value_counts()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar chart
    bars = ax.bar(
        range(len(value_counts)),
        value_counts.values,
        color='steelblue',
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси X
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
    
    # Заголовок и подписи
    ax.set_title(f'Частотное распределение: {feature}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Категория', fontsize=14)
    ax.set_ylabel('Количество', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения над столбцами
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = (height / len(df)) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}\n({percentage:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OK: график частот сохранен: {save_path}")


# ============================================================================
# ФУНКЦИЯ 4: BAR CHART APPROVAL RATE (КЛЮЧЕВОЙ ГРАФИК!)
# ============================================================================

def plot_approval_rate_bar(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    save_path: Union[str, Path]
) -> None:
    """
    Создаёт bar chart для approval rate по категориям.
    
    ЭТО КЛЮЧЕВОЙ ГРАФИК ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ!
    Показывает какие категории имеют высокий/низкий approval rate.
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название категориального признака
    target_col : str
        Название целевой переменной (обычно 'loan_status')
    save_path : Union[str, Path]
        Путь для сохранения PNG файла
    
    Returns
    -------
    None
        График сохраняется в файл
    """
    
    # Вычисляем approval rate по категориям
    approval_rate = df.groupby(feature)[target_col].agg([
        ('count', 'count'),
        ('approved', lambda x: (x == 1).sum())
    ])
    approval_rate['rate'] = (approval_rate['approved'] / approval_rate['count'] * 100)
    approval_rate = approval_rate.sort_values('rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar chart с цветовой кодировкой
    colors = ['#2ecc71' if rate > 20 else '#e74c3c' for rate in approval_rate['rate']]
    
    bars = ax.bar(
        range(len(approval_rate)),
        approval_rate['rate'].values,
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Подписи по оси X
    ax.set_xticks(range(len(approval_rate)))
    ax.set_xticklabels(approval_rate.index, rotation=45, ha='right')
    
    # Горизонтальная линия среднего approval rate
    mean_rate = df[target_col].mean() * 100
    ax.axhline(mean_rate, color='blue', linestyle='--', linewidth=2,
               label=f'Средний Approval Rate: {mean_rate:.2f}%')
    
    # Заголовок и подписи
    ax.set_title(f'Approval Rate по категориям: {feature}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Категория', fontsize=14)
    ax.set_ylabel('Approval Rate (%)', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения над столбцами
    for i, bar in enumerate(bars):
        height = bar.get_height()
        count = approval_rate['count'].iloc[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.1f}%\n(n={count})',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OK: график approval rate сохранен: {save_path}")


# ============================================================================
# ФУНКЦИЯ 5: STACKED BAR CHART (РАСПРЕДЕЛЕНИЕ ПО КЛАССАМ)
# ============================================================================

def plot_stacked_bar_by_class(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    save_path: Union[str, Path]
) -> None:
    """
    Создаёт stacked bar chart показывающий распределение loan_status
    внутри каждой категории.
    
    График показывает сколько одобренных (зелёный) и отклонённых (красный)
    в каждой категории.
    
    Parameters
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    feature : str
        Название категориального признака
    target_col : str
        Название целевой переменной (обычно 'loan_status')
    save_path : Union[str, Path]
        Путь для сохранения PNG файла
    
    Returns
    -------
    None
        График сохраняется в файл
    """
    
    # Создаём crosstab
    ct = pd.crosstab(df[feature], df[target_col], normalize='index') * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Stacked bar chart
    ct.plot(
        kind='bar',
        stacked=True,
        color=[CLASS_COLORS[0], CLASS_COLORS[1]],
        alpha=0.7,
        edgecolor='black',
        linewidth=1,
        ax=ax
    )
    
    # Заголовок и подписи
    ax.set_title(f'Распределение по классам: {feature}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Категория', fontsize=14)
    ax.set_ylabel('Процент (%)', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend([CLASS_LABELS[0], CLASS_LABELS[1]], title='Статус заявки', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"OK: stacked bar chart сохранен: {save_path}")


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
    table.scale(1, 2)
    
    # Заголовки колонок - жирным шрифтом и с фоном
    for i in range(len(df.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white', fontsize=11)
    
    # Чередующиеся цвета строк
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
    
    print(f"OK: таблица (PNG) сохранена: {save_path}")


# ============================================================================
# КОНЕЦ МОДУЛЯ
# ============================================================================