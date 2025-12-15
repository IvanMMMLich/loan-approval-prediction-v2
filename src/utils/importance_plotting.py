"""
Функции для визуализации Permutation Importance.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_permutation_importance(importance_df, new_features=None, top_n=15, save_path=None):
    """
    Создаёт bar chart для Permutation Importance с выделением новых признаков.
    
    Args:
        importance_df: DataFrame с колонками ['Признак', 'Importance_mean', 'Importance_std']
        new_features: список новых признаков для выделения цветом (опционально)
        top_n: сколько топ признаков показать (по умолчанию 15)
        save_path: путь для сохранения графика (опционально)
    
    Returns:
        fig, ax: объекты matplotlib
    """
    
    if new_features is None:
        new_features = []
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Берём топ признаков
    df_plot = importance_df.head(top_n).copy()
    df_plot = df_plot.sort_values('Importance_mean', ascending=True)
    
    # Цвета: новые признаки — красный, старые — синий
    colors = ['#e74c3c' if f in new_features else '#3498db' for f in df_plot['Признак']]
    
    # Bar chart
    bars = ax.barh(
        range(len(df_plot)),
        df_plot['Importance_mean'],
        color=colors,
        alpha=0.7,
        edgecolor='black',
        linewidth=1
    )
    
    # Error bars (если есть std)
    if 'Importance_std' in df_plot.columns:
        ax.errorbar(
            df_plot['Importance_mean'],
            range(len(df_plot)),
            xerr=df_plot['Importance_std'],
            fmt='none',
            color='black',
            capsize=3,
            alpha=0.5
        )
    
    # Подписи по оси Y
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot['Признак'], fontsize=11)
    
    # Заголовок
    if new_features:
        title = f'Permutation Importance (Топ-{top_n} признаков)\nКрасный = новые признаки'
    else:
        title = f'Permutation Importance (Топ-{top_n} признаков)'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Importance (падение ROC-AUC при перемешивании)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Добавляем значения на столбцах
    max_val = df_plot['Importance_mean'].max()
    for i, (bar, value) in enumerate(zip(bars, df_plot['Importance_mean'])):
        ax.text(
            value + max_val * 0.02,
            i,
            f'{value:.4f}',
            va='center',
            ha='left',
            fontsize=9,
            fontweight='bold'
        )
    
    # Легенда (если есть новые признаки)
    if new_features:
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Исходные признаки'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Новые признаки')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Permutation Importance: {save_path}")
    
    plt.close()
    
    return fig, ax