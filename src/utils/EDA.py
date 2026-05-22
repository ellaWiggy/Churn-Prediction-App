import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes


def categorize_and_print_columns(dataframe, max_unique=10):
    """
    Categorizes DataFrame columns into binary, categorical, numeric categorical, and numerical types based on unique value counts and data types. 
    Prints the categorized columns in a formatted manner.
    """
    categories = {
        'binary': [],
        'categorical': [],
        'numeric_categorical': [],
        'numerical': []
    }
    for col in dataframe.columns:
        n_unique = dataframe[col].nunique()

        if n_unique == 2:
            categories['binary'].append(col)
        elif dataframe[col].dtype != 'object':
            if n_unique <= max_unique:
                categories['numeric_categorical'].append(col)
            else:
                categories['numerical'].append(col)
        else:
            categories['categorical'].append(col)

    print("\n--- Feature Categorization ---")
    
    for label, cols in categories.items():
        if not cols:
            continue      
        print(f"\n{label.capitalize()} Columns ({len(cols)}):")
        print('-' * 50)

        mid = (len(cols) + 1) // 2
        for i in range(mid):
            col1 = cols[i]
            col2 = cols[i + mid] if (i + mid) < len(cols) else ""
            print(f" {col1:<23} | {col2}")
        print('-' * 50)

    return categories

def plot_stacked_bar(ax, df_plot, col, target, palette, show_legend):
    """
    Plots a horizontal stacked bar chart for a given feature against the target variable, with annotations for counts.
    """
    pd_ct = pd.crosstab(df_plot[col], df_plot[target])[['Stay', 'Churn']]
    pd_ct.plot(kind='barh', stacked=True, ax=ax, color=[palette['Stay'], palette['Churn']], legend=show_legend, width=0.8)

    max_x = pd_ct.sum(axis=1).max()

    for i, (idx, row) in enumerate(pd_ct.iterrows()):
        stay_val = row['Stay']
        churn_val = row['Churn']
        total_val = stay_val + churn_val

        if stay_val > (max_x * 0.10):
            ax.text(stay_val / 2, i, f'{int(stay_val)}', color='white', 
                    va='center', ha='center', weight='bold', fontsize=9)
        elif stay_val > 0:
            ax.text(stay_val + (max_x * 0.04), i + 0.2, f'Stay: {int(stay_val)}', 
                    color=palette['Stay'], va='center', ha='left', weight='bold', fontsize=9)
            
        if stay_val <= (max_x * 0.15):
            ax.text(total_val + (max_x * 0.01), i - 0.2, f'Churn: {int(churn_val)}', 
                    color=palette['Churn'], va='center', ha='left', weight='bold', fontsize=9)
        else:
            ax.text(total_val + (max_x * 0.04), i, f'Churn: {int(churn_val)}', 
                    color=palette['Churn'], va='center', ha='left', weight='bold', fontsize=9)

    ax.set_xlim(0, max_x * 1.45)
    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)


def plot_kde(ax, df_plot, col, target, palette):
    """
    Plots a KDE plot for a given feature against the target variable, with vertical lines indicating class means and annotations.
    """
    sns.kdeplot(
        data=df_plot, x=col, hue=target, 
        fill=True, ax=ax, palette=palette, 
        alpha=0.5, linewidth=2
    )
    y_max = ax.get_ylim()[1]

    for label, color in palette.items():
        avg = df_plot[df_plot[target] == label][col].mean()
        ax.axvline(avg, color=color, linestyle='--', linewidth=1.5)

        offset = 0.9 if label == 'Stay' else 0.8
        ax.text(avg, y_max * offset, f'{label} Avg: {avg:.1f}', 
                color=color, fontweight='bold', ha='center', 
                bbox={'facecolor':'white', 'alpha':0.7, 'edgecolor':'none'})


def plot_feature_grid(data, features, barplot=False, target='churn'):
    """
    Plots a grid of features with either KDE plots or stacked bar charts against the target variable.
    """
    df_plot = data.copy()
    df_plot[target] = df_plot[target].map({0: 'Stay', 1: 'Churn'})

    palette = {'Stay': '#2a9d8f', 'Churn': '#e76f51'}

    n_features = len(features)
    rows = (n_features // 3) + (1 if n_features % 3 != 0 else 0)
    fig, axes = plt.subplots(rows, 3, figsize=(18, 14))
    axes = axes.flatten()

    for i, col in enumerate(features):
        ax = axes[i]
        if barplot:
            plot_stacked_bar(ax, df_plot, col, target, palette, show_legend=(i == 0))
        else:
            plot_kde(ax, df_plot, col, target, palette)

        ax.set_title(col.replace('_', ' ').upper())
        ax.set_xlabel('')
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)

    for ax in axes[len(features):]:
        ax.axis('off')
    plt.tight_layout()



def find_outliers(data, features=None, method='iqr', plot=True):
    if features is None:
        features = data.select_dtypes(include=['number']).columns
        
    outlier_indices = {}
    
    for col in features:
        series = data[col]
        if method == 'zscore':
            z_scores = (series - series.mean()) / series.std()
            outliers = series[z_scores.abs() > 3]
        else:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = series[(series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))]
        
        outlier_indices[col] = outliers.index.tolist()

    if plot:
        n_features = len(features)
        rows = (n_features // 3) + (1 if n_features % 3 != 0 else 0)
        fig, axes = plt.subplots(rows, 3, figsize=(18, 4 * rows))
        axes = axes.flatten()

        for i, col in enumerate(features):
            ax = axes[i]
            sns.boxplot(x=data[col], ax=ax, color='#2a9d8f')
            
            count = len(outlier_indices[col])
            ax.set_title(f"{col.replace('_', ' ').upper()}\n({count} Outliers)")
            
            for s in ['top', 'right', 'left']:
                ax.spines[s].set_visible(False)


        for ax in axes[len(features):]:
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
