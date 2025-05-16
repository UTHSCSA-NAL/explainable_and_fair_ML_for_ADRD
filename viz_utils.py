import os
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import nibabel as nib
from nilearn import plotting
from nibabel.processing import resample_from_to

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Rectangle, Ellipse
from matplotlib.lines import Line2D
from matplotlib import cm

from itertools import combinations
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
import fairness_metrics as FM
import scipy.special

from tqdm import tqdm
from statsmodels.nonparametric.smoothers_lowess import lowess
from concurrent.futures import ThreadPoolExecutor
from joblib import Parallel, delayed

# ---------------------------
# Visualization functions for performance and fairness metrics
# ---------------------------

def compute_metric(df, metric='ba'):
    result = []
    for row in df:
        y_true = row["DX"].values
        y_pred = np.array(row["PRED_DX"].values > 0.56, dtype=np.int32)
        if metric in ['balanced_accuracy', 'ba']:
            result.append(balanced_accuracy_score(y_true, y_pred))
        else:
            result.append(FM.calculate_fpr_fnr(y_true, row["PRED_DX"].values, 0.56)[{
                'tpr': 0, 'tnr': 1, 'fpr': 2, 'fnr': 3
            }[metric]])
    return result

def sem_func(probs):
    return np.std(probs) / np.sqrt(len(probs))

def average_absolute_difference(arr):
    return np.array([
        np.mean([abs(i - j) for i, j in combinations(arr[:, col], 2)])
        for col in range(arr.shape[1])
    ])

def generate_result(list_paths, metric=None):
    variables = []
    for path in list_paths:
        with open(path, 'rb') as f:
            _, var1, var2, var3, _ = pickle.load(f)
            variables.append({"NHW": var1, "NHA": var2, "HWA": var3})

    results = [
        {
            grp: [fold['y'] for fold in run[grp]]
            for grp in ['NHW', 'NHA', 'HWA']
        }
        for run in variables
    ]

    performance = np.array([
        [np.mean(compute_metric(results[i][grp], metric)) for i in range(4)]
        for grp in ['NHW', 'NHA', 'HWA']
    ]) * 100

    error = np.array([
        [sem_func(compute_metric(results[i][grp], metric)) for i in range(4)]
        for grp in ['NHW', 'NHA', 'HWA']
    ]) * 100

    p_value = np.array([
        [stats.ttest_ind(compute_metric(results[0]['NHW'], metric), compute_metric(results[0][grp], metric))[1] for grp in ['NHA', 'HWA']] + [stats.ttest_ind(compute_metric(results[0]['NHA'], metric), compute_metric(results[0]['HWA'], metric))[1]],
        [stats.ttest_ind(compute_metric(results[1]['NHW'], metric), compute_metric(results[1]['NHA'], metric))[1],
         stats.ttest_ind(compute_metric(results[1]['NHW'], metric), compute_metric(results[1]['HWA'], metric))[1], 0],
        [stats.ttest_ind(compute_metric(results[2]['NHA'], metric), compute_metric(results[2]['NHW'], metric))[1],
         stats.ttest_ind(compute_metric(results[2]['NHA'], metric), compute_metric(results[2]['HWA'], metric))[1], 0],
        [stats.ttest_ind(compute_metric(results[3]['HWA'], metric), compute_metric(results[3]['NHA'], metric))[1],
         stats.ttest_ind(compute_metric(results[3]['HWA'], metric), compute_metric(results[3]['NHW'], metric))[1], 0]
    ])

    return performance, error, p_value

def generate_result_from_csv(outdir, scenarios, num_folds, metric=None):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    group_names = ['nhw', 'nha', 'hisp']
    num_group = len(group_names)
    num_scenario = len(scenarios)
    
    results = {sc: {g: [] for g in group_names} for sc in scenarios}
    
    for scenario in scenarios:
        for fold in range(1, num_folds+1):
            file_path = os.path.join(outdir, f"scenario_{scenario}_fold_{fold}_predict.csv")
            df = pd.read_csv(file_path)
            df["PRED_DX"] = sigmoid(df["PREDICTION"])
            df["DX"] = df["NACCUDSD"]
            
            df_nhw = df[(df["RACE"]==1) & (df["HISPANIC"]==0)]
            df_nha = df[(df["RACE"]==2) & (df["HISPANIC"]==0)]
            df_hisp = df[(df["RACE"]==1) & (df["HISPANIC"]==1)]
            dfs = {'nhw': df_nhw, 'nha': df_nha, 'hisp': df_hisp}
            
            for g in group_names:
                if not dfs[g].empty:
                    metric_val = compute_metric([dfs[g]], metric=metric)[0]
                    results[scenario][g].append(metric_val)
                else:
                    results[scenario][g].append(np.nan)
    
    performance = np.zeros((num_group, num_scenario))
    error = np.zeros((num_group, num_scenario))
    
    for s_idx, scenario in enumerate(scenarios):
        for g_idx, group in enumerate(group_names):
            vals = np.array(results[scenario][group], dtype=float)
            valid = vals[~np.isnan(vals)]
            mean_val = valid.mean() if valid.size > 0 else np.nan
            sem_val = valid.std(ddof=1) / np.sqrt(valid.size) if valid.size > 0 else np.nan
            performance[g_idx, s_idx] = mean_val * 100
            error[g_idx, s_idx] = sem_val * 100
    
    p_value = np.zeros((num_scenario, 3))
    for s_idx, scenario in enumerate(scenarios):
        arr_nhw = np.array(results[scenario]['nhw'], dtype=float)
        arr_nha = np.array(results[scenario]['nha'], dtype=float)
        arr_hisp = np.array(results[scenario]['hisp'], dtype=float)
        arr_nhw = arr_nhw[~np.isnan(arr_nhw)]
        arr_nha = arr_nha[~np.isnan(arr_nha)]
        arr_hisp = arr_hisp[~np.isnan(arr_hisp)]
        
        if s_idx == 0:
            # Row 0: arr_nhw vs arr_nha, arr_nhw vs arr_hisp, arr_nha vs arr_hisp
            p1 = stats.ttest_ind(arr_nhw, arr_nha, equal_var=False)[1] if arr_nhw.size and arr_nha.size else np.nan
            p2 = stats.ttest_ind(arr_nhw, arr_hisp, equal_var=False)[1] if arr_nhw.size and arr_hisp.size else np.nan
            p3 = stats.ttest_ind(arr_nha, arr_hisp, equal_var=False)[1] if arr_nha.size and arr_hisp.size else np.nan
        elif s_idx == 1:
            # Row 1: arr_nhw vs arr_nha, arr_nhw vs arr_hisp, 0
            p1 = stats.ttest_ind(arr_nhw, arr_nha, equal_var=False)[1] if arr_nhw.size and arr_nha.size else np.nan
            p2 = stats.ttest_ind(arr_nhw, arr_hisp, equal_var=False)[1] if arr_nhw.size and arr_hisp.size else np.nan
            p3 = 0.0
        elif s_idx == 2:
            # Row 2: arr_nha vs arr_nhw, arr_nha vs arr_hisp, 0
            p1 = stats.ttest_ind(arr_nha, arr_nhw, equal_var=False)[1] if arr_nha.size and arr_nhw.size else np.nan
            p2 = stats.ttest_ind(arr_nha, arr_hisp, equal_var=False)[1] if arr_nha.size and arr_hisp.size else np.nan
            p3 = 0.0
        elif s_idx == 3:
            # Row 3: arr_hisp vs arr_nhw, arr_hisp vs arr_nha, 0
            p1 = stats.ttest_ind(arr_hisp, arr_nhw, equal_var=False)[1] if arr_hisp.size and arr_nhw.size else np.nan
            p2 = stats.ttest_ind(arr_hisp, arr_nha, equal_var=False)[1] if arr_hisp.size and arr_nha.size else np.nan
            p3 = 0.0
        p_value[s_idx, :] = [p1, p2, p3]
    
    return performance, error, p_value

def plot_bar_chart(performance, errors, groups, datasets, legend, config,
                   p_value=None, metric=None, title=None, save_path=None,
                   plot_avg=False, font_weight='normal'):
    ###############################################
    # - config:
    #   + [0] - y position of the text
    #   + [1] - width rate
    #   + [2] - height
    #   + [3] - first line average text
    #   + [4] - second line average gap text
    #   + [5] - y limit min
    #   + [6] - y limit max
    #
    ###############################################
    indices = (0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11)
    new_dataset = [datasets[i] for i in indices]
    df = pd.DataFrame({
        'Group' : np.repeat(groups, 3),
        'Dataset': datasets,
        'Legend': np.tile(legend, 4),
        'Performance': performance.flatten(order='F'),
        'Error': errors.flatten(order='F')
    })

    custom_palette = {
        "Training": "white",
        "Test": "white",
        legend[0]: "#c6dbef",
        legend[1]: "#f2d0a9",
        legend[2]: "#c5e8b7",
    }

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(16, 9))
    ax = sns.barplot(data=df, x='Group', y='Performance', hue='Legend',
                     palette=custom_palette, errorbar=None,
                     capsize=0.1, linewidth=2.5, edgecolor="black")

    for i, bar in enumerate(ax.patches):
        if i < len(new_dataset) and new_dataset[i] == "Test":
            bar.set_linestyle("dashed")

    legend_elements = [
        Line2D([0], [0], color='black', linestyle='solid', lw=2.5, label='Training'),
        Line2D([0], [0], color='black', linestyle='dashed', lw=2.5, label='Test'),
        *[Patch(facecolor=custom_palette[l], edgecolor='black', label=l) for l in legend]
    ]

    for i, bar in enumerate(ax.patches):
        if i < len(indices):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ax.errorbar(x, y, yerr=df['Error'][indices[i]], fmt='none', ecolor='black', capsize=4)

    if plot_avg:
        bar_width = 0.2
        avg_vals = np.mean(performance, axis=0)
        avg_diffs = average_absolute_difference(performance)
        for i, (avg, fa) in enumerate(zip(avg_vals, avg_diffs)):
            rect = Rectangle((i - bar_width - 0.27, config[0]), config[1] * bar_width, config[2], color=config[5], alpha=0.3)
            ax.add_patch(rect)
            ax.text(i, config[3], f'Avg = {avg:.1f}%', ha='center', fontsize=31, fontweight=font_weight)
            ax.text(i, config[4], f'AvD = {fa:.1f}%', ha='center', fontsize=31, fontweight=font_weight)
    
    # p-value annotations (if provided)
    if p_value is not None and p_value.any():
        for i, group in enumerate(groups):
            if i in [0, 1, 2, 3]:
                if i == 0:
                    valid_bar = ax.containers[0].patches[i]
                    test_bar_1 = ax.containers[1].patches[i]
                    test_bar_2 = ax.containers[2].patches[i]
                elif i == 1:
                    valid_bar = ax.containers[0].patches[i]
                    test_bar_1 = ax.containers[1].patches[i]
                    test_bar_2 = ax.containers[2].patches[i]
                elif i == 2:
                    valid_bar = ax.containers[1].patches[i]
                    test_bar_1 = ax.containers[0].patches[i]
                    test_bar_2 = ax.containers[2].patches[i]
                elif i == 3:
                    valid_bar = ax.containers[2].patches[i]
                    test_bar_1 = ax.containers[1].patches[i]
                    test_bar_2 = ax.containers[0].patches[i]

                height_1 = max(valid_bar.get_height(), test_bar_1.get_height()) + 1
                height_2 = max(valid_bar.get_height(), test_bar_2.get_height()) + 1
                max_height = max(height_1, height_2)

                # First comparison annotation
                x_start = valid_bar.get_x() + valid_bar.get_width() / 2
                x_end = test_bar_1.get_x() + test_bar_1.get_width() / 2
                ax.plot([x_start, x_end], [max_height+6, max_height+6], 'k-', lw=1.5)
                ax.plot([x_start, x_start], [max_height+5.5, max_height+6], 'k-', lw=1.5)
                ax.plot([x_end, x_end], [max_height+5.5, max_height+6], 'k-', lw=1.5)
                pv_text = "NS" if p_value[i, 0] >= 0.05 else "p<0.05"
                ax.text((x_start + x_end) / 2, max_height+6.5, pv_text, ha='center', va='bottom', fontsize=17)

                # Second comparison annotation
                x_start = valid_bar.get_x() + valid_bar.get_width() / 2
                x_end = test_bar_2.get_x() + test_bar_2.get_width() / 2
                ax.plot([x_start, x_end], [max_height+10.5, max_height+10.5], 'k-', lw=1.5)
                ax.plot([x_start, x_start], [max_height+10, max_height+10.5], 'k-', lw=1.5)
                ax.plot([x_end, x_end], [max_height+10, max_height+10.5], 'k-', lw=1.5)
                pv_text = "NS" if p_value[i, 1] >= 0.05 else "p<0.05"
                ax.text((x_start + x_end) / 2, max_height+11.5, pv_text, ha='center', va='bottom', fontsize=17)
                
                # Third p-value annotation (for the first group)
                if i == 0:
                    x_start = test_bar_1.get_x() + test_bar_1.get_width() / 2
                    x_end = test_bar_2.get_x() + test_bar_2.get_width() / 2
                    ax.plot([x_start, x_end], [max_height+2, max_height+2], 'k-', lw=1.5)
                    ax.plot([x_start, x_start], [max_height+1.5, max_height+2], 'k-', lw=1.5)
                    ax.plot([x_end, x_end], [max_height+1.5, max_height+2], 'k-', lw=1.5)
                    pv_text = "NS" if p_value[i, 2] >= 0.05 else "p<0.05"
                    ax.text((x_start + x_end) / 2, max_height+2.5, pv_text, ha='center', va='bottom', fontsize=17)

    ax.set_xlabel('')
    ax.set_ylabel(f'{metric} (%)', fontsize=36)
    ax.set_title(title, loc='left', fontsize=44)
    plt.xticks(np.arange(4), labels=['ALL', 'NHW', 'NHA', 'HISP'], fontsize=36)
    plt.yticks(np.arange(config[6], config[7], 10), fontsize=36)
    plt.ylim(config[6], config[7]+1)
    if metric == 'Balanced Accuracy':
        ax.legend(handles=legend_elements, loc='lower right', fontsize=30)
    elif metric == 'FNR':
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.001, 0.60), loc='center left', fontsize=30)
    elif metric == 'FPR':
        ax.legend(handles=legend_elements, bbox_to_anchor=(0.001, 0.50), loc='center left', fontsize=30)
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def plot_confidence_ellipse(ax, x, y, n_std=2.0, **kwargs):
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    cov = np.cov(x, y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    ellipse = Ellipse((mean_x, mean_y), width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
def pareto_front(x, y):
    # we sort by y (performance) descending.
    points = sorted(zip(x, y), key=lambda pt: pt[1], reverse=True)
    pareto_points = [points[0]]
    for point in points[1:]:
        # Accept point if its fairness value is lower (better) than the last accepted point.
        if point[0] < pareto_points[-1][0]:
            pareto_points.append(point)
    return zip(*pareto_points)

def compute_performance(list_paths, metric):
    # Helper functions to determine labels from the file path.
    def get_model_name(path):
        s = path.lower()
        contains_kmm = "kmm" in s
        contains_cr  = "cr" in s
        contains_harm = "harm" in s
        if contains_kmm and contains_cr and contains_harm:
            return "Harm_CR_KMM"
        elif contains_kmm and contains_cr:
            return "CR_KMM"
        elif contains_kmm and contains_harm:
            return "Harm_KMM"
        elif contains_cr and contains_harm:
            return "Harm_CR"
        elif contains_kmm:
            return "KMM"
        elif contains_cr:
            return "CR"
        elif contains_harm:
            return "Harm"
        else:
            return "Base"

    def get_scenario_name(path):
        s = path.lower()
        # Check for scenario tags in a prioritized order.
        if "all" in s:
            return "ALL"
        elif "nhw" in s:
            return "NHW"
        elif "nha" in s:
            return "NHA"
        elif "hispanic" in s:
            return "HISP"
        else:
            return "Unknown"

    # Determine labels and load pickle files.
    model_labels = []
    scenario_labels = []
    variables = []  # each element is a dict with keys "NHW", "NHA", and "HISP"
    for path in list_paths:
        model_labels.append(get_model_name(path))
        scenario_labels.append(get_scenario_name(path))
        with open(path, 'rb') as f:
            _, var1, var2, var3, _ = pickle.load(f)
            # Do not extract only the "y" values so that the structure remains intact.
            variables.append({
                "NHW": var1,
                "NHA": var2,
                "HISP": var3,
            })

    results = []
    for var in variables:
        preds = {
            "NHW": [var["NHW"][i]['y'] for i in range(len(var["NHW"]))],
            "NHA": [var["NHA"][i]['y'] for i in range(len(var["NHA"]))],
            "HISP": [var["HISP"][i]['y'] for i in range(len(var["HISP"]))],
        }
        results.append(preds)
    
    df = pd.DataFrame()
    df.index = range(len(model_labels)*10)
    for i in range(len(model_labels)):
        df.loc[10*i:10*(i+1), 'Model'] = model_labels[i]
        df.loc[10*i:10*(i+1), 'Scenario'] = scenario_labels[i]
        df.loc[10*i:10*(i+1)-1, 'Fold'] = list(range(10))
        df.loc[10*i:10*(i+1)-1, 'NHW'] = compute_metric(results[i]["NHW"], metric=metric)
        df.loc[10*i:10*(i+1)-1, 'NHA'] = compute_metric(results[i]["NHA"], metric=metric)
        df.loc[10*i:10*(i+1)-1, 'HISP'] = compute_metric(results[i]["HISP"], metric=metric)
    df['Fold'] = df['Fold'].astype(int)
    return df

def compute_metric_new(df_list, metric='ba', threshold=0.56):
    result = []
    for df in df_list:
        y_true = df['NACCUDSD'].values
        y_score = scipy.special.expit(df['PREDICTION'].values)  # apply sigmoid
        y_pred = (y_score > threshold).astype(int)
        if metric in ['ba', 'balanced_accuracy']:
            result.append(balanced_accuracy_score(y_true, y_pred))
        else:
            fpr = np.sum((y_true == 0) & (y_pred == 1)) / np.sum(y_true == 0)
            fnr = np.sum((y_true == 1) & (y_pred == 0)) / np.sum(y_true == 1)
            result.append(fpr if metric == 'fpr' else fnr)
    return result

def compute_performance_new(list_dirs, metric='ba'):
    all_records = []

    for model_dir in list_dirs:
        if '1B' in model_dir:
            model_name = 'RegAlign'
        elif '2B' in model_dir:
            model_name = 'SSDA'
        else:
            model_name = 'Unknown'
        for scenario in ['all', 'nhw', 'nha', 'hisp']:
            for fold in range(1, 11):
                file_path = os.path.join(model_dir, f'scenario_{scenario}_fold_{fold}_predict.csv')
                if not os.path.exists(file_path):
                    continue

                df = pd.read_csv(file_path)
                df['group'] = df.apply(
                    lambda row: 'NHW' if (row['RACE'] == 1 and row['HISPANIC'] == 0)
                    else 'NHA' if (row['RACE'] == 2 and row['HISPANIC'] == 0)
                    else 'HISP' if (row['RACE'] == 1 and row['HISPANIC'] == 1)
                    else 'Other', axis=1
                )
                group_metrics = {}
                for g in ['NHW', 'NHA', 'HISP']:
                    df_group = df[df['group'] == g]
                    if df_group.empty:
                        group_metrics[g] = np.nan
                    else:
                        group_metrics[g] = compute_metric_new([df_group], metric)[0]

                all_records.append({
                    'Model': model_name,
                    'Scenario': scenario.upper() if scenario != 'hisp' else 'HISP',
                    'Fold': fold - 1,
                    'NHW': group_metrics['NHW'],
                    'NHA': group_metrics['NHA'],
                    'HISP': group_metrics['HISP']
                })

    return pd.DataFrame(all_records)

###############################################
# PARETO FRONT VISUALIZATION FUNCTION
###############################################

from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np

def plot_fit_central_pareto_front(performance, methods, groups, fig_config,
                                  centroid_type, metric_titles):
    def plot_confidence_ellipse(ax, x, y, n_std=1.2, **kwargs):
        cov = np.cov(x, y)
        mean_x, mean_y = np.mean(x), np.mean(y)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * n_std * np.sqrt(eigvals)
        ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle, **kwargs)
        ax.add_patch(ellipse)

    x_all, y_all = performance
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.grid(True)

    if centroid_type == 'method':
        labels = methods
        entities = list(dict.fromkeys(methods))
        color_map = {e: 'black' for e in entities}
        marker_list = ["^", "x", "_", "o", "+", "d", "s", "P", "8"]
    else:
        labels = groups
        entities = list(dict.fromkeys(groups))
        cmap = 'royalblue'
        color_map = {e: cmap for i, e in enumerate(entities)}
        marker_list = ["^", "8", "s", "P", "x", "o", "d"]

    # Generate distinct colors from colormap
    colormap = plt.cm.get_cmap('tab10', len(entities))
    color_list = [colormap(i) for i in range(len(entities))]
    color_names = {}

    style = {
        ent: {
            "marker": marker_list[i % len(marker_list)],
            "color": color_list[i],
            "alpha": 0.8,
            "label": ent.replace("_", " + ") if centroid_type == 'method' else ent
        }
        for i, ent in enumerate(entities)
    }

    for ent, cfg in style.items():
        idx = [i for i, val in enumerate(labels) if val == ent]
        x = x_all[idx]
        y = y_all[idx]
        fc = 'none' if cfg["marker"] in ['^', 'o', '8', 's', 'P', 'd'] else cfg["color"]
        ec = cfg["color"]
        ax.scatter(x, y, marker=cfg["marker"], facecolors=fc, edgecolors=ec,
                   s=150, alpha=cfg["alpha"], label=cfg["label"])
        plot_confidence_ellipse(ax, x, y, edgecolor=cfg["color"],
                                facecolor='none', linestyle='--')

    def pareto_front(x, y):
        points = sorted(zip(x, y), key=lambda p: p[1])
        front = [points[0]]
        for pt in points[1:]:
            if pt[0] > front[-1][0]:
                front.append(pt)
        return zip(*front)

    px, py = pareto_front(x_all, y_all)
    ax.plot(px, py, color='grey', lw=3, alpha=0.6, label="Pareto front")

    ideal_x, ideal_y = np.max(x_all), np.min(y_all)
    ax.scatter(ideal_x, ideal_y, c='red', marker='*', s=250, label='Ideal point')

    # Plot centroids
    centroids = {}
    for ent in entities:
        idx = [i for i, val in enumerate(labels) if val == ent]
        cen_x, cen_y = np.mean(x_all[idx]), np.mean(y_all[idx])
        centroids[ent] = (cen_x, cen_y)
        marker = style[ent]["marker"]
        color = style[ent]["color"]
        if marker in ['^', 'o']:
            ax.scatter(cen_x, cen_y, marker=marker, c=[color], s=300, linewidths=3)
        else:
            ax.scatter(cen_x, cen_y, marker=marker, c=[color], s=300, linewidths=3)

    # Print out the color assignments
    print("Centroid Colors:")
    for ent in entities:
        rgba = style[ent]["color"]
        print(f"{ent}: RGBA {tuple(np.round(rgba, 3))}")

    min_dist, closest_ent = float('inf'), None
    for ent, (cx, cy) in centroids.items():
        dist = np.sqrt((cx - ideal_x) ** 2 + (cy - ideal_y) ** 2)
        if dist < min_dist:
            min_dist = dist
            closest_ent = ent

    if closest_ent:
        cx, cy = centroids[closest_ent]
        ax.plot([ideal_x, cx], [ideal_y, cy], linestyle='--', color='grey', lw=1)
        ax.text(ideal_x + 2.5, ideal_y - 1.5, 'Ideal Point',
                fontsize=14, color='red', ha='right')

    ax.set_xlabel(f"Overall {metric_titles[0]} (%)", fontsize=30)
    ax.set_ylabel(f"Overall {metric_titles[1]} (%)", fontsize=30)
    ax.set_xlim([65, 90])
    ax.set_ylim([fig_config[0], fig_config[1]])
    ax.tick_params(labelsize=24)

    ax.arrow(66.5, fig_config[2], fig_config[3], 0,
             lw=2, head_width=fig_config[4], head_length=fig_config[5])
    ax.text(67, fig_config[6], 'Better performance', fontsize=24)

    ax.arrow(87.5, fig_config[7], 0, fig_config[8],
             lw=2, head_width=fig_config[9], head_length=fig_config[10])
    ax.text(88, fig_config[11], 'Better fairness', fontsize=24, rotation=270)

    # Filter legend (exclude duplicates and centroid markers)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend(*zip(*filtered), loc='center left', fontsize=20)

    plt.tight_layout()
    plt.show()

# ---------------------------
# Visualization functions for feature maps
# ---------------------------
# Dictionary mapping feature codes to brain region names.
brain_regions = {
    "4": "3rd Ventricle-VN", "11": "4th Ventricle-VN", "23": "R Accumbens-GM", "30": "L Accumbens-GM", 
    "31": "R Amygdala-GM", "32": "L Amygdala-GM", "36": "R Caudate-GM", "37": "L Caudate-GM", 
    "38": "R Cerebellum Ext-GM", "39": "L Cerebellum Ext-GM", "40": "R Cerebellum WM-WM", 
    "41": "L Cerebellum WM-WM", "47": "R Hippocampus-GM", "48": "L Hippocampus-GM", 
    "49": "R Inf Lat Vent-VN", "50": "L Inf Lat Vent-VN", "51": "R Lat Ventricle-VN", 
    "52": "L Lat Ventricle-VN", "55": "R Pallidum-GM", "56": "L Pallidum-GM", "57": "R Putamen-GM", 
    "58": "L Putamen-GM", "59": "R Thalamus-GM", "60": "L Thalamus-GM", "61": "R Ventral DC-WM", 
    "62": "L Ventral DC-WM", "71": "Vermis I-V-GM", "72": "Vermis VI-VII-GM", "73": "Vermis VIII-X-GM", 
    "75": "L Basal Forebrain-GM", "76": "R Basal Forebrain-GM", "81": "R Frontal WM-WM", 
    "82": "L Frontal WM-WM", "83": "R Occipital WM-WM", "84": "L Occipital WM-WM", 
    "85": "R Parietal WM-WM", "86": "L Parietal WM-WM", "87": "R Temporal WM-WM", 
    "88": "L Temporal WM-WM", "89": "R Fornix-WM", "90": "L Fornix-WM", "91": "R ALIC-WM", 
    "92": "L ALIC-WM", "93": "R PLIC-WM", "94": "L PLIC-WM", "95": "Corpus Callosum-WM", 
    "100": "R Ant Cingulate-GM", "101": "L Ant Cingulate-GM", "102": "R Ant Insula-GM", 
    "103": "L Ant Insula-GM", "104": "R Ant Orb Gyrus-GM", "105": "L Ant Orb Gyrus-GM", 
    "106": "R Angular Gyrus-GM", "107": "L Angular Gyrus-GM", "108": "R Calcarine-GM", 
    "109": "L Calcarine-GM", "112": "R Central Operc-GM", "113": "L Central Operc-GM", 
    "114": "R Cuneus-GM", "115": "L Cuneus-GM", "116": "R Entorhinal-GM", "117": "L Entorhinal-GM", 
    "118": "R Frontal Operc-GM", "119": "L Frontal Operc-GM", "120": "R Frontal Pole-GM", 
    "121": "L Frontal Pole-GM", "122": "R Fusiform-GM", "123": "L Fusiform-GM", 
    "124": "R Gyrus Rectus-GM", "125": "L Gyrus Rectus-GM", "128": "R Inf Occipital-GM", 
    "129": "L Inf Occipital-GM", "132": "R Inf Temporal-GM", "133": "L Inf Temporal-GM", 
    "134": "R Lingual-GM", "135": "L Lingual-GM", "136": "R Lat Orb Gyrus-GM", 
    "137": "L Lat Orb Gyrus-GM", "138": "R Mid Cingulate-GM", "139": "L Mid Cingulate-GM", 
    "140": "R Med Frontal-GM", "141": "L Med Frontal-GM", "142": "R Mid Frontal-GM", 
    "143": "L Mid Frontal-GM", "144": "R Mid Occipital-GM", "145": "L Mid Occipital-GM", 
    "146": "R Med Orb Gyrus-GM", "147": "L Med Orb Gyrus-GM", "148": "R Med Postcentral-GM", 
    "149": "L Med Postcentral-GM", "150": "R Med Precentral-GM", "151": "L Med Precentral-GM", 
    "152": "R Med Sup Frontal-GM", "153": "L Med Sup Frontal-GM", "154": "R Mid Temporal-GM", 
    "155": "L Mid Temporal-GM", "156": "R Occipital Pole-GM", "157": "L Occipital Pole-GM", 
    "160": "R Occ Fusiform-GM", "161": "L Occ Fusiform-GM", "162": "R OpIFG-GM", 
    "163": "L OpIFG-GM", "164": "R OrIFG-GM", "165": "L OrIFG-GM", "166": "R Post Cingulate-GM", 
    "167": "L Post Cingulate-GM", "168": "R Precuneus-GM", "169": "L Precuneus-GM", 
    "170": "R Parahippocampal-GM", "171": "L Parahippocampal-GM", "172": "R Post Insula-GM", 
    "173": "L Post Insula-GM", "174": "R Parietal Operc-GM", "175": "L Parietal Operc-GM", 
    "176": "R Postcentral-GM", "177": "L Postcentral-GM", "178": "R Post Orb Gyrus-GM", 
    "179": "L Post Orb Gyrus-GM", "180": "R Planum Polare-GM", "181": "L Planum Polare-GM", 
    "182": "R Precentral-GM", "183": "L Precentral-GM", "184": "R Planum Temp-GM", 
    "185": "L Planum Temp-GM", "186": "R Subcallosal-GM", "187": "L Subcallosal-GM", 
    "190": "R Sup Frontal-GM", "191": "L Sup Frontal-GM", "192": "R Supp Motor-GM", 
    "193": "L Supp Motor-GM", "194": "R Supramarginal-GM", "195": "L Supramarginal-GM", 
    "196": "R Sup Occipital-GM", "197": "L Sup Occipital-GM", "198": "R Sup Parietal-GM", 
    "199": "L Sup Parietal-GM", "200": "R Sup Temporal-GM", "201": "L Sup Temporal-GM", 
    "202": "R Temporal Pole-GM", "203": "L Temporal Pole-GM", "204": "R TrIFG-GM", 
    "205": "L TrIFG-GM", "206": "R Transv Temp Gyrus-GM", "207": "L Transv Temp Gyrus-GM"
}

def load_dataset_unique(data_pkl_path):
    """
    Load dataset pickle and build unique samples based on the sample ID (assumed in column 0).
    Returns:
      - X_all_unique: DataFrame of unique samples.
      - id2fold_rows: Mapping from sample ID to list of (fold, row_index) occurrences.
      - X_folds_list: List of DataFrames per fold (concatenation of NHW, NHA, and HWA).
      - feature_names: List of feature names (all columns except the first, which is the ID).
    """
    with open(data_pkl_path, 'rb') as f:
        train, nhw, nha, hwa, est = pickle.load(f)
    
    X_folds_list = []
    n_folds = len(nhw)
    for fold in range(n_folds):
        X_nhw = nhw[fold]['data']
        X_nha = nha[fold]['data']
        X_hwa = hwa[fold]['data']
        X_fold = pd.concat([X_nhw, X_nha, X_hwa], axis=0)
        X_folds_list.append(X_fold)
    
    unique_rows = []
    id2fold_rows = {}
    for fold_idx, df in enumerate(X_folds_list):
        ids = df.iloc[:, 0].astype(str).values  # first column is ID
        for row_idx, sample_id in enumerate(ids):
            if sample_id not in id2fold_rows:
                id2fold_rows[sample_id] = [(fold_idx, row_idx)]
                unique_rows.append(df.iloc[row_idx])
            else:
                id2fold_rows[sample_id].append((fold_idx, row_idx))
    
    X_all_unique = pd.DataFrame(unique_rows).reset_index(drop=True)
    feature_names = list(X_all_unique.columns[1:])  # exclude ID column
    return X_all_unique, id2fold_rows, X_folds_list, feature_names

def extract_iter_number(filename):
    match = re.search(r'iter_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def get_bootstrap_file_list(bootstrap_dir):
    files = [os.path.join(bootstrap_dir, f) for f in os.listdir(bootstrap_dir) if f.endswith('.pkl')]
    return sorted(files, key=extract_iter_number)

def get_valid_indices(x, factor=1.5):
    """
    Return indices within  [Q1 - factor*IQR, Q3 + factor*IQR]
    """
    Q1 = np.percentile(x, 25)
    Q3 = np.percentile(x, 75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    valid = np.where((x >= lower) & (x <= upper))[0]
    return valid if valid.size > 0 else np.arange(len(x))

def compute_metrics_for_feature(x_feat, valid_idx, feature_index, file_list, NBINS, IQR_FACTOR):
    """
    For a given feature vector (x_feat), stream over bootstrap files.
    Compute the mean of the absolute SHAP values for valid indices, and derive 
    reliability and precision measures.
    """
    shap_vals_all_iter = []
    bins = np.linspace(x_feat[valid_idx].min(), x_feat[valid_idx].max(), NBINS+1)
    bin_idx = np.clip(np.digitize(x_feat[valid_idx], bins, right=True)-1, 0, NBINS-1)
    
    for file in tqdm(file_list, desc="Processing bootstrap files", leave=False):
        with open(file, 'rb') as f:
            shap_folds = pickle.load(f)
        # Concatenate across folds (each fold may be an array or DataFrame)
        shap_all = np.concatenate([
            fold.values if hasattr(fold, "values") else fold
            for fold in shap_folds
        ], axis=0)
        # Compute for this feature at valid indices.
        shap_feat = shap_all[valid_idx, feature_index]
        shap_vals_all_iter.append(np.mean(np.abs(shap_feat)))
    
    shap_vals_all_iter = np.array(shap_vals_all_iter)
    mean_val = np.mean(shap_vals_all_iter)
    std_val = np.std(shap_vals_all_iter, ddof=1)
    CV = std_val / (mean_val + 1e-8)
    reliability = max(0, (1 - min(CV, 1))) * 100
    ci_lower = np.percentile(shap_vals_all_iter, 2.5)
    ci_upper = np.percentile(shap_vals_all_iter, 97.5)
    precision = max(0, (1 - ((ci_upper - ci_lower) / (mean_val + 1e-8)))) * 100
    return mean_val, std_val, reliability, precision

def compute_feature_stats(X_all_unique, feature_names, file_list, IQR_FACTOR, NBINS):
    """
    For each feature (based on unique samples), compute SHAP-based metrics 
    and derive reliability and precision.
    """
    feature_stats = []
    for idx, feat in enumerate(tqdm(feature_names, desc="Processing Features")):
        x_feat = X_all_unique[feat].astype(float).values
        valid_idx = get_valid_indices(x_feat, factor=IQR_FACTOR)
        mean_val, std_val, reliability, precision = compute_metrics_for_feature(
            x_feat, valid_idx, idx, file_list, NBINS, IQR_FACTOR)
        feature_stats.append({
            'feature_name': feat,
            'mean_shap': mean_val,
            'std_shap': std_val,
            'reliability': reliability,
            'precision': precision
        })
    df_stats = pd.DataFrame(feature_stats)
    df_stats['rank_reliability'] = df_stats['reliability'].rank(ascending=False)
    df_stats['rank_precision'] = df_stats['precision'].rank(ascending=False)
    df_stats['composite_rank'] = np.sqrt(df_stats['rank_reliability']**2 + df_stats['rank_precision']**2)
    return df_stats

def select_paired_features(df_stats, brain_regions, TOP_K):
    """
    Select left/right paired features for visualization.
    
    Looks for feature names that map (via brain_regions) to strings starting
    with "L " or "R ". Returns the top TOP_K pairs based on a composite rank.
    """
    df_stats['region_side'] = df_stats['feature_name'].apply(lambda x: brain_regions.get(x, ""))
    # Only include features whose brain region string starts with "L " or "R "
    paired_df = df_stats[df_stats['region_side'].str.startswith(("L ", "R "))].copy()
    paired_df['region'] = paired_df['region_side'].apply(lambda x: x[2:] if x else "")
    pair_list = []
    for region, group in paired_df.groupby('region'):
        if group['region_side'].str.contains("L ").any() and group['region_side'].str.contains("R ").any():
            left_row = group[group['region_side'].str.startswith("L ")].iloc[0]
            right_row = group[group['region_side'].str.startswith("R ")].iloc[0]
            avg_reliability = (left_row['reliability'] + right_row['reliability']) / 2
            avg_precision = (left_row['precision'] + right_row['precision']) / 2
            avg_composite = (avg_reliability + avg_precision) / 2
            pair_list.append({
                'region': region,
                'L_feature': left_row['feature_name'],
                'R_feature': right_row['feature_name'],
                'avg_reliability': avg_reliability,
                'avg_precision': avg_precision,
                'avg_composite': avg_composite
            })
    pairs_df = pd.DataFrame(pair_list)
    return pairs_df.sort_values(by='avg_composite', ascending=False).head(TOP_K)

def compute_bootstrap_binned_for_feature(feature_index, feature_names, X_all_unique, X_folds_list, file_list, NBINS, IQR_FACTOR, id2fold_rows):
    """
    For the given feature (index in feature_names), stream over bootstrap files.
    For each bootstrap iteration, aggregate SHAP values for unique samples and then bin the aggregated vector.
    
    Returns:
      - bin_centers: 1D array of bin centers (length = num_bins)
      - binned_matrix: 2D array (num_iterations x num_bins) of the binned mean aggregated SHAP values.
    """
    feat_name = feature_names[feature_index]
    x_vals = X_all_unique[feat_name].astype(float).values
    order = np.argsort(x_vals)
    bins_indices = np.array_split(order, NBINS)
    bin_centers = np.array([np.mean(x_vals[indices]) for indices in bins_indices])
    
    unique_ids = X_all_unique.iloc[:, 0].astype(str).values
    sample_id_to_index = {sid: idx for idx, sid in enumerate(unique_ids)}
    
    # Precompute valid indices per fold for current feature.
    fold_valid = []
    for fold in range(len(X_folds_list)):
        if feat_name in X_folds_list[fold].columns:
            values = X_folds_list[fold][feat_name].astype(float).values
        else:
            values = X_folds_list[fold].iloc[:, 1+feature_index].astype(float).values
        valid_idx = set(get_valid_indices(values, factor=IQR_FACTOR))
        fold_valid.append(valid_idx)
    
    binned_values = []
    for file in tqdm(file_list, desc=f"Binning for {feat_name}", leave=False):
        with open(file, 'rb') as f:
            shap_folds = pickle.load(f)
        # Initialize aggregated vector for unique samples.
        agg_vector = np.full(x_vals.shape, np.nan)
        # For each unique sample, check across folds (using mapping id2fold_rows)
        for sample_id, occ_list in id2fold_rows.items():
            sample_vals = []
            for (fold, row_idx) in occ_list:
                if row_idx in fold_valid[fold]:
                    data_fold = shap_folds[fold]
                    if hasattr(data_fold, "iloc"):
                        val = data_fold.iloc[row_idx, feature_index]
                    else:
                        val = data_fold[row_idx, feature_index]
                    sample_vals.append(val)
            if sample_vals:
                agg_val = np.mean(sample_vals)
            else:
                agg_val = np.nan
            if sample_id in sample_id_to_index:
                agg_vector[sample_id_to_index[sample_id]] = agg_val
        # Bin the aggregated vector.
        binned_iter = [np.nanmean(agg_vector[indices]) if len(indices) > 0 else np.nan for indices in bins_indices]
        binned_values.append(binned_iter)
    binned_matrix = np.array(binned_values)
    return bin_centers, binned_matrix

def plot_top_pairs(df_stats, top_pairs, feature_names, X_all_unique, X_folds_list, file_list, IQR_FACTOR, NBINS, id2fold_rows):
    """
    Create subplots displaying the binned aggregated SHAP profiles for left/right paired features.
    Each subplot shows scatter and a smoothed line with a 95% confidence interval.
    """
    fig, axs = plt.subplots(3, 6, figsize=(18, 9))
    axs = axs.flatten()
    plot_idx = 0

    for _, pair in top_pairs.iterrows():
        # Process left feature.
        feat_left = pair['L_feature']
        fi_left = feature_names.index(feat_left)
        ax_left = axs[plot_idx]
        bin_centers, binned_matrix = compute_bootstrap_binned_for_feature(
            fi_left, feature_names, X_all_unique, X_folds_list, file_list, NBINS, IQR_FACTOR, id2fold_rows)
        binned_mean = np.nanmean(binned_matrix, axis=0)
        binned_lower = np.nanpercentile(binned_matrix, 2.5, axis=0)
        binned_upper = np.nanpercentile(binned_matrix, 97.5, axis=0)
        smooth_mean = lowess(binned_mean, bin_centers, frac=0.3, return_sorted=True)
        smooth_lower = lowess(binned_lower, bin_centers, frac=0.3, return_sorted=True)
        smooth_upper = lowess(binned_upper, bin_centers, frac=0.3, return_sorted=True)
        ax_left.scatter(bin_centers, binned_mean, color='black', s=8, label='Mean SHAP')
        ax_left.plot(smooth_mean[:, 0], smooth_mean[:, 1], color='black', label='Mean line')
        ax_left.fill_between(smooth_mean[:,0], smooth_lower[:,1], smooth_upper[:,1], color='blue', alpha=0.2, label='95% CI')
        ax_left.axhline(y=0, color='red', linestyle='--')
        ax_left.set_xlabel(f"{brain_regions.get(feat_left, feat_left)} (Volume)", fontsize=10)
        row_left = df_stats[df_stats['feature_name'] == feat_left].iloc[0] 
        ax_left.set_title(f"Rel {row_left['reliability']:.2f}, CIT {row_left['precision']:.2f}", fontsize=10)
        ax_left.legend(fontsize=8)
        plot_idx += 1

        # Process right feature.
        feat_right = pair['R_feature']
        fi_right = feature_names.index(feat_right)
        ax_right = axs[plot_idx]
        bin_centers, binned_matrix = compute_bootstrap_binned_for_feature(
            fi_right, feature_names, X_all_unique, X_folds_list, file_list, NBINS, IQR_FACTOR, id2fold_rows)
        binned_mean = np.nanmean(binned_matrix, axis=0)
        binned_lower = np.nanpercentile(binned_matrix, 2.5, axis=0)
        binned_upper = np.nanpercentile(binned_matrix, 97.5, axis=0)
        smooth_mean = lowess(binned_mean, bin_centers, frac=0.3, return_sorted=True)
        smooth_lower = lowess(binned_lower, bin_centers, frac=0.3, return_sorted=True)
        smooth_upper = lowess(binned_upper, bin_centers, frac=0.3, return_sorted=True)
        ax_right.scatter(bin_centers, binned_mean, color='black', s=8, label='Mean SHAP')
        ax_right.plot(smooth_mean[:, 0], smooth_mean[:, 1], color='black', label='Mean line')
        ax_right.fill_between(smooth_mean[:,0], smooth_lower[:,1], smooth_upper[:,1], color='blue', alpha=0.2, label='95% CI')
        ax_right.axhline(y=0, color='red', linestyle='--')
        ax_right.set_xlabel(f"{brain_regions.get(feat_right, feat_right)} (Volume)", fontsize=10)
        # Similarly, lookup stats for right feature.
        row_right = df_stats[df_stats['feature_name'] == feat_right].iloc[0]
        ax_right.set_title(f"Rel {row_right['reliability']:.2f}, CIT {row_right['precision']:.2f}", fontsize=10)
        ax_right.legend(fontsize=8)
        plot_idx += 1

    # Remove extra subplots
    for j in range(plot_idx, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    plt.show()
    
def load_boot_feature_means(bootstrap_dir, num_iterations, num_features):
    """
    Load and aggregate SHAP values from a series of bootstrap files.
    
    Args:
      bootstrap_dir (str): Directory containing bootstrap files.
      num_iterations (int): Number of bootstrap iterations.
      num_features (int): Total number of features.
      
    Returns:
      boot_feature_means (np.ndarray): Array of shape (num_iterations, num_features)
        where each row represents the average SHAP values for that iteration.
    """
    boot_feature_means = np.empty((num_iterations, num_features))
    for i in range(num_iterations):
        filename = f'bootstrap_shap_iter_{i+1}.pkl'
        path = os.path.join(bootstrap_dir, filename)
        with open(path, 'rb') as f:
            shap_values_fold = pickle.load(f)
        # Each file contains a list of folds; compute mean over folds, per feature.
        fold_means = [np.mean(fold, axis=0) for fold in shap_values_fold]
        iteration_mean = np.mean(fold_means, axis=0)
        boot_feature_means[i, :] = iteration_mean
    return boot_feature_means

def compute_confidence_intervals(boot_feature_means):
    """
    Compute the 95% confidence intervals and overall means from bootstrap aggregated data.
    
    Args:
      boot_feature_means (np.ndarray): Array of shape (iterations, num_features)
    
    Returns:
      mean_values (np.ndarray): Mean SHAP values per feature.
      lower_ci (np.ndarray): 2.5th percentile for each feature.
      upper_ci (np.ndarray): 97.5th percentile for each feature.
    """
    lower_ci = np.percentile(boot_feature_means, 2.5, axis=0)
    upper_ci = np.percentile(boot_feature_means, 97.5, axis=0)
    mean_values = np.mean(boot_feature_means, axis=0)
    return mean_values, lower_ci, upper_ci

def get_feature_labels(feature_names, brain_regions):
    """
    Get feature labels based on the brain_regions dictionary.
    
    Args:
      feature_names (list): List of feature names.
      brain_regions (dict): Mapping from feature code to brain region string.
      
    Returns:
      feature_labels (list): List of labels corresponding to each feature.
    """
    return [brain_regions.get(fn, fn) for fn in feature_names]

def group_feature_indices(feature_labels):
    """
    Group feature indices into Ventricles, White Matter, Gray Matter, 
    and further into gray matter subgroups based on keywords.
    
    Returns:
      main_groups (dict): e.g., {"Ventricles": [...], "White Matter": [...]}
      gm_subgroups (dict): e.g., {"Subcortical": [...], "Frontal": [...], ... , "Other": [...]}
    """
    ventricles = [i for i, label in enumerate(feature_labels) if '-VN' in label]
    white_matter = [i for i, label in enumerate(feature_labels) if '-WM' in label]
    gray_matter = [i for i, label in enumerate(feature_labels) if '-GM' in label]
    
    gm_keywords = {
        "Subcortical": ["Accumbens", "Amygdala", "Caudate", "Hippocampus", "Pallidum", "Putamen", "Thalamus", "Basal Forebrain"],
        "Frontal": ["Frontal", "Cingulate", "Insula", "Operc", "Gyrus Rectus", "Precentral", "Postcentral"],
        "Temporal": ["Temporal", "Planum Temp"],
        "Parietal_Cerebellum": ["Parietal", "Precuneus", "Supramarginal", "Cerebellum"],
        "Occipital": ["Occipital", "Calcarine", "Cuneus", "Lingual", "Fusiform", "Occipital Pole"]
    }
    gm_subgroups = {key: [] for key in gm_keywords.keys()}
    gm_subgroups["Other"] = []
    
    for i in gray_matter:
        label = feature_labels[i]
        assigned = False
        for subgroup, keywords in gm_keywords.items():
            if any(keyword in label for keyword in keywords):
                gm_subgroups[subgroup].append(i)
                assigned = True
                break
        if not assigned:
            gm_subgroups["Other"].append(i)
    
    main_groups = {
        "Ventricles": ventricles,
        "White Matter": white_matter,
    }
    return main_groups, gm_subgroups

def plot_group_results(group_name, indices, mean_values, lower_ci, upper_ci, feature_labels, ylim=(-0.15, 0.4),
                       save_path=None):
    """
    Plot error bars for a given group of features.
    
    Args:
      group_name (str): Name of the group (e.g., "Ventricles").
      indices (list): Indices of features belonging to the group.
      mean_values, lower_ci, upper_ci (np.ndarray): Arrays with computed metrics.
      feature_labels (list): Labels for the features.
      ylim (tuple): y-axis limits.
    """
    group_mean = mean_values[indices]
    group_lower = lower_ci[indices]
    group_upper = upper_ci[indices]
    group_labels = [feature_labels[i][:-3] for i in indices]
    
    # Compute error bars
    error_lower = np.clip(group_mean - group_lower, 0, None)
    error_upper = np.clip(group_upper - group_mean, 0, None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(indices))
    ax.errorbar(x, group_mean, yerr=[error_lower, error_upper], fmt='o', capsize=5)
    #ax.set_ylim(*ylim)
    ax.axhline(y=0, color='red', linestyle='--')
    #ax.axhline(y=0.05, color='blue', linestyle='--', alpha=0.4)
    #ax.axhline(y=-0.05, color='blue', linestyle='--', alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=75, ha="right")
    #ax.set_xlabel(f"{group_name} Regions")
    ax.set_ylabel("Mean SHAP", fontsize=24)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title(f"{group_name}", fontsize=30)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()

def plot_subgroup_results(subgroup_name, indices, mean_values, lower_ci, upper_ci, feature_labels, ylim=(-0.15, 0.4),
                          save_path=None):
    """
    Similar to plot_group_results but for gray matter subgroups.
    """
    subgroup_mean = mean_values[indices]
    subgroup_lower = lower_ci[indices]
    subgroup_upper = upper_ci[indices]
    subgroup_labels = [feature_labels[i][:-3] for i in indices]
    
    error_lower = np.clip(subgroup_mean - subgroup_lower, 0, None)
    error_upper = np.clip(subgroup_upper - subgroup_mean, 0, None)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(indices))
    ax.errorbar(x, subgroup_mean, yerr=[error_lower, error_upper], fmt='o', capsize=5)
    #ax.set_ylim(*ylim)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(subgroup_labels, rotation=75, ha="right")
    #ax.set_xlabel(f"Gray Matter Regions: {subgroup_name}")
    ax.set_ylabel("Mean SHAP", fontsize=24)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_title(f"{subgroup_name} (Gray Matter)", fontsize=30)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()

def plot_partial_dependence_single_region(region_key, brain_regions, DATA_PKL_PATH, BOOTSTRAP_DIR,
                                          NBINS=200, IQR_FACTOR=1.2, save_path=None, n_jobs=8,
                                          batch_size=20, stored_data_dir=None):
    """Production-optimized function with mini-batch memory management and optional loading from .npz."""

    region_name = brain_regions.get(region_key, f"Region {region_key}")

    # If stored data exists, load and skip computation
    if stored_data_dir is not None:
        stored_data_path = os.path.join(stored_data_dir, f'{region_key}-{brain_regions[region_key]}.npz')
        data = np.load(stored_data_path, allow_pickle=True)
        bin_centers = data['bin_centers']
        binned_mean = data['binned_mean']
        binned_lower = data['binned_lower']
        binned_upper = data['binned_upper']
        smooth_mean = data['smooth_mean']
        smooth_lower = data['smooth_lower']
        smooth_upper = data['smooth_upper']
        reliability = data['reliability'].item()
        consistency = data['consistency'].item()
        print(f"Loaded precomputed data from {stored_data_path}")
    else:
        # Load unique data
        X_all_unique, id2fold_rows, X_folds_list, feature_names = load_dataset_unique(DATA_PKL_PATH)

        if region_key not in feature_names:
            raise ValueError(f"Region key '{region_key}' not found in feature names.")

        feat_index = feature_names.index(region_key)
        x_vals = X_all_unique[region_key].astype(np.float32).values
        order = np.argsort(x_vals)
        bins_indices = np.array_split(order, NBINS)
        bin_centers = np.array([np.mean(x_vals[indices]) for indices in bins_indices])

        unique_ids = X_all_unique.iloc[:, 0].astype(str).values
        sample_id_to_index = {sid: idx for idx, sid in enumerate(unique_ids)}

        # Precompute valid indices
        fold_valid = []
        for fold in range(len(X_folds_list)):
            values = X_folds_list[fold][region_key].astype(np.float32).values \
                if region_key in X_folds_list[fold].columns else X_folds_list[fold].iloc[:, 1 + feat_index].astype(np.float32).values
            valid_idx = set(get_valid_indices(values, factor=IQR_FACTOR))
            fold_valid.append(valid_idx)

        file_list = get_bootstrap_file_list(BOOTSTRAP_DIR)
        n_files = len(file_list)

        # --- Helper function ---
        def load_and_process(file_path):
            with open(file_path, 'rb') as f:
                shap_folds = pickle.load(f)
            shap_folds = [fold.to_numpy(dtype=np.float32) if hasattr(fold, "to_numpy") else np.array(fold, dtype=np.float32) for fold in shap_folds]

            agg_vector = np.full(x_vals.shape, np.nan, dtype=np.float32)
            for sample_id, occ_list in id2fold_rows.items():
                sample_vals = []
                for (fold, row_idx) in occ_list:
                    if row_idx in fold_valid[fold]:
                        val = shap_folds[fold][row_idx, feat_index]
                        sample_vals.append(val)
                if sample_vals:
                    agg_vector[sample_id_to_index[sample_id]] = np.mean(sample_vals)

            binned_mean = []
            for indices in bins_indices:
                values = agg_vector[indices]
                valid_values = values[~np.isnan(values)]
                binned_mean.append(np.mean(valid_values) if valid_values.size > 0 else np.nan)
            mean_abs_val = np.nanmean(np.abs(agg_vector))
            return binned_mean, mean_abs_val

        # --- Mini-batch processing ---
        binned_values = []
        mean_shap_vals = []

        for batch_start in tqdm(range(0, n_files, batch_size), desc=f"Processing {region_name} in batches"):
            batch_files = file_list[batch_start:batch_start + batch_size]
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(load_and_process)(file_path) for file_path in batch_files
            )
            batch_binned_values, batch_mean_vals = zip(*batch_results)
            binned_values.extend(batch_binned_values)
            mean_shap_vals.extend(batch_mean_vals)

        binned_values = np.array(binned_values, dtype=np.float32)
        mean_vals = np.array(mean_shap_vals, dtype=np.float32)

        # Reliability
        mean_val = np.mean(mean_vals)
        std_val = np.std(mean_vals, ddof=1)
        CV = std_val / (mean_val + 1e-8)
        reliability = max(0, (1 - min(CV, 1)))

        # Consistency
        sign_matrix = np.sign(binned_values)
        avg_sign = np.nanmean(sign_matrix, axis=0)
        consistency = np.mean((avg_sign > 0.7) | (avg_sign < -0.7))

        # Mean and CI
        binned_mean = np.nanmean(binned_values, axis=0)
        binned_lower = np.nanpercentile(binned_values, 2.5, axis=0)
        binned_upper = np.nanpercentile(binned_values, 97.5, axis=0)

        # Smoothing with LOWESS
        smooth_mean = lowess(binned_mean, bin_centers, frac=0.2, return_sorted=True)
        smooth_lower = lowess(binned_lower, bin_centers, frac=0.2, return_sorted=True)
        smooth_upper = lowess(binned_upper, bin_centers, frac=0.2, return_sorted=True)

        # Save computed data if save_path is provided
        if save_path:
            os.makedirs(os.path.join(save_path, 'plot'), exist_ok=True)
            os.makedirs(os.path.join(save_path, 'data'), exist_ok=True)
            np.savez(
                os.path.join(save_path, 'data', f'{region_key}-{brain_regions[region_key]}.npz'),
                bin_centers=bin_centers,
                binned_mean=binned_mean,
                binned_lower=binned_lower,
                binned_upper=binned_upper,
                smooth_mean=smooth_mean,
                smooth_lower=smooth_lower,
                smooth_upper=smooth_upper,
                reliability=reliability,
                consistency=consistency,
                mean_shap_vals=mean_shap_vals,
                binned_values=binned_values
            )
            print(f"Saved computed data to {save_path}")

    # --- Plotting ---
    if len(region_name) >= 3:
        region_label = region_name[:-3]
    else:
        region_label = region_name
    
    fig, ax = plt.subplots(figsize=(11.5, 13.5))
    ax.scatter(bin_centers, binned_mean, color='black', s=100, label='Mean SHAP')
    ax.plot(smooth_mean[:, 0], smooth_mean[:, 1], linewidth=4.5, color='black', label='Smoothed Mean')
    ax.fill_between(smooth_mean[:, 0], smooth_lower[:, 1], smooth_upper[:, 1], color='blue', alpha=0.2, label='95% CI')
    ax.axhline(y=0, color='red', linewidth=3, linestyle='--')
    ax.set_ylim(-1, 2)
    ax.set_xlabel(f"{region_label}\n (Volume)", fontsize=44, fontweight='bold')
    ax.tick_params(axis='both', labelsize=40)
    ax.set_title(f"Reliability : {reliability:.2f} | Consistency: {consistency:.2f}", fontsize=44)
    ax.legend(fontsize=40)
    #ax.set_position([0.15, 0.15, 0.7, 0.7])
    plt.tight_layout()

    if save_path:
        plt.savefig(os.path.join(save_path, 'plot', f'{region_key}-{brain_regions[region_key]}.png'), dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"Saved plot to {save_path}")

def custom_normalize(nifti_img, min_val, max_val, scale_val=1):
    """Normalize image data to the range (-1, 1) and apply optional scaling."""
    data = nifti_img.get_fdata()
    scale = 2.0 / (max_val - min_val)   # Map difference to 2 (i.e., -1 to 1)
    offset = -1 - min_val * scale
    norm_data = data * scale + offset
    return nib.Nifti1Image(scale_val * norm_data, affine=nifti_img.affine)

def normalize_image(nifti_img, scale_val=1, max_val=None):
    """Normalize image data using custom normalization and return the normalized image and max absolute value."""
    if max_val is None:
        data = nifti_img.get_fdata()
        max_val = np.abs(data).max()
    norm_img = custom_normalize(nifti_img, min_val=-max_val, max_val=max_val, scale_val=scale_val)
    return norm_img, max_val

def cast_to_float32(nifti_img):
    """Convert image data to float32 to avoid precision warnings in ITK-SNAP."""
    data = nifti_img.get_fdata().astype(np.float32)
    return nib.Nifti1Image(data, affine=nifti_img.affine)

def mask_positive(nifti_img, reverse=False):
    """Mask out non-positive values (or reverse sign if needed)."""
    data = nifti_img.get_fdata().copy()
    if reverse:
        data = -data
    data[data <= 0] = 0
    return nib.Nifti1Image(data, affine=nifti_img.affine)

def convert_to_template(shap_values, rois_selected, feat_columns, muse_dir, df_muse):
    """
    Create a brain map from SHAP values by mapping each selected ROI to a SHAP value.
    
    Args:
      shap_values (np.ndarray): 1D array of SHAP values (one per feature).
      rois_selected (list): List of ROI IDs (as strings) for which to create the map.
      feat_columns (list): List of feature names (ROI IDs as strings).
      muse_dir (str): Directory containing MUSE ROI images.
      df_muse (pd.DataFrame): DataFrame with MUSE levels; must include 'ROI_ID' and 'ROI_level'.
      
    Returns:
      q_shap_nifti: Nifti1Image containing the SHAP map.
    """
    # Load base image for shape and affine (using MUSE level 0)
    base_img = nib.load(os.path.join(muse_dir, 'MUSE_level_0.nii.gz'))
    template_flat = base_img.get_fdata().flatten()
    aff = base_img.affine
    shap_map_flat = np.zeros(template_flat.shape)
    
    df_shap = pd.DataFrame(shap_values[None, :], columns=feat_columns)
    
    for roi in rois_selected:
        roi_int = int(roi)
        if roi_int in df_muse['ROI_ID'].values:
            level = df_muse.loc[df_muse['ROI_ID'] == roi_int, 'ROI_level'].values[0]
            roi_img = nib.load(os.path.join(muse_dir, f'MUSE_level_{level}.nii.gz'))
            roi_flat = roi_img.get_fdata().flatten()
            ind = np.where(roi_flat == roi_int)
            shap_map_flat[ind] = df_shap[str(roi)].values[0]
    
    # Reshape (assuming dimensions 182x218x182)
    q_shap = shap_map_flat.reshape(182, 218, 182)
    return nib.Nifti1Image(q_shap, affine=aff)

def visualize_roi_map(nifti_img, template, vmin=None, vmax=None, coords=None, cmap=None, title=None, save_path=None):
    """Display an overlay of the SHAP map on the template using nilearn."""
    fig, ax = plt.subplots(figsize=(30, 3))
    plotting.plot_stat_map(
        nifti_img,
        display_mode='z',
        cmap=cmap,
        annotate=False,
        axes=ax,
        vmin=vmin,
        vmax=vmax,
        bg_img=template,
        black_bg=False,
        cut_coords=coords,
        draw_cross=False,
        alpha=1
    )
    if title:
        ax.set_title(title, size=20, bbox=dict(facecolor='none', edgecolor='none'))
    if save_path:
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
    plt.show()
    
# --- Prepare Labels and Valid Indices --- 

def get_valid_folds_and_labels(nhw, nha, hwa, y_label='DX', y_pred_label='PRED_DX'): 
    y_folds, y_pred_folds, valid_folds = [], [], [] 
    feature_names = list(nhw[0]['data'].iloc[:, 1:].columns) 
    for i in range(10): 
        X_data = pd.concat([nhw[i]['data'], nha[i]['data'], hwa[i]['data']], axis=0) 
        valid_indices = [get_valid_indices(X_data.iloc[:, j].values, factor=1.2) for j in range(1, X_data.shape[1])] 
        y_data = pd.concat([nhw[i]['y'][y_label], nha[i]['y'][y_label], hwa[i]['y'][y_label]], axis=0) 
        y_pred_data = pd.concat([nhw[i]['y'][y_pred_label], nha[i]['y'][y_pred_label], hwa[i]['y'][y_pred_label]], axis=0) 
        y_folds.append(y_data) 
        y_pred_folds.append(y_pred_data) 
        valid_folds.append(valid_indices) 
    return y_folds, y_pred_folds, valid_folds, feature_names 

def compute_bootstrap_iteration(i, bootstrap_dir, valid_folds, y_folds, y_pred_folds, feature_names, desired_class=1, reverse=False): 
    filename = f'bootstrap_shap_iter_{i+1}.pkl' 
    with open(os.path.join(bootstrap_dir, filename), 'rb') as f: 
        shap_values_fold = pickle.load(f) 
    fold_means = [] 
    for j, fold in enumerate(shap_values_fold): 
        feature_means = [] 
        for k in range(len(feature_names)): 
            valid_idx = valid_folds[j][k] 
            y_vals = y_folds[j].iloc[valid_idx].values 
            y_preds = (y_pred_folds[j].iloc[valid_idx].values > 0.5).astype(int) 
            if not reverse: 
                mask = (y_vals == desired_class) & (y_preds == desired_class) 
            else: 
                mask = (y_vals == desired_class) & (y_preds == np.abs(1 - desired_class)) 
            filtered_idx = np.array(valid_idx)[mask] 
            if len(filtered_idx) > 0:
                mean_val = np.mean(fold.iloc[filtered_idx, k])
            else:
                mean_val = np.mean(fold.iloc[valid_idx, k]) 
            feature_means.append(mean_val) 
        fold_means.append(feature_means) 
    return np.mean(fold_means, axis=0) 

def simple_compute_bootstrap_iteration(i, bootstrap_dir, valid_folds, feature_names): 
    """ 
    Process one bootstrap iteration to compute average SHAP values across folds and features. 
    Args: 
        i (int): Iteration index. 
        bootstrap_dir (str): Directory containing bootstrap SHAP files. 
        valid_folds (list): List of valid indices for each fold and feature. 
        feature_names (list): List of feature names. 

    Returns: 
        np.ndarray: Mean SHAP values per feature for this iteration. 
    """ 
    filename = f'bootstrap_shap_iter_{i+1}.pkl' 
    with open(os.path.join(bootstrap_dir, filename), 'rb') as f: 
        shap_values_fold = pickle.load(f) 

    fold_means = [] 
    for j, fold in enumerate(shap_values_fold): 
        feature_means = [] 
        for k in range(len(feature_names)): 
            valid_idx = valid_folds[j][k] 
            if len(valid_idx) > 0: 
                mean_val = np.mean(fold.iloc[valid_idx, k]) 
            else: 
                mean_val = np.nan  # fallback if no valid idx 
            feature_means.append(mean_val) 
        fold_means.append(feature_means) 
    return np.nanmean(fold_means, axis=0)

def create_binary_map(img_path, output_path):
    # Load image
    img = nib.load(img_path)
    data = img.get_fdata()

    # Find minimum nonzero value
    nonzero_values = data[data > 0]
    min_nonzero = nonzero_values.min()
    print(f"Minimum nonzero value: {min_nonzero}")

    # Create binary map
    binary_map = (data >= min_nonzero).astype(np.uint8)

    # Save binary map as new NIfTI
    binary_img = nib.Nifti1Image(binary_map, img.affine, img.header)
    nib.save(binary_img, output_path)
    print(f"Binary map saved to {output_path}")

def coregister_to_template(moving_img_path, template_img_path, output_path, is_binary=False):
    # Load moving image and template
    moving_img = nib.load(moving_img_path)
    template_img = nib.load(template_img_path)

    # Choose interpolation order
    order = 0 if is_binary else 1  # 0 = nearest neighbor for binary; 1 = linear for continuous

    # Resample moving image to match template
    resampled_img = resample_from_to(moving_img, template_img, order=order)

    if is_binary:
        # Post-process to ensure strictly binary values (0 or 1)
        resampled_data = resampled_img.get_fdata()
        binary_data = (resampled_data >= 0.5).astype(np.uint8)
        resampled_img = nib.Nifti1Image(binary_data, resampled_img.affine, resampled_img.header)

    # Save the co-registered image
    nib.save(resampled_img, output_path)
    print(f"Co-registered image saved to {output_path}")

def load_folds_from_pickle(pkl_path):
    with open(pkl_path, "rb") as f:
        train, nhw, nha, hwa, est = pickle.load(f)

    feature_names = list(nhw[0]['data'].iloc[:, 1:].columns)
    y_folds, y_pred_folds, valid_folds = [], [], []

    for i in range(10):
        X = pd.concat([nhw[i]['data'], nha[i]['data'], hwa[i]['data']], axis=0)
        val_idx = [
            get_valid_indices(X.iloc[:, j].values, factor=1.2)
            for j in range(1, X.shape[1])
        ]
        y_true = pd.concat([
            nhw[i]['y']['DX'], nha[i]['y']['DX'], hwa[i]['y']['DX']
        ], axis=0).values
        y_pred = pd.concat([
            nhw[i]['y']['PRED_DX'], nha[i]['y']['PRED_DX'], hwa[i]['y']['PRED_DX']
        ], axis=0).values > 0.5

        y_folds.append(y_true)
        y_pred_folds.append(y_pred.astype(int))
        valid_folds.append(val_idx)

    return feature_names, y_folds, y_pred_folds, valid_folds

def load_folds_from_csv(scenario, csv_model_dir):
    from scipy.special import expit
    data_dir = "/home/Codes/ad_classification/data/split_fold_w_augmented"
    example_csv = os.path.join(data_dir, f"scenario_{scenario}_fold_1_test.csv")
    df_example = pd.read_csv(example_csv)
    feature_names = list(df_example.columns[8:])

    valid_folds, y_folds, y_pred_folds = [], [], []

    for i in range(10):
        test_csv = os.path.join(data_dir, f"scenario_{scenario}_fold_{i+1}_test.csv")
        df_test = pd.read_csv(test_csv)

        pred_csv = os.path.join(csv_model_dir, f"scenario_{scenario}_fold_{i+1}_predict.csv")
        df_pred = pd.read_csv(pred_csv)

        # Align prediction with test by ID
        df_test = df_test.set_index("ID")
        df_pred = df_pred.set_index("ID")
        df_pred = df_pred.loc[df_test.index]  # reorder to match

        X_data = df_test[feature_names].values
        valid_indices = [get_valid_indices(X_data[:, j], factor=1.2) for j in range(X_data.shape[1])]
        valid_folds.append(valid_indices)

        y_folds.append(df_pred["NACCUDSD"].values)
        y_pred_folds.append((expit(df_pred["PREDICTION"].values) > 0.5).astype(int))

    return feature_names, y_folds, y_pred_folds, valid_folds