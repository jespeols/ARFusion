import numpy as np
import pandas as pd
import pickle
import yaml
import torch
import os
import matplotlib.pyplot as plt

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

with open(os.path.join(BASE_DIR, 'config_MM.yaml'), 'r') as f:
    config = yaml.safe_load(f)


def get_split_indices(size_to_split, val_share, random_state: int = 42):
    indices = np.arange(size_to_split)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    train_share = 1 - val_share
    
    train_size = int(train_share * size_to_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    return train_indices, val_indices


def get_multimodal_split_indices(sizes: list[int], val_share, random_state:int=42):
    train_indices = []
    val_indices = []
    np.random.seed(random_state)
    for size in sizes:
        assert isinstance(size, int), "sizes must be a list of integers"
        indices = np.arange(size)
        np.random.shuffle(indices)
        train_size = int((1 - val_share) * size)
        train_indices.append(indices[:train_size])
        val_indices.append(indices[train_size:])
    
    return train_indices, val_indices
    


def filter_gene_counts(df, threshold_num):
    # get indices of samples with more than threshold_num genotypes
    indices = df[df['num_genotypes'] > threshold_num].index
    num_above = len(indices)
    # drop samples with more than threshold_num genotypes
    df.drop(indices, inplace=True)
    print(f"Dropping {num_above} isolates with more than {threshold_num} genotypes")
    return df


def impute_col(df, col, random_state=42):
    print(f"Imputing column {col} from the distribution of non-NaN values")
    indices = df[df[col].isnull()].index
    np.random.seed(random_state)
    sample = np.random.choice(df[col].dropna(), size=len(indices))
    df.loc[indices, col] = sample
    
    return df


def export_results(results, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {savepath}")

############################# Results analysis #############################

def get_average_and_std_df(results_dict, with_metric_as_index=False):
    losses = results_dict['losses']
    accs = results_dict['accs']
    iso_accs = results_dict['iso_accs']
    sensitivities = results_dict['sensitivities']
    specificities = results_dict['specificities']
    f1_scores = results_dict['F1_scores']
    
    losses_avg = np.mean(losses)
    losses_std = np.std(losses)
    accs_avg = np.mean(accs)
    accs_std = np.std(accs)
    iso_accs_avg = np.mean(iso_accs)
    iso_accs_std = np.std(iso_accs)
    sens_avg = np.mean(sensitivities)
    sens_std = np.std(sensitivities)
    spec_avg = np.mean(specificities)
    spec_std = np.std(specificities)
    f1_avg = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    
    df_CV = pd.DataFrame(data={
        "metric": ["Loss", 'Accuracy', 'Isolate accuracy', 'Sensitivity', 'Specificity', 'F1'], 
        "avg": [losses_avg, accs_avg, iso_accs_avg, sens_avg, spec_avg, f1_avg], 
        "std": [losses_std, accs_std, iso_accs_std, sens_std, spec_std, f1_std]
    })
    if with_metric_as_index:
        df_CV.set_index("metric", inplace=True)
    return df_CV


def get_ab_stats_df(results_dict, with_ab_as_index=False):
    ab_stats_list = results_dict['ab_stats']
    
    data_dict = {}
    antibiotics = ab_stats_list[0]['antibiotic'].tolist()
    data_dict.update({"antibiotic": antibiotics})
    for ab_stats in ab_stats_list:
        ab_stats['S_share'] = ab_stats['num_masked_S'] / ab_stats['num_masked_tot']
        ab_stats['R_share'] = ab_stats['num_masked_R'] / ab_stats['num_masked_tot']

    num_tot = np.array([ab_stats['num_masked_tot'].tolist() for ab_stats in ab_stats_list])
    avg_num = np.mean(num_tot, axis=0).astype(int).tolist()
    std_num = np.std(num_tot, axis=0).tolist()
    s_shares = np.array([ab_stats['S_share'].tolist() for ab_stats in ab_stats_list])
    r_shares = np.array([ab_stats['R_share'].tolist() for ab_stats in ab_stats_list])
    s_share_median = np.median(s_shares, axis=0).tolist()
    s_share_std = np.std(s_shares, axis=0).tolist()
    r_share_median = np.median(r_shares, axis=0).tolist()
    r_share_std = np.std(r_shares, axis=0).tolist()
    data_dict.update({
        "avg_num": avg_num, "std_num":std_num,
        "S_share_median": s_share_median, "R_share_median": r_share_median,
        "S_share_std":s_share_std, "R_share_std": r_share_std
    })
    
    metrics = ['accuracy', 'sensitivity', 'specificity', "precision", 'F1']
    for metric in metrics:
        arr = np.array([ab_stats[metric] for ab_stats in ab_stats_list])
        avg = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        data_dict.update({metric+"_avg": avg.tolist(), metric+"_std": std.tolist()})

    df_ab_CV = pd.DataFrame(data=data_dict)
    if with_ab_as_index:
        df_ab_CV.set_index("antibiotic", inplace=True)
    return df_ab_CV


def plot_metric_by_ab(
    df_CV_ab,
    metric,
    use_std = True,
    sort_by_desc: str = 'metric',
    sort_by_desc_S_share = False,
    use_legend = True,
    legend_labels = None,
    legend_loc = None,
    colors = ['slategray', 'forestgreen', 'darkgreen', 'gold', 'darkgoldenrod', 'red', 'darkred'],
    title = None, 
    figsize = (12, 8), 
    bar_width = None,
    save_path = None
):
    _, ax = plt.subplots(figsize=figsize)
    if sort_by_desc == 'metric':
        df_CV_ab = df_CV_ab.sort_values(by=(metric+'_avg', 'No PT'), ascending=False)
    elif sort_by_desc == 'S_share':
        df_CV_ab = df_CV_ab.sort_values(by='S_share_median', ascending=False)
    elif sort_by_desc == 'R_share':
        df_CV_ab = df_CV_ab.sort_values(by='R_share_median', ascending=False)
    else:
        raise ValueError('sort_by_desc must be either "metric" or "S_share"')
    
    if use_std:
        if bar_width:
            df_CV_ab.plot(kind='bar', y=metric+'_avg', yerr=metric+'_std', rot=0,
                        capsize=2, ecolor='k', color=colors, width=bar_width, ax=ax, legend=False)
        else:
            df_CV_ab.plot(kind='bar', y=metric+'_avg', yerr=metric+'_std', rot=0,
                        capsize=2, ecolor='k', color=colors, ax=ax, legend=False)
    else:
        if bar_width:
            df_CV_ab.plot(kind='bar', y=metric+'_avg', rot=0, color=colors, width=bar_width, ax=ax, legend=False)
        else:
            df_CV_ab.plot(kind='bar', y=metric+'_avg', rot=0, color=colors, ax=ax, legend=False)
    if title:
        if title == 'none':
            pass
        else:
            ax.set_title(title) 
    else:
        ax.set_title(f'{metric} by antibiotic')
    if sort_by_desc == 'metric':
        ax.set_xlabel(f'Antibiotic (desc. {metric})')
    else:
        ax.set_xlabel(f'Antibiotic (desc. {sort_by_desc})')
    ax.set_ylabel(metric)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    model_names = df_CV_ab[metric+'_avg'].columns.tolist()
    if use_legend:
        ax.legend(legend_labels if legend_labels else model_names, loc=legend_loc, framealpha=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()


def plot_metric_by_ab_with_distr(
    df_CV_ab,
    metric,
    use_std = True,
    sort_by_desc_S_share = True,
    show_distr_std = False,
    legend_labels = None,
    legend_loc = None,
    colors = ['slategray', 'forestgreen', 'darkgreen', 'gold', 'darkgoldenrod', 'red', 'darkred'],
    title = None, 
    figsize = (12, 8), 
    save_path = None
):
    model_names = df_CV_ab[metric+'_avg'].columns.tolist()
    _, ax = plt.subplots(figsize=figsize)
    if sort_by_desc_S_share:
        df_CV_ab = df_CV_ab.sort_values(by='S_share_median', ascending=False)
    ind = np.arange(len(df_CV_ab.index))
    width = 0.15
    if use_std:
        for i, model in enumerate(model_names):
            ax.bar(
                ind + (i-1)*width, df_CV_ab[metric+'_avg'][model], width, yerr=df_CV_ab[metric+'_std'][model], 
                label=model, color=colors[i], capsize=2, ecolor='k'
            )
    else:
        for i, model in enumerate(model_names):
            ax.bar(ind + (i-1)*width, df_CV_ab[metric+'_avg'][model], width, label=model, color=colors[i])
    if show_distr_std:
        ax.bar(ind + 3*width, df_CV_ab['S_share_median'], width, color='darkgreen', label='S share', yerr=df_CV_ab['S_share_std'], capsize=2, ecolor='k')
        ax.bar(ind + 3*width, df_CV_ab['R_share_median'], width, bottom=df_CV_ab['S_share_median'], color='darkred', label='R share', yerr=df_CV_ab['R_share_std'], capsize=2, ecolor='k')
    else:
        ax.bar(ind + 3*width, df_CV_ab['R_share_median'], width, bottom=df_CV_ab['S_share_median'], color='darkred', label='R share')
        ax.bar(ind + 3*width, df_CV_ab['S_share_median'], width, color='darkgreen', label='S share')
    ax.set_xticks(ind + width, df_CV_ab.index)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    if title:
        if title == 'none':
            pass
        else:
            ax.set_title(title) 
    else:
        ax.set_title(f'{metric} by antibiotic')
    ax.set_xlabel('Antibiotic')
    ax.set_ylabel(metric)
    ax.legend(legend_labels if legend_labels else model_names, loc=legend_loc, framealpha=0)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def load_and_create_ab_df(
    train_params: str,
    exp_folder: str = None,
    model_names = ['No PT', 'Easy RPT', 'Easy CPT', 'Medium RPT', 'Medium CPT', 'Hard RPT', 'Hard CPT'],
    train_share: str = None,
):
    results_dict_list = []
    for model_name in model_names:
        model_name = model_name.replace(' ', '')
        if train_share:
            s = f"FT_{model_name}_{train_params}_train_share{train_share}"
        else:
            s = f"FT_{model_name}_{train_params}"
        if exp_folder:
            p = os.path.join(BASE_DIR, 'results', 'MM', exp_folder, s, 'CV_results.pkl')
        else:
            p = os.path.join(BASE_DIR, 'results', 'MM', s, 'CV_results.pkl')
        results_dict_list.append(pd.read_pickle(p))
        
    df_CV_ab_list = [get_ab_stats_df(results_dict) for results_dict in results_dict_list]
    df_CV_ab = pd.concat(df_CV_ab_list, keys=model_names, names=['model']).reset_index(level=1, drop=True).set_index('antibiotic', append=True).unstack(level=0)
    df_CV_ab = df_CV_ab.reindex(columns=model_names, level=1)
    reduce_cols = ['avg_num', 'std_num', 'S_share_median', 'R_share_median', 'S_share_std', 'R_share_std']
    df_CV_ab_tmp = df_CV_ab.drop(reduce_cols, axis=1)
    for i, col in enumerate(reduce_cols):
        df_CV_ab_tmp.insert(i, col, df_CV_ab[col].agg('mean', axis=1))
    class_map = config['data']['antibiotics']['abbr_to_class']
    df_CV_ab_tmp.insert(0, 'ab_class', df_CV_ab_tmp.index.map(class_map))
    return df_CV_ab_tmp


def load_and_create_abs_and_rel_diff_dfs(
    train_params: str,
    train_share: str = None,
    exp_folder: str = None,
    model_names = ['No PT', 'Easy RPT', 'Easy CPT', 'Medium RPT', 'Medium CPT', 'Hard RPT', 'Hard CPT'], 
):
    results_dict_list = []
    if train_share:
        for model_name in model_names:
            model_name = model_name.replace(' ', '')
            if exp_folder:
                p = os.path.join(BASE_DIR, 'results', 'MM', exp_folder, f'FT_{model_name}_{train_params}_train_share{train_share}', 'CV_results.pkl')
            else:
                p = os.path.join(BASE_DIR, 'results', 'MM', f'FT_{model_name}_{train_params}_train_share{train_share}', 'CV_results.pkl')
            results_dict_list.append(pd.read_pickle(p))
    else:
        for model_name in model_names:
            model_name = model_name.replace(' ', '')
            if exp_folder:
                p = os.path.join(BASE_DIR, 'results', 'MM', exp_folder, f'FT_{model_name}_{train_params}', 'CV_results.pkl')
            else:
                p = os.path.join(BASE_DIR, 'results', 'MM', f'FT_{model_name}_{train_params}', 'CV_results.pkl')
            results_dict_list.append(pd.read_pickle(p))
            
    df_CV_list = [get_average_and_std_df(results_dict) for results_dict in results_dict_list]
    df_CV = pd.concat(df_CV_list, keys=model_names, names=['model']).reset_index(level=1, drop=True).set_index('metric', append=True).unstack(level=0)
    df_CV = df_CV.reindex(columns=model_names, level=1)
    df_diff = df_CV.drop(('avg', 'No PT'), axis=1).drop(('std', 'No PT'), axis=1)
    for i in range(df_diff.shape[0]):
        df_diff.iloc[i, :].loc['avg'] = df_diff.iloc[i, :] - df_CV.loc[:, ('avg', 'No PT')].values[i]
        num_folds = 5 # sample size
        var_1 = df_diff.iloc[i, :].loc['std'].values**2
        var_2 = df_CV.loc[:, ('std', 'No PT')].values[i]**2
        df_diff.iloc[i, :].loc['std'] = np.sqrt((var_1 + var_2)/num_folds) # standard error of the difference of means
    return df_CV, df_diff


def load_and_create_train_share_df(
    model_prefix: str,
    train_params: str,
    exp_folder: str = None,
    train_shares = [0.01, 0.05, 0.1, 0.2, 0.3],
):
    model_names = [f'{model_prefix}_{share}' for share in train_shares]
    results_dict_list = []
    for share in train_shares:
        if not share == 0.8:
            s = f"FT_{model_prefix.replace(' ', '')}_{train_params}_train_share{share}"
        else:
            s = f"FT_{model_prefix.replace(' ', '')}_{train_params}"
        if exp_folder:
            p = os.path.join(BASE_DIR, 'results', 'MM', exp_folder, s, 'CV_results.pkl')
        else:
            p = os.path.join(BASE_DIR, 'results', 'MM', s, 'CV_results.pkl')
        results_dict_list.append(pd.read_pickle(p))
    df_CV_list = [get_average_and_std_df(results_dict) for results_dict in results_dict_list]
    df_CV = pd.concat(df_CV_list, keys=model_names, names=['model']).reset_index(level=1, drop=True).set_index('metric', append=True).unstack(level=0)
    df_CV = df_CV.reindex(columns=model_names, level=1)
    return df_CV 


def plot_metric_vs_train_shares(
    train_shares, 
    df_CV_list,
    metric,
    model_names = ['No PT', 'Easy RPT', 'Easy CPT', 'Medium RPT', 'Medium CPT', 'Hard RPT', 'Hard CPT'],
    colors = ['slategray', 'forestgreen', 'darkgreen', 'gold', 'darkgoldenrod', 'red', 'darkred'],
    legend_loc = None,
    plot_title = None,
    figsize = (12, 8),
    save_path = None
):
    _, ax = plt.subplots(figsize=figsize)
    j = 0
    for i, df_CV in enumerate(df_CV_list):
        if i == 0:
            ax.errorbar(train_shares, df_CV.loc[metric, 'avg'], yerr=df_CV.loc[metric, 'std'], 
                        label=model_names[i], color=colors[i], ecolor='k', capsize=2, fmt='o')
        elif i % 2 != 0: # odd
            j += 0.5
            ax.errorbar([s+np.ceil(j)*0.4 for s in train_shares], df_CV.loc[metric, 'avg'], yerr=df_CV.loc[metric, 'std'],
                        label=model_names[i], color=colors[i], ecolor='k', capsize=2, fmt='o')
        else: # even
            j += 0.5
            ax.errorbar([s-np.ceil(j)*0.4 for s in train_shares], df_CV.loc[metric, 'avg'], yerr=df_CV.loc[metric, 'std'],
                        label=model_names[i], color=colors[i], ecolor='k', capsize=2, fmt='o')
    ax.set_xlabel('Train share [% of total data]')
    ax.set_ylabel(metric)
    ax.set_xticks(train_shares)
    if legend_loc:
        ax.legend(loc=legend_loc, framealpha=0)
    else:
        ax.legend(framealpha=0)
    if plot_title:
        ax.set_title(plot_title)
    if save_path:
        print(f'Saving plot to {save_path}')
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=True)
    plt.show()