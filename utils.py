import numpy as np
import pandas as pd
import pickle
import yaml
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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


############################ Loss functions ############################

from torch.nn import functional as F

class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, alpha:float, reduction='mean'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target):
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BinaryFocalWithLogitsLoss(torch.nn.Module):
    def __init__(self, alpha:float, gamma:float, reduction='mean'):
        super(BinaryFocalWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        
    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = BCE_loss * (1-pt)**self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1-self.alpha) * (1-target)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss       

    
############################# Data processing #############################

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


def get_genotype_to_ab_class(unique_genotypes):
    genotype_ref = pd.read_csv(os.path.join(BASE_DIR, config['data']['NCBI']['refgenes_path']), sep='\t')
    genotype_ref = genotype_ref[(genotype_ref['Scope'] == 'core') & (genotype_ref['Type'] == 'AMR')].reset_index(drop=True)
    pm_ref = genotype_ref[genotype_ref['Subtype'] == 'POINT']
    gene_ref = genotype_ref[genotype_ref['Subtype'] != 'POINT']
    pm_to_ab_class = pm_ref.set_index('#Allele')['Class'].to_dict()
    gene_to_ab_class = gene_ref.set_index('Gene family')['Class'].to_dict()
    gene_allele_to_ab_class = gene_ref[gene_ref['#Allele'].notna()].set_index('#Allele')['Class'].to_dict()
    
    genotype_to_ab_class = {}
    for g in unique_genotypes:
        if "aac(3)-II" in g: # aac(3)-II does not appear exactly in either the Allele or Gene family columns
                    genotype_to_ab_class[g] = 'AMINOGLYCOSIDE'
        else:   
            if g.endswith('=POINT'):
                genotype_to_ab_class[g] = pm_to_ab_class[g.split('=')[0]]
            elif ('=PARTIAL' in g) or ('=HMM' in g) or ('=MISTRANSLATION' in g):
                genotype_to_ab_class[g] = gene_to_ab_class[g.split('=')[0]]
            else:
                try:
                    genotype_to_ab_class[g] = gene_to_ab_class[g]
                except:
                    genotype_to_ab_class[g] = gene_allele_to_ab_class[g]
    return genotype_to_ab_class


country_code_to_name = {
    'AD': 'Andorra', 'AL': 'Albania', 'AM': 'Armenia', 'AT': 'Austria', 'AZ': 'Azerbaijan', 'BA': 'Bosnia and Herzegovina',
    'BE': 'Belgium', 'BG': 'Bulgaria', 'BY': 'Belarus', 'CH': 'Switzerland', 'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany',
    'DK': 'Denmark', 'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'GE': 'Georgia',
    'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IL': 'Israel', 'IS': 'Iceland', 'IT': 'Italy', 'KG': 'Kyrgyzstan',
    'KZ': 'Kazakhstan', 'LI': 'Liechtenstein', 'LT': 'Lithuania', 'LU': 'Luxembourg', 'LV': 'Latvia', 'MC': 'Monaco',
    'MD': 'Moldova', 'ME': 'Montenegro', 'MK': 'Republic of North Macedonia', 'MT': 'Malta', 'NL': 'Netherlands',
    'NO': 'Norway', 'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'RU': 'Russia', 'SE': 'Sweden',
    'SI': 'Slovenia', 'SK': 'Slovakia', 'SM': 'San Marino', 'TJ': 'Tajikistan', 'TM': 'Turkmenistan', 'TR': 'Türkiye',
    'UA': 'Ukraine', 'UK': 'UK', 'UZ': 'Uzbekistan', 'XK': 'Kosovo'
}

############################# Results analysis #############################

def export_results(results, savepath):
    with open(savepath, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {savepath}")

def get_average_and_std_df(results_dict, include_auc=False):   
    metrics = {'loss': 'losses', 'accuracy': 'accs', 'isolate accuracy': 'iso_accs', 
               'sensitivity': 'sensitivities', 'specificity': 'specificities', 'F1': 'F1_scores'}
    if include_auc:
        metrics.update({'auc_score': 'auc_scores'})
    data_dict = {}
    for metric, key in metrics.items():
        list_ = results_dict[key]
        avg = np.mean(list_)
        std = np.std(list_)
        data_dict.update({metric: [avg, std]})
    df_CV = pd.DataFrame.from_dict(data_dict, orient='index', columns=['avg', 'std'])
    df_CV.index.name = 'metric'
    return df_CV


def get_ab_stats_df(results_dict, with_ab_as_index=False, include_auc=False):
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
    if include_auc:
        metrics.append('auc_score')
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
    sort_by_desc: str = 'S_share',
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
    include_auc = True,
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
        
    df_CV_ab_list = [get_ab_stats_df(results_dict, include_auc=include_auc) for results_dict in results_dict_list]
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
    ref_model = 'No PT',  ## default could be model_names[0],
    include_auc = True,
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
            
    df_CV_list = [get_average_and_std_df(results_dict, include_auc=include_auc) for results_dict in results_dict_list]
    df_CV = pd.concat(df_CV_list, keys=model_names, names=['model']).unstack(level=0)
    df_diff = df_CV.drop(('avg', ref_model), axis=1).drop(('std', ref_model), axis=1)
    for i in range(df_diff.shape[0]):
        df_diff.iloc[i, :].loc['avg'] = df_diff.iloc[i, :] - df_CV.loc[:, ('avg', ref_model)].values[i]
        num_folds = 5 # sample size
        var_1 = df_diff.iloc[i, :].loc['std'].values**2
        var_2 = df_CV.loc[:, ('std', ref_model)].values[i]**2
        df_diff.iloc[i, :].loc['std'] = np.sqrt((var_1 + var_2)/num_folds) # standard error of the difference of means
    return df_CV, df_diff


def load_and_create_train_share_df(
    model_prefix: str,
    train_params: str,
    exp_folder: str = None,
    train_shares = [0.01, 0.05, 0.1, 0.2, 0.3],
    include_auc = False,
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
    df_CV_list = [get_average_and_std_df(results_dict, include_auc=include_auc) for results_dict in results_dict_list]
    df_CV = pd.concat(df_CV_list, keys=model_names, names=['model']).unstack(level=0)
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
    
    
def calculate_ab_level_differences(
    df_ab_0, 
    df_ab_1, 
    metrics=['sensitivity', 'specificity', 'precision', 'F1'], 
    num_folds = 5,
):
    df_diff = df_ab_1.copy()
    for metric in metrics:
        avg = metric+'_avg'
        std = metric+'_std'
        df_diff[avg] = df_ab_1[avg] - df_ab_0[avg]
        df_diff[std] = np.sqrt((df_ab_0[std]**2 + df_ab_1[std]**2)/num_folds)
    return df_diff  

def plot_ab_level_differences(
    df_diff_ab,
    plot_metric,
    figsize=(13, 6),
    plot_title=None,
    colors=['slategray', 'forestgreen', 'darkgreen', 'gold', 'darkgoldenrod', 'red', 'darkred'],
    ab_classes=None,
    savepath=None
):
    if ab_classes:
        df_diff_plot = df_diff_ab[df_diff_ab['ab_class'].isin(ab_classes)]
    else:
        df_diff_plot = df_diff_ab
    _, ax = plt.subplots(figsize=figsize)
    df_diff_plot.plot(kind='bar', y=plot_metric+'_avg', yerr=plot_metric+'_std', ax=ax, rot=0, capsize=2, color=colors)
    if ab_classes:
        ax.set_xlabel('Antibiotics in classes: ' + ', '.join(ab_classes))
    else:
        ax.set_xlabel('Antibiotic')
    if plot_title:
        ax.set_title(plot_title)
    else:
        ax.set_title(f'Model-wise differences in {plot_metric}')
    # ax.legend(fontsize=9, ncol=2)
    easyPT = Rectangle((0,0),1,1,fc='forestgreen', edgecolor='k', linewidth=0.5)
    mediumPT = Rectangle((0,0),1,1,fc='gold', edgecolor='k', linewidth=0.5)
    hardPT = Rectangle((0,0),1,1,fc='red', edgecolor='k', linewidth=0.5)
    rpt = Rectangle((0,0),1,1,fc='lightgrey', edgecolor='k', linewidth=0.5)
    cpt = Rectangle((0,0),1,1,fc='grey', edgecolor='k', linewidth=0.5)
    ax.legend(
        handles=[easyPT, mediumPT, hardPT, rpt, cpt],
        labels=['Easy', 'Medium', 'Hard', 'RPT', 'CPT'],
        ncols=2,
        framealpha=0,
    )
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=300)
    plt.show()          