import torch
import yaml
import wandb
import argparse
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.data import Dataset, DataLoader

BASE_DIR = Path(os.path.abspath(''))
sys.path.append(str(BASE_DIR))
os.chdir(BASE_DIR)

# user-defined modules
from multimodal.models import BERT
from multimodal.datasets import MMInferenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--selected_ab", type=str)
    argparser.add_argument("--save_figures", action='store_true')
    
    args = argparser.parse_args()
    
    config = yaml.safe_load(open("config_MM.yaml"))
    data_config = config['data']
    defined_antibiotics = sorted(list(set(data_config['antibiotics']['abbr_to_name'].keys()) - set(data_config['exclude_antibiotics'])))
    ab_to_idx = {ab: idx for idx, ab in enumerate(defined_antibiotics)}
    specials = config['specials']
    cls_token, pad_token, mask_token = specials['CLS'], specials['PAD'], specials['AB_MASK']
    max_seq_len = 56

    ds_path = data_config['TESSy']['load_path']
    # ds_path = 'data/TESSy_15_all_pathogens.pkl'
    ds_TESSy = pd.read_pickle(ds_path)
    ds_NCBI = pd.read_pickle(data_config['NCBI']['load_path'])
    ds_MM = ds_NCBI[ds_NCBI['num_ab'] > 1].reset_index(drop=True)
    print(f"Total number of samples in TESSy: {len(ds_TESSy):,}")

    # antibiotics = ['CAZ', 'CIP', 'AMP', 'GEN'] # original request
    antibiotics = ['CAZ', 'CIP', 'AMP', 'GEN'] # original request
    ds_CAZ = ds_TESSy.copy()
    ds_CAZ['phenotypes'] = ds_CAZ['phenotypes'].apply(lambda x: [p for p in x if p.split('_')[0] in antibiotics])
    # ds_CAZ = ds_CAZ[ds_CAZ['phenotypes'].apply(lambda x: 'CAZ_R' in x)].reset_index(drop=True)
    ds_CAZ = ds_CAZ[ds_CAZ['phenotypes'].apply(lambda x: all([ab in [p.split('_')[0] for p in x] for ab in antibiotics]))].reset_index(drop=True)
    ds_CAZ.drop(columns=['num_R', 'num_S', 'num_ab'], inplace=True)
    ds_CAZ = ds_CAZ.sample(frac=1, random_state=config['random_state']).reset_index(drop=True)
    print(f"Number of selected samples in ds_CAZ: {len(ds_CAZ):,}")

    antibiotics = ['CIP', 'AMP', 'CAZ', 'CTX', 'CRO', 'ETP', 'FEP']
    ds_CIP = ds_TESSy.copy()
    ds_CIP['phenotypes'] = ds_CIP['phenotypes'].apply(lambda x: [p for p in x if p.split('_')[0] in antibiotics])
    # ds_CIP = ds_CIP[ds_CIP['phenotypes'].apply(lambda x: 'CIP_R' in x)].reset_index(drop=True)
    ds_CIP = ds_CIP[ds_CIP['phenotypes'].apply(lambda x: all([ab in [p.split('_')[0] for p in x] for ab in antibiotics]))].reset_index(drop=True)
    ds_CIP.drop(columns=['num_R', 'num_S', 'num_ab'], inplace=True)
    ds_CIP = ds_CIP.sample(frac=1, random_state=config['random_state']).reset_index(drop=True)
    print(f"Number of selected samples in ds_CIP: {len(ds_CIP):,}")

    vocab_path = BASE_DIR / config['fine_tuning']['loadpath_vocab']
    vocab = torch.load(vocab_path)

    CAZ_idx = ab_to_idx['CAZ']
    CIP_idx = ab_to_idx['CIP']
    AMP_idx = ab_to_idx['AMP']
    GEN_idx = ab_to_idx['GEN']

    selected_ab = args.selected_ab
    if selected_ab == 'CAZ':
        ds_exp = ds_CAZ.copy()
        num_samples = 40000
        ab_idx = CAZ_idx
    elif selected_ab == 'CIP':
        ds_exp = ds_CIP.copy()
        num_samples = len(ds_exp)
        ab_idx = CIP_idx
    else:
        ab_idx = ab_to_idx[selected_ab]
        ds_exp = ds_CAZ.copy()
        num_samples = 40000

    selected_ab = defined_antibiotics[ab_idx]
    print("selected ab", selected_ab)
    for patient_info_only in [True, False]:
        print("="*60)
        print("="*60)
        print(f"Patient info only: {patient_info_only}")
        ds_inference = MMInferenceDataset(
            ds_exp.iloc[:num_samples],
            vocab,
            defined_antibiotics,
            max_seq_len,
            specials,
            ab_idx,
            patient_info_only=patient_info_only
        )
        ds_inference.prepare_dataset()
        inference_loader = DataLoader(ds_inference, batch_size=512, shuffle=False)
        
        vocab_size = len(vocab)
        num_ab = 15 # from fine-tuning

        bert = BERT(
            config,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_ab=num_ab,
            pad_idx=vocab[pad_token],
            pheno_only=True
        ).to(device)
        bert.load_state_dict(torch.load(args.model_path))
        
        with torch.no_grad():
            bert.eval()
            # ds = ds_inference_NCBI
            # loader = inference_loader_NCBI  
            ds = ds_inference
            loader = inference_loader
            ds.prepare_dataset()
            tot_num_S, tot_num_R = 0, 0
            tot_correct, tot_num_correct_S, tot_num_correct_R = 0, 0, 0
            tot_num_pred_S, tot_num_pred_R = 0, 0
            pred_sigmoids = torch.tensor([]).to(device)
            targets = torch.tensor([]).to(device)
            i = 0
            for input, token_types, attn_mask, target_res, masked_sequences in loader:
                token_types[token_types == 2] = 1
                pred_logits = bert(input, token_types, attn_mask)
                pred_res = torch.where(pred_logits > 0, torch.ones_like(pred_logits), torch.zeros_like(pred_logits))
                pred_sigmoids = torch.cat((pred_sigmoids, torch.sigmoid(pred_logits[:, ab_idx])))
                targets = torch.cat((targets, target_res))
                ab_preds = pred_res[:, ab_idx]
                num_R_pred = ab_preds.sum().item()
                tot_num_pred_R += num_R_pred
                num_S_pred = ab_preds.shape[0] - num_R_pred
                tot_num_pred_S += num_S_pred 
                # if i == 0 or i == len(loader)-1:
                    # print(f"first masked sequence: {vocab.lookup_tokens(input[0].tolist())}")
                    # print(f"target_res: {target_res[0]}")
                    # print(f"pred_res: {pred_res[0]}")
                    # print(f"pred_res ({selected_ab}): {pred_res[0, ab_idx]}")
                    # print(f"Attention mask: {attn_mask[0]}")
                    # print(f"Token types: {token_types[0]}")
                i += 1
                    
                num_S = target_res.eq(0).sum().item()
                tot_num_S += num_S
                num_R = target_res.eq(1).sum().item()
                tot_num_R += num_R
                
                eq = torch.eq(pred_res[:, ab_idx], target_res)
                num_correct = eq.sum().item()
                tot_correct += num_correct
                num_correct_R = eq[target_res == 1].sum().item()
                tot_num_correct_R += num_correct_R
                num_correct_S = eq[target_res == 0].sum().item()
                tot_num_correct_S += num_correct_S
                num_R_pred = pred_res[:, ab_idx].sum().item()
            print(f"Total {selected_ab} accuracy: {tot_correct/(tot_num_S+tot_num_R):.4f}")
            print(f"Data imbalance (R_share) of {selected_ab}: {tot_num_R/(tot_num_S+tot_num_R):.4f}")
            print(f"Share of predictions that were {selected_ab}_R: {tot_num_pred_R/(tot_num_S+tot_num_R):.4f}")
            print()
            print(f"Share of predictions that were {selected_ab}_S: {tot_num_pred_S/(tot_num_S+tot_num_R):.4f}")
            print(f"Total {selected_ab}_R accuracy: {tot_num_correct_R/tot_num_R:.4f}")
            if tot_num_pred_R > 0:
                print(f"Precision: {tot_num_correct_R/tot_num_pred_R:.4f}")
            print(f"Total {selected_ab}_S accuracy: {tot_num_correct_S/tot_num_S:.4f}")
            
            ## ROC stats
            pred_sigmoids_np = pred_sigmoids.cpu().numpy()
            targets_np = targets.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(targets_np, pred_sigmoids_np)
            auc_score = auc(fpr, tpr)
            print(f"AUC: {auc_score:.4f}")
            best_index = np.argmax(tpr-fpr)
            best_threshold = thresholds[best_index]
            fpr_best, tpr_best = fpr[best_index], tpr[best_index]
            fdr = fpr_best/(fpr_best+tpr_best)
            print(f"Best threshold: {best_threshold:.4f}")
            print(f"At best, TPR: {tpr[best_index]:.4f}, with FPR: {fpr[best_index]:.4f} and FDR: {fdr:.4f}")
            
            ## Plot ROC curve
            fig1, ax = plt.subplots(figsize=(6, 6))
            ax.plot(fpr, tpr, color='forestgreen', lw=2, label=f'AUROC = {auc_score:.3f}')
            ax.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
            label = f'Best threshold = {best_threshold:.3f}\nTPR = {tpr_best:.3f}, FPR = {fpr_best:.3f}\nFDR = {fdr:.3f}'
            ax.scatter(fpr_best, tpr_best, color='red', label=label)
            ax.set_xlim([0.0, 1.0])
            ax.legend(loc='lower right')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC curve for {selected_ab}_R prediction')
            if patient_info_only:
                ax.set_title(f'ROC curve for {selected_ab}_R prediction (patient info only)')
            else:
                ax.set_title(f'ROC curve for {selected_ab}_R prediction')
            if args.save_figures:
                parent_dir = Path(args.model_path).parent
                if patient_info_only:
                    save_path = parent_dir / f"pat_only_ROC_{selected_ab}.png"
                else:
                    save_path = parent_dir / f"ROC_{selected_ab}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(3.5)
            plt.close(fig1)
            
            ## Sigmoid histogram
            pred_sigmoids_R = pred_sigmoids_np[targets_np == 1]
            pred_sigmoids_S = pred_sigmoids_np[targets_np == 0]
            fig2, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].hist(pred_sigmoids_S, bins=50, color='green', label=f'{selected_ab}_S')
            axes[0].set_title(f'{selected_ab}_S sigmoid distribution')
            axes[0].set_xlim(0, 1)
            axes[1].hist(pred_sigmoids_R, bins=50, color='red', label=f'{selected_ab}_R')
            axes[1].set_title(f'{selected_ab}_R sigmoid distribution')
            axes[1].set_xlim(0, 1)
            if patient_info_only:
                plt.suptitle(f'Sigmoid distributions (patient info only)')
            else:
                plt.suptitle(f'Sigmoid distributions')
            if args.save_figures:
                parent_dir = Path(args.model_path).parent
                if patient_info_only:
                    save_path = parent_dir / f"pat_only_sigmoid_{selected_ab}.png"
                else:
                    save_path = parent_dir / f"sigmoid_{selected_ab}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(3.5)
            plt.close(fig2)
