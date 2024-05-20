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
from pheno.models import BERT
from pheno.datasets import PhenoInferenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--model_name", type=str)
    argparser.add_argument("--selected_ab", type=str)
    argparser.add_argument("--save_figures", action='store_true')
    
    args = argparser.parse_args()
    
    config = yaml.safe_load(open("config_pheno.yaml"))
    data_config = config['data']
    defined_antibiotics = sorted(list(set(data_config['antibiotics']['abbr_to_names'].keys()) - set(data_config['exclude_antibiotics'])))
    ab_to_idx = {ab: idx for idx, ab in enumerate(defined_antibiotics)}
    specials = config['specials']
    cls_token, pad_token, mask_token = specials['CLS'], specials['PAD'], specials['MASK']
    max_seq_len = 19

    ds_path = data_config['load_path']
    # ds_path = 'data/TESSy_15_all_pathogens.pkl'
    ds_TESSy = pd.read_pickle(ds_path)
    ds_TESSy = ds_TESSy.sample(frac=1, random_state=config['random_state']).reset_index(drop=True)
    print(f"Total number of samples in TESSy: {len(ds_TESSy):,}")

    antibiotics = ['CAZ', 'CIP', 'AMP', 'GEN'] # original request
    ds_CAZ = ds_TESSy.copy()
    ds_CAZ['phenotypes'] = ds_CAZ['phenotypes'].apply(lambda x: [p for p in x if p.split('_')[0] in antibiotics])
    # ds_CAZ = ds_CAZ[ds_CAZ['phenotypes'].apply(lambda x: 'CAZ_R' in x)].reset_index(drop=True)
    ds_CAZ = ds_CAZ[ds_CAZ['phenotypes'].apply(lambda x: all([ab in [p.split('_')[0] for p in x] for ab in antibiotics]))].reset_index(drop=True)
    ds_CAZ.drop(columns=['num_R', 'num_S', 'num_ab'], inplace=True)
    print(f"Number of selected samples in ds_CAZ: {len(ds_CAZ):,}")

    antibiotics = ['CIP', 'AMP', 'CAZ', 'CTX', 'CRO', 'ETP', 'FEP']
    ds_CIP = ds_TESSy.copy()
    ds_CIP['phenotypes'] = ds_CIP['phenotypes'].apply(lambda x: [p for p in x if p.split('_')[0] in antibiotics])
    # ds_CIP = ds_CIP[ds_CIP['phenotypes'].apply(lambda x: 'CIP_R' in x)].reset_index(drop=True)
    ds_CIP = ds_CIP[ds_CIP['phenotypes'].apply(lambda x: all([ab in [p.split('_')[0] for p in x] for ab in antibiotics]))].reset_index(drop=True)
    ds_CIP.drop(columns=['num_R', 'num_S', 'num_ab'], inplace=True)
    print(f"Number of selected samples in ds_CIP: {len(ds_CIP):,}")

    vocab_path = BASE_DIR / 'pheno_vocab.pt'
    vocab = torch.load(vocab_path)

    CAZ_idx = ab_to_idx['CAZ']
    CIP_idx = ab_to_idx['CIP']
    AMP_idx = ab_to_idx['AMP']
    GEN_idx = ab_to_idx['GEN']

    selected_ab = args.selected_ab
    ab_selected = True
    if selected_ab == 'CAZ':
        ds_exp = ds_CAZ.copy()
        num_samples = 40000
        ab_idx = CAZ_idx
    elif selected_ab == 'CIP':
        ds_exp = ds_CIP.copy()
        num_samples = len(ds_exp)
        ab_idx = CIP_idx
    else:
        if args.selected_ab is not None:
            print("Invalid antibiotic selected, will loop through all antibiotics")
        else:
            print("No antibiotic selected, will loop through all antibiotics")
        ab_selected = False
    if ab_selected:
        selected_ab = defined_antibiotics[ab_idx]
        print("selected ab", selected_ab)
        for patient_info_only in [True, False]:
            print("="*60)
            print("="*60)
            print(f"Patient info only: {patient_info_only}")
            ds_inference = PhenoInferenceDataset(
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
            ).to(device)
            bert.load_state_dict(torch.load(args.model_path))
            
            with torch.no_grad():
                bert.eval()
                # ds = ds_inference_NCBI
                # loader = inference_loader_NCBI  
                ds = ds_inference
                loader = inference_loader
                print("selected ab:", selected_ab)
                print("Number of samples in inference dataset:", len(ds))
                print("Number of batches in inference loader:", len(loader))
                print("="*50)
                tot_num_S, tot_num_R = 0, 0
                tot_correct, tot_num_correct_S, tot_num_correct_R = 0, 0, 0
                tot_num_pred_S, tot_num_pred_R = 0, 0
                pred_sigmoids = torch.tensor([]).to(device)
                targets = torch.tensor([]).to(device)
                for input, attn_mask, target_res in loader:
                    pred_logits = bert(input, attn_mask)
                    pred_res = torch.where(pred_logits > 0, torch.ones_like(pred_logits), torch.zeros_like(pred_logits))
                    pred_sigmoids = torch.cat((pred_sigmoids, torch.sigmoid(pred_logits[:, ab_idx])))
                    targets = torch.cat((targets, target_res))
                    ab_preds = pred_res[:, ab_idx]
                    num_R_pred = ab_preds.sum().item()
                    tot_num_pred_R += num_R_pred
                    num_S_pred = ab_preds.shape[0] - num_R_pred
                    tot_num_pred_S += num_S_pred 
                        
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
                ## Find values at FPR = 0.1
                fpr_val = 0.1
                fpr_index = np.argmin(np.abs(fpr - fpr_val))
                best_threshold = thresholds[fpr_index]
                fpr_best, tpr_best = fpr[fpr_index], tpr[fpr_index]
                fdr = fpr_best/(fpr_best+tpr_best)
                
                ## Find values at threshold = 0.5
                threshold_val = 0.5
                threshold_index = np.argmin(np.abs(thresholds - threshold_val))
                threshold_standard = thresholds[threshold_index]
                fpr_standard, tpr_standard = fpr[threshold_index], tpr[threshold_index]
                fdr_standard = fpr_standard/(fpr_standard+tpr_standard)
                
                ## Find "best threshold", as in largest TPR - FPR difference
                # best_index = np.argmax(tpr-fpr)
                # best_threshold = thresholds[best_index]
                # fpr_best, tpr_best = fpr[best_index], tpr[best_index]
                # fdr = fpr_best/(fpr_best+tpr_best)
                # print(f"Best threshold: {best_threshold:.4f}")
                # print(f"At best, TPR: {tpr[best_index]:.4f}, with FPR: {fpr[best_index]:.4f} and FDR: {fdr:.4f}")
                
                ## Plot ROC curve
                fig1, ax = plt.subplots(figsize=(6, 6))
                ax.plot(fpr, tpr, color='forestgreen', lw=2, label=f'AUROC = {auc_score:.3f}')
                ax.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
                label = f'FPR = {fpr_best:.1f}, TPR = {tpr_best:.3f}\nthreshold = {best_threshold:.3f}, FDR = {fdr:.3f}'
                ax.scatter(fpr_best, tpr_best, color='red', label=label)
                label_standard = f'Threshold = {threshold_standard:.1f}, TPR = {tpr_standard:.3f}\nFPR = {fpr_standard:.3f}, FDR = {fdr_standard:.3f}'
                ax.scatter(fpr_standard, tpr_standard, color='orange', label=label_standard)
                ax.set_xlim([0.0, 1.0])
                ax.legend(loc='lower right')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
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
    else:
        print("="*60)
        print("Generate 'patient_info_only' predictions for all antibiotics")
        num_ab = len(defined_antibiotics)
        tpr_vals, fpr_vals = np.zeros(num_ab), np.zeros(num_ab) # at threshold 0.5
        tpr_vals_point1, thresholds_point1 = np.zeros(num_ab), np.zeros(num_ab) # at FPR = 0.1
        auc_scores = np.zeros(num_ab)
        S_shares, R_shares = np.zeros(num_ab), np.zeros(num_ab)
        for i, ab in enumerate(defined_antibiotics):
            ab_idx = ab_to_idx[ab]
            print("="*60)
            print(f"Predicting {ab} ({i+1}/{num_ab})...")
            ds_ab = ds_TESSy[ds_TESSy['phenotypes'].apply(lambda x: ab in [p.split('_')[0] for p in x])].reset_index(drop=True)
            max_num_samples = 100000
            # num_samples = min(max_num_samples, len(ds_ab))
            num_samples = len(ds_ab)
            print(f"Number of samples in ds_{ab}: {len(ds_ab):,}")
            ds_inference = PhenoInferenceDataset(
                ds_ab.iloc[:num_samples],
                vocab,
                defined_antibiotics,
                max_seq_len,
                specials,
                ab_idx,
                patient_info_only=True
            )
            ds_inference.prepare_dataset()
            inference_loader = DataLoader(ds_inference, batch_size=1024, shuffle=False)
            
            vocab_size = len(vocab)
            bert = BERT(
                config,
                vocab_size=vocab_size,
                max_seq_len=max_seq_len,
                num_ab=num_ab,
                pad_idx=vocab[pad_token],
            ).to(device)
            bert.load_state_dict(torch.load(args.model_path))
            
            with torch.no_grad():
                bert.eval()
                tot_num_S, tot_num_R = 0, 0
                tot_correct, tot_num_correct_S, tot_num_correct_R = 0, 0, 0
                tot_num_pred_S, tot_num_pred_R = 0, 0
                pred_sigmoids = torch.tensor([]).to(device)
                targets = torch.tensor([]).to(device)
                loader = inference_loader
                for input, attn_mask, target_res in loader:
                    pred_logits = bert(input, attn_mask)
                    pred_res = torch.where(pred_logits > 0, torch.ones_like(pred_logits), torch.zeros_like(pred_logits))
                    pred_sigmoids = torch.cat((pred_sigmoids, torch.sigmoid(pred_logits[:, ab_idx])))
                    targets = torch.cat((targets, target_res))
                    ab_preds = pred_res[:, ab_idx]
                    num_R_pred = ab_preds.sum().item()
                    tot_num_pred_R += num_R_pred
                    num_S_pred = ab_preds.shape[0] - num_R_pred
                    tot_num_pred_S += num_S_pred 
                        
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
                print(f"Overall {ab} accuracy: {tot_correct/(tot_num_S+tot_num_R):.4f}")
                print(f"{ab}_R accuracy: {tot_num_correct_R/tot_num_R:.4f}")
                if tot_num_pred_R > 0:
                    print(f"Precision: {tot_num_correct_R/tot_num_pred_R:.4f}")
                print(f"{ab}_S accuracy: {tot_num_correct_S/tot_num_S:.4f}")
                tpr_vals[ab_idx] = tot_num_correct_R/tot_num_R
                fpr_vals[ab_idx] = 1 - tot_num_correct_S/tot_num_S
                S_shares[ab_idx] = tot_num_S/(tot_num_S+tot_num_R)
                R_shares[ab_idx] = tot_num_R/(tot_num_S+tot_num_R)
                
                ## ROC stats
                pred_sigmoids_np = pred_sigmoids.cpu().numpy()
                targets_np = targets.cpu().numpy()
                fpr, tpr, thresholds = roc_curve(targets_np, pred_sigmoids_np)
                auc_score = auc(fpr, tpr)
                auc_scores[ab_idx] = auc_score
                print(f"AUC: {auc_score:.4f}")
                
                ## Find values at FPR = 0.1
                fpr_val = 0.1
                fpr_index = np.argmin(np.abs(fpr - fpr_val))
                threshold_point1 = thresholds[fpr_index]
                fpr_point1, tpr_point1 = fpr[fpr_index], tpr[fpr_index]
                fdr_point1 = fpr_point1/(fpr_point1+tpr_point1)
                thresholds_point1[ab_idx] = threshold_point1
                tpr_vals_point1[ab_idx] = tpr_point1
                
                ## Find values at threshold = 0.5
                threshold_val = 0.5
                threshold_index = np.argmin(np.abs(thresholds - threshold_val))
                threshold_standard = thresholds[threshold_index]
                fpr_standard, tpr_standard = fpr[threshold_index], tpr[threshold_index]
                fdr_standard = fpr_standard/(fpr_standard+tpr_standard)
                fpr_vals[ab_idx] = fpr_standard
                tpr_vals[ab_idx] = tpr_standard
            abbr_to_class = data_config['antibiotics']['abbr_to_class']
        results = pd.DataFrame({
            'Antibiotic': defined_antibiotics,
            'Antibiotic class': [abbr_to_class[ab] for ab in defined_antibiotics],
            'TPR (at threshold 0.5)': tpr_vals,
            'FPR (at threshold 0.5)': fpr_vals,
            'TPR (at FPR=0.1)': tpr_vals_point1,
            'threshold (at FPR=0.1)': thresholds_point1,
            'AUROC': auc_scores,
            'S_share': S_shares,
            'R_share': R_shares
        })
        print(results)
        if args.model_name:
            results.to_csv(f"{args.model_name}_pat_info_only_all_ab.csv", index=False, float_format='%.4f')
        else:    
            results.to_csv(f"pat_info_only_all_ab.csv", index=False, float_format='%.4f')
