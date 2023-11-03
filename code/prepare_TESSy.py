# %%
import os
import numpy as np
import pandas as pd

from pathlib import Path

# user-defined functions
from utils import impute_col

BASE_DIR = Path(__file__).resolve().parent.parent

def preprocess_TESSy(path,
                     pathogens: list,
                     base_dir: Path = BASE_DIR, 
                     save_path = None,
                     impute_gender: bool = False,
                     impute_age: bool = False,
                     ):
    os.chdir(base_dir)
    
    print(f"Reading in TESSy data from '{path}'...")
    TESSy_data = pd.read_csv(path, low_memory=False)
    print(f"Isolating pathogens: {pathogens}")
    TESSy_data = TESSy_data[TESSy_data['Pathogen'].isin(pathogens)]
    print(f"Number of tests before parsing: {TESSy_data.shape[0]:,}")
    if len(pathogens) > 1:
        cols = ['ReportingCountry', 'DateUsedForStatisticsISO', 'LaboratoryCode', 'PatientCounter',
                'Gender', 'Age','IsolateId', 'Pathogen' 'Antibiotic', 'SIR']
        df = TESSy_data[cols]
        df = df.rename(columns={'ReportingCountry': 'country',
                'DateUsedForStatisticsISO': 'date',
                'Gender': 'gender',
                'Age': 'age',
                'Pathogen': 'pathogen',
                'Antibiotic': 'antibiotic',
                'SIR': 'phenotype'})
    else:
            cols = ['ReportingCountry', 'DateUsedForStatisticsISO', 'LaboratoryCode', 'PatientCounter',
                    'Gender', 'Age','IsolateId', 'Antibiotic', 'SIR']
            df = TESSy_data[cols]
            df = df.rename(columns={'ReportingCountry': 'country',
                    'DateUsedForStatisticsISO': 'date',
                    'Gender': 'gender',
                    'Age': 'age',
                    'Antibiotic': 'antibiotic',
                    'SIR': 'phenotype'})
    
    # drop tests
    alternative_nan = ['unknown', 'UNKNOWN']
    df['IsolateId'] = df['IsolateId'].replace(alternative_nan, np.nan)
    print(f"Dropping {df['IsolateId'].isnull().sum():,} tests with missing IsolateId") 
    df = df[df['IsolateId'].notnull()] 
    I_indices = df[df['phenotype'] == 'I'].index
    print(f"Dropping {len(I_indices):,} tests with 'I' phenotype")  
    df.drop(I_indices, inplace=True)
    
    df['year'] = df['date'].str.split('-').str[0]
    # df.drop(columns=['date'], inplace=True)
    print("Create new ID of the form: country_lab_patientID_year_IsolateID")
    df['ID'] = df['country'].astype(str) + '_' + df['LaboratoryCode'].astype(str) + '_' + df['PatientCounter'].astype(str) + '_' + df['year'].astype(str) + '_' + df['IsolateId'].astype(str)
    print(f"Number of unique IDs: {df['ID'].nunique():,}")
    
    print(f"Are there any ID-antibiotic combinations with more than one date? {'Yes' if any(df.groupby(['ID', 'antibiotic'])['date'].nunique() > 1) else 'No'}")
    duplicates = df.duplicated(subset=['ID', 'antibiotic'])
    print(f"Are there duplicates of ID-antibiotic combination? {'Yes' if duplicates.any() else 'No'}")
    if duplicates.any():
        print(f"Dropping {duplicates.sum():,} duplicates")

        df.drop_duplicates(subset=['ID', 'antibiotic'], inplace=True)
    
    ## Code to look more deeply at duplicates, seeing if there is an antibiotic with different phenotypes. 
    # num_unique_phenotypes = df.groupby(['ID', 'antibiotic'])['phenotype'].nunique().sort_values(ascending=False)
    # print(f"Are there IDs with more than one phenotype per antibiotic? {'Yes' if any(num_unique_phenotypes > 1) else 'No'}")
    # if any(num_unique_phenotypes > 1):
    #     df = df.groupby(['ID', 'antibiotic']).first().reset_index()
    
    print(f"Number of tests after parsing: {df.shape[0]:,}")
    print(f"Aggregating phenotypes for each ID...")
    df_agg = df.groupby('ID')[['antibiotic', 'phenotype']].agg(list).reset_index()
    df_agg['phenotypes'] = df_agg.apply(
        lambda x: [x['antibiotic'][i] + "_" + x['phenotype'][i] for i in range(len(x['antibiotic']))], axis=1)
    df_agg.drop(columns=['antibiotic', 'phenotype'], inplace=True)
    
    df_others = df.drop(columns=['antibiotic', 'phenotype']).groupby('ID').first().reset_index() 
    df_pheno = df_agg.merge(df_others, on='ID')
    
    cols_in_order = ['year', 'country', 'gender', 'age', 'phenotypes'] # can change to date 
    if len(pathogens) > 1:
        df_pheno = df_pheno[['pathogen'] + cols_in_order]
    else:
        df_pheno = df_pheno[cols_in_order]
    df_pheno['num_phenotypes'] = df_pheno['phenotypes'].apply(lambda x: len(x))
    df_pheno['num_R'] = df_pheno['phenotypes'].apply(lambda x: len([p for p in x if p.endswith('R')]))
    df_pheno['num_S'] = df_pheno['phenotypes'].apply(lambda x: len([p for p in x if p.endswith('S')]))
    # make sure there are no samples without phenotypes
    df_pheno = df_pheno[df_pheno['num_phenotypes'] > 0]
    
    if impute_age:
        df_pheno = impute_col(df_pheno, 'age', print_examples=True, random_state=42)
    if impute_gender:
        df_pheno = impute_col(df_pheno, 'gender', print_examples=True, random_state=42)
    
    print(f"Final number of samples: {df_pheno.shape[0]:,}")
    if save_path:
        print(f"Saving to {save_path}")
        df_pheno.to_pickle(save_path)
    print("Done!")

    return df_pheno

if __name__ == '__main__':
    path = 'data/raw/TESSy.csv'
    save_path = 'data/TESSy_parsed.pkl'

    df_TESSy = preprocess_TESSy(path=path,
                                pathogens=['ESCCOL'],
                                save_path=save_path,
                                impute_age=True,
                                impute_gender=True)
                            