# %%
import os
import numpy as np
import pandas as pd

from pathlib import Path

# user-defined functions
from utils import filter_gene_counts

BASE_DIR = Path(__file__).resolve().parent.parent

def preprocess_NCBI(path,
                    base_dir: Path = BASE_DIR,
                    include_phenotype: bool = False, 
                    save_path = None,
                    threshold_year: int = None,
                    exclude_genotypes: list = None,
                    exclude_assembly_variants: list = None,
                    exclusion_chars: list = None,
                    gene_count_threshold: int = None,
                    ):
    os.chdir(base_dir)
    
    NCBI_data = pd.read_csv(path, sep='\t', low_memory=False)
    cols = ['collection_date', 'geo_loc_name', 'AMR_genotypes_core']
    cols += ['AST_phenotypes'] if include_phenotype else []
    df = NCBI_data[cols]
    df = df.rename(columns={'AMR_genotypes_core': 'genotypes'})
    if include_phenotype:
        df = df.rename(columns={'AST_phenotypes': 'phenotypes'}) 
    df = df[df['genotypes'].notnull()] # filter missing genotypes
    
    #################################### PARSING ####################################
    
    #### geo_loc_name -> country 
    alternative_nan = ['not determined', 'not collected', 'not provided', 'Not Provided',
                   'OUTPATIENT', 'Not collected', 'Not Collected', 'not available']
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].replace(alternative_nan, np.nan) 
    
    # Remove regional information
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].str.split(',').str[0]
    df.loc[:,'geo_loc_name'] = df['geo_loc_name'].str.split(':').str[0] 
    df = df.rename(columns={'geo_loc_name': 'country'})
    
    ##### collection_date -> year
    alternative_nan = ['missing']
    df.loc[:,'collection_date'] = df['collection_date'].replace(alternative_nan, np.nan)
    df.loc[:,'collection_date'] = df['collection_date'].str.split('-').str[0]
    df.loc[:,'collection_date'] = df['collection_date'].str.split('/').str[0]
    df = df.rename(columns={'collection_date': 'year'})
    if threshold_year:
        indices = df[df['year'].astype(float) <= threshold_year].index
        df.drop(indices, inplace=True)
    
    #### Parse genotypes
    df['genotypes'] = df['genotypes'].str.split(',')
    print("Parsing genotypes...")
    print(f"Number of samples before parsing: {df.shape[0]}")
    
    df['genotypes'] = df['genotypes'].apply(lambda x: list(set([g.strip() for g in x]))) # remove whitespace and duplicates
    if exclude_genotypes: 
        print(f"Removing genotypes: {exclude_genotypes}")
        df['genotypes'] = df['genotypes'].apply(lambda x: [g for g in x if g not in exclude_genotypes]) 
    
    if exclude_assembly_variants: # Examples: ["=PARTIAL", "=MISTRANSLATION", "=HMM"]
        print(f"Removing genotypes with assembly variants: {exclude_assembly_variants}")
        df['genotypes'] = df['genotypes'].apply(lambda x: [g for g in x if not g.endswith(tuple(exclude_assembly_variants))]) 
    df = df[df['genotypes'].apply(lambda x: len(x) > 0)] # Remove any rows where genotypes are empty
    
    if exclusion_chars:
        # old_genotypes = df['genotypes'].copy() # save old genotypes for later
        for char in exclusion_chars:
            print(f"Splitting genotypes by '{char}', removing it and everything after it")
            df['genotypes'] = df['genotypes'].apply(lambda x: list(set([g.split(char)[0] for g in x])))
            
    # Remove cases where there is both a genotype and an assembly variant
    assembly_chars = ['=PARTIAL', '=MISTRANSLATION', '=HMM', '=PARTIAL_END_OF_CONTIG']
    df['genotypes'] = df['genotypes'].apply(
        lambda x: list(set(x) - set([g for g in x if g.endswith(tuple(assembly_chars)) and g.split("=")[0] in x])))
          
    df['num_genotypes'] = df['genotypes'].apply(lambda x: len(set(x)))
    df['num_point_mutations'] = df['genotypes'].apply(lambda x: len([g for g in x if '=POINT' in g]))
    
    if gene_count_threshold:
        df = filter_gene_counts(df, gene_count_threshold)

    # Exclude cases where there is only one genotype and no other info
    df = df[~((df['num_genotypes'] == 1) & (df['country'].isnull()) & (df['year'].isnull()))]
    print(f"Number of samples after parsing: {df.shape[0]}")
    
    df.to_pickle(save_path) if save_path else None
    
    return df

##################################################################################################################################

from utils import impute_col


def preprocess_TESSy(path,
                     pathogens: list,
                     base_dir: Path = BASE_DIR, 
                     save_path = None,
                     exclude_antibiotics: list = None,
                     impute_gender: bool = False,
                     impute_age: bool = False,
                     ):
    os.chdir(base_dir)
    
    print(f"Reading in TESSy data from '{path}'...")
    TESSy_data = pd.read_csv(path, low_memory=False)
    print(f"Pathogens: {pathogens}")
    TESSy_data = TESSy_data[TESSy_data['Pathogen'].isin(pathogens)]
    print(f"Number of tests before parsing: {TESSy_data.shape[0]:,}")
    TESSy_data['year'] = pd.to_datetime(TESSy_data['DateUsedForStatisticsISO']).dt.year
    TESSy_data['date'] = pd.to_datetime(TESSy_data['DateUsedForStatisticsISO'], format='%Y-%m-%d')
    TESSy_data.drop(columns=['DateUsedForStatisticsISO'], inplace=True)
    TESSy_data = TESSy_data[TESSy_data['SIR'] != 'I']
    if len(pathogens) > 1:
        cols = ['ReportingCountry', 'date', 'year', 'LaboratoryCode', 'PatientCounter',
                'Gender', 'Age','IsolateId', 'Pathogen' 'Antibiotic', 'SIR']
        df = TESSy_data[cols]
        df = df.rename(columns={'ReportingCountry': 'country',
                'Gender': 'gender',
                'Age': 'age',
                'Pathogen': 'pathogen',
                'Antibiotic': 'antibiotic',
                'SIR': 'phenotype'})
    else:
            cols = ['ReportingCountry', 'date', 'year', 'LaboratoryCode', 'PatientCounter',
                    'Gender', 'Age','IsolateId', 'Antibiotic', 'SIR']
            df = TESSy_data[cols]
            df = df.rename(columns={'ReportingCountry': 'country',
                    'Gender': 'gender',
                    'Age': 'age',
                    'Antibiotic': 'antibiotic',
                    'SIR': 'phenotype'})
    
    # drop tests
    alternative_nan = ['unknown', 'UNKNOWN']
    df['IsolateId'] = df['IsolateId'].replace(alternative_nan, np.nan)
    print(f"Dropping {df['IsolateId'].isnull().sum():,} tests with missing IsolateId") 
    df = df[df['IsolateId'].notnull()] 
    
    # filter out antibiotics
    if exclude_antibiotics:
        print(f"Filtering out antibiotics: {exclude_antibiotics}")
        df = df[~df['antibiotic'].isin(exclude_antibiotics)]
        print(f"Number of antibiotics: {df['antibiotic'].nunique():,}")
    
    print("Creating new ID of the form: country_year_labID_patientID_IsolateID")
    id_cols = ['country', 'year', 'LaboratoryCode', 'PatientCounter', 'IsolateId']
    df['ID'] = df[id_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
    print(f"Number of unique IDs: {df['ID'].nunique():,}")
    
    print(f"Are there any ID-antibiotic combinations with more than one date? {'Yes' if any(df.groupby(['ID', 'antibiotic'])['date'].nunique() > 1) else 'No'}")
    duplicates = df.duplicated(subset=['ID', 'antibiotic', 'phenotype'])
    print(f"Are there duplicates of ID-antibiotic-phenotype combination? {'Yes' if duplicates.any() else 'No'}")
    if duplicates.any():
        print(f"Dropping {duplicates.sum():,} duplicates")
        df.drop_duplicates(subset=['ID', 'antibiotic', 'phenotype'], inplace=True, keep='first')
    
    ## Code to look more deeply at duplicates, seeing if there is an antibiotic with different phenotypes. 
    num_unique_phenotypes = df.groupby(['ID', 'antibiotic'])['phenotype'].nunique().sort_values(ascending=False)
    print(f"Are there IDs with more than one phenotype per antibiotic? {'Yes' if any(num_unique_phenotypes > 1) else 'No'}")
    if any(num_unique_phenotypes > 1):
        df = df.groupby(['ID', 'antibiotic']).first().reset_index()
    
    print(f"Number of tests after parsing: {df.shape[0]:,}")
    print(f"Aggregating phenotypes for each ID...")
    df_agg = df.groupby('ID')[['antibiotic', 'phenotype']].agg(list).reset_index()
    df_agg['phenotypes'] = df_agg.apply(
        lambda x: [x['antibiotic'][i] + "_" + x['phenotype'][i] for i in range(len(x['antibiotic']))], axis=1)
    df_agg.drop(columns=['antibiotic', 'phenotype'], inplace=True)
    
    df_others = df.drop(columns=['antibiotic', 'phenotype']).groupby('ID').first().reset_index() 
    df_pheno = df_agg.merge(df_others, on='ID')
    
    cols_in_order = ['year', 'country', 'gender', 'age', 'phenotypes'] # can change to date or year-month here
    if len(pathogens) > 1:
        df_pheno = df_pheno[['pathogen'] + cols_in_order]
    else:
        df_pheno = df_pheno[cols_in_order]
    df_pheno['num_ab'] = df_pheno['phenotypes'].apply(lambda x: len(x))
    df_pheno['num_R'] = df_pheno['phenotypes'].apply(lambda x: len([p for p in x if p.endswith('R')]))
    df_pheno['num_S'] = df_pheno['phenotypes'].apply(lambda x: len([p for p in x if p.endswith('S')]))
    # make sure there are no samples without phenotypes
    df_pheno = df_pheno[df_pheno['num_ab'] > 0]
    
    if impute_age:
        df_pheno = impute_col(df_pheno, 'age', random_state=42)
    else:
        print(f"Dropping {df_pheno['age'].isnull().sum():,} samples with missing value in the 'age' column")
        df_pheno.dropna(subset=['age'], inplace=True)
        
    alternative_nan = ["UNK", "O"]
    df_pheno['gender'].replace(alternative_nan, np.nan, inplace=True)
    if impute_gender:
        df_pheno = impute_col(df_pheno, 'gender', random_state=42)
    else:
        print(f"Dropping {df_pheno['gender'].isnull().sum():,} samples with missing value in the 'gender' column")
        df_pheno.dropna(subset=['gender'], inplace=True)

    if not any([impute_age, impute_gender]):
        print(f"Number of samples after dropping samples with missing values: {df_pheno.shape[0]:,}")
    else:
        print(f"Final number of samples: {df_pheno.shape[0]:,}")
    
    df_pheno.reset_index(drop=True, inplace=True)
    if save_path:
        print(f"Saving to {save_path}")
        df_pheno.to_pickle(save_path)
    print("Done!")

    return df_pheno