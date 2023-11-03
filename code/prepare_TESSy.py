# %%
from pathlib import Path

# user-defined functions
from data_preprocessing import preprocess_TESSy

BASE_DIR = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    path = 'data/raw/TESSy.csv'
    save_path = 'data/TESSy_parsed.pkl'

    df_TESSy = preprocess_TESSy(path=path,
                                pathogens=['ESCCOL'],
                                save_path=save_path,
                                except_antibiotics=['POL', 'DOR'],
                                impute_age=True,
                                impute_gender=True)
                            