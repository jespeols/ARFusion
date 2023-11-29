# %%
import yaml
from pathlib import Path

# user-defined functions
from data_preprocessing import preprocess_TESSy

BASE_DIR = Path(__file__).resolve().parent.parent

if __name__ == '__main__':
    config_path = BASE_DIR / "config_pheno.yaml"
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    config = config['data']
    _ = preprocess_TESSy(path=config['data']['path'],
                         pathogens=config['data']['pathogens'],
                         save_path=config['data']['save_path'],
                         exclude_antibiotics=config['data']['exclude_antibiotics'],
                         impute_age=config['data']['impute_age'],
                         impute_gender=config['data']['impute_gender'])