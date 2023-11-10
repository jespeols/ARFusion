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
    _ = preprocess_TESSy(path=config['path'],
                        pathogens=config['pathogens'],
                        save_path=config['save_path'],
                        except_antibiotics=config['except_antibiotics'],
                        impute_age=config['impute_age'],
                        impute_gender=config['impute_gender'])
                            