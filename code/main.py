# %%
import os
from datetime import datetime
import torch
import yaml

from pathlib import Path
from model import BERT
from datasets import GenotypeDataset
from trainers import BertMLMTrainer

BASE_DIR = Path(__file__).resolve().parent.parent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()

time_str = datetime.now().strftime("%Y%m%d-%H%M%S")

CHECKPOINT_DIR = Path(os.path.join(BASE_DIR / "checkpoints", "experiment_" + str(time_str)))
# LOG_DIR = Path(os.path.join(BASE_DIR / "logs"))
LOG_DIR = Path(os.path.join(BASE_DIR / "logs", "experiment_" + str(time_str)))
RESULTS_DIR = Path(os.path.join(BASE_DIR / "results", "experiment_" + str(time_str)))


if __name__ == "__main__":
    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    else:
        print("Using CPU")
        
    print(f"\nCurrent working directory: {BASE_DIR}")
    print("Loading config file...")
    with open(BASE_DIR / "config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
        
    print("Loading dataset...")
    ds_path = BASE_DIR / "data" / "NCBI" / "genotype_parsed.pkl"
    savepath_vocab = 'data/NCBI/geno_vocab.pt'
    ds = GenotypeDataset(ds_path, subset_share=0.1, savepath_vocab=savepath_vocab, base_dir=BASE_DIR)
            
    max_seq_len = ds.max_seq_len
    vocab_size = len(ds.vocab)
    
    print("Loading model...")
    bert = BERT(config, vocab_size).to(device)
    trainer = BertMLMTrainer(
        model=bert,
        dataset=ds,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
        results_dir=RESULTS_DIR,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        mask_prob=config["mask_prob"],
        report_every=config["report_every"],
        print_progress_every=config["print_progress_every"]
    )
    
    trainer.print_model_summary()
    trainer.print_trainer_summary()
    trainer()
