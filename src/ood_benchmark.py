import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from core.configuration import load_config
from edge.ood import msp_score, energy_score, calibrate_threshold, is_ood_score

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# the scores to compare, each one maps a batch of logits to one score per utterance (higher = known intent)
OOD_METHODS = {
    "MSP": msp_score,
    "Energy T=1": lambda logits: energy_score(logits, T=1.0),
    "Energy T=2": lambda logits: energy_score(logits, T=2.0),
    "Energy T=3": lambda logits: energy_score(logits, T=3.0),
    "Energy T=5": lambda logits: energy_score(logits, T=5.0),
}


def load_known_splits(config):
    """
    Loads MASSIVE directly (not through DataClass, so intent_registry.json is not touched).
    This loads the utterances of the pretrained intents from the configuration
    """
    ds = load_dataset(
        config["dataset"]["name"],
        "en-US",
        trust_remote_code=True,
    )
    known_ids = set(config["dataset"]["pretrain_intents"])  # raw MASSIVE intent ids

    def known_split(split):
        return ds[split].filter(lambda ex: ex["intent"] in known_ids)["utt"]

    return known_split("validation"), known_split("test")


def run_known_benchmark(student_key="student1", keep=0.95):
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # V0 checkpoint, same path convention as run_simulation._checkpoint_dir
    checkpoint = (
        os.path.join(SRC_DIR, config[student_key]["distill"]["output_dir"]) + "_v0"
    )
    model = (
        AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(config[student_key]["name"])

    utts_a, utts_b = load_known_splits(config)
    print(f"validation: {len(utts_a)} test: {len(utts_b)} utterances")


if __name__ == "__main__":
    # run from src/ (like run_simulation.py): python ood_benchmark.py
    run_known_benchmark()
