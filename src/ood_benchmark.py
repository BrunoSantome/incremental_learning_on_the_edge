import os

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
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


def load_ood_split(config):
    """
    Set C: the test utterances of the intents V0 does NOT know (the other 45, reserve ones included).
    The intent name comes back only to group the per-intent table, the model never sees it.
    """
    ds = load_dataset(
        config["dataset"]["name"],
        "en-US",
        trust_remote_code=True,
    )
    known_ids = set(config["dataset"]["pretrain_intents"])
    massive_names = ds["test"].features["intent"].names  # raw MASSIVE id -> intent name

    rows = ds["test"].filter(lambda ex: ex["intent"] not in known_ids)
    return rows["utt"], np.array([massive_names[i] for i in rows["intent"]])


def get_logits(model, tokenizer, utterances, batch_size=64):
    # same forward pass as edge/inference.predict, but over many utterances at once
    device = next(model.parameters()).device
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(utterances), batch_size):
            batch = utterances[start : start + batch_size]
            inputs = tokenizer(
                batch,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            ).to(device)
            all_logits.append(model(**inputs).logits.float().cpu().numpy())
    return np.concatenate(all_logits)  # shape [n utterances, 15]


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

    logits_a = get_logits(model, tokenizer, utts_a)
    logits_b = get_logits(model, tokenizer, utts_b)

    results = {}
    for method, score_fn in OOD_METHODS.items():
        scores_a = score_fn(logits_a)
        scores_b = score_fn(logits_b)
        threshold = calibrate_threshold(scores_a, keep=keep)
        passed_a = np.mean(~is_ood_score(scores_a, threshold))
        passed_b = np.mean(
            ~is_ood_score(scores_b, threshold)
        )  # the real check: should be close to keep
        results[method] = {
            "threshold": threshold,
            "passed_A": passed_a,
            "passed_B": passed_b,
        }
        print(f"{method:<12} {threshold:>10.4f} {passed_a:>10.1%} {passed_b:>10.1%}")
    return results


if __name__ == "__main__":
    # run from src/ (like run_simulation.py): python ood_benchmark.py
    run_known_benchmark()
    """
    
    validation: 657 test: 914 utterances
MSP              0.9986      95.0%      91.6%
Energy T=1       9.5659      95.0%      92.3%
Energy T=2       9.8607      95.0%      91.4%
Energy T=3      10.8940      95.0%      91.7%
Energy T=5      14.7127      95.0%      94.1%
    
    """
