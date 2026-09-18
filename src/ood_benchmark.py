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
    The intent name and the near/far label come back only to group the results, the model never sees them.

    near-OOD: the intent's scenario also contains a known intent
    far-OOD: the intent's scenario has no known intent
    """
    ds = load_dataset(
        config["dataset"]["name"],
        "en-US",
        trust_remote_code=True,
    )
    known_ids = set(config["dataset"]["pretrain_intents"])
    massive_names = ds["test"].features["intent"].names  # raw MASSIVE id -> intent name
    known_scenarios = {
        massive_names[i].split("_")[0] for i in known_ids
    }  # if there is a match of scenario vs a new intent

    rows = ds["test"].filter(lambda ex: ex["intent"] not in known_ids)
    intents = np.array([massive_names[i] for i in rows["intent"]])
    is_far = np.array([name.split("_")[0] not in known_scenarios for name in intents])
    return rows["utt"], intents, is_far


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
    utts_c, intents_c, is_far = load_ood_split(config)
    print(
        f"validation: {len(utts_a)} test: {len(utts_b)} utterances"
        f" , unknown intents: {len(utts_c)} utterances over {len(set(intents_c))} intents"
        f" , far: {is_far.sum()} utterances over {len(set(intents_c[is_far]))} intents"
        f" , near: {(~is_far).sum()} utterances over {len(set(intents_c[~is_far]))} intents"
    )

    logits_a = get_logits(model, tokenizer, utts_a)
    logits_b = get_logits(model, tokenizer, utts_b)
    logits_c = get_logits(model, tokenizer, utts_c)

    # AUROC labels truth: known (B) = 1, far-OOD = 0, the labels that the model knows vs the ones that dont
    is_known_far = np.concatenate([np.ones(len(utts_b)), np.zeros(is_far.sum())])

    print(
        f"\n{'method':<12} {'threshold':>12} {'passed B':>12}"
        f" {'caught far':>12} {'caught near':>12} {'AUROC far':>12}"
    )
    results = {}
    caught_per_intent = {}  # method -> {intent name: share of its utterances escalated}
    for method, score_fn in OOD_METHODS.items():
        scores_a = score_fn(logits_a)
        scores_b = score_fn(logits_b)
        scores_c = score_fn(logits_c)

        threshold = calibrate_threshold(scores_a, keep=keep)  # set A only
        passed_b = np.mean(
            ~is_ood_score(scores_b, threshold)
        )  # cost: 1 - this = known escalated
        escalated_c = is_ood_score(scores_c, threshold)
        caught_far = escalated_c[is_far].mean()  # far-OOD is the target
        caught_near = escalated_c[~is_far].mean()  # documented limitation
        auroc_far = roc_auc_score(
            is_known_far, np.concatenate([scores_b, scores_c[is_far]])
        )  # comparison with the paper

        results[method] = {
            "threshold": threshold,
            "passed_B": passed_b,
            "caught_far": caught_far,
            "caught_near": caught_near,
            "auroc_far": auroc_far,
        }
        caught_per_intent[method] = {
            intent: escalated_c[intents_c == intent].mean()
            for intent in sorted(set(intents_c))
        }
        print(
            f"{method:<12} {threshold:>10.4f} {passed_b:>10.1%}"
            f" {caught_far:>11.1%} {caught_near:>12.1%} {auroc_far:>10.3f}"
        )

    # per-intent: which unknown intents slip through, grouped far/near, hardest (least caught) first
    per_intent = pd.DataFrame(caught_per_intent)
    far_intents = set(intents_c[is_far])
    per_intent.insert(
        0, "group", ["far" if i in far_intents else "near" for i in per_intent.index]
    )
    per_intent["mean"] = per_intent[list(OOD_METHODS)].mean(axis=1)
    per_intent = per_intent.sort_values(["group", "mean"]).drop(columns="mean")
    print(
        f"\ncaught per unknown intent\n{per_intent.to_string(float_format='{:.1%}'.format)}"
    )
    return results, per_intent


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

    """

method          threshold     passed B   caught far  caught near    AUROC far
MSP              0.9986      91.6%       64.5%        52.1%      0.874
Energy T=1       9.5659      92.3%       67.3%        53.9%      0.863
Energy T=2       9.8607      91.4%       67.8%        55.7%      0.862

caught per unknown intent
                         group   MSP  Energy T=1  Energy T=2
general_greet              far  0.0%        0.0%        0.0%
music_dislikeness          far  0.0%        0.0%        0.0%
music_settings             far 16.7%        0.0%        0.0%
music_query                far  5.7%        5.7%        5.7%
social_query               far  4.0%       16.0%       16.0%
music_likeness             far 13.9%       13.9%       13.9%
recommendation_events      far 25.6%       27.9%       27.9%
qa_stock                   far 38.5%       46.2%       46.2%
iot_wemo_off               far 27.8%       50.0%       55.6%
iot_hue_lightoff           far 30.2%       53.5%       55.8%
calendar_query             far 48.4%       51.6%       52.4%
general_joke               far 47.4%       57.9%       57.9%
general_quirky             far 55.0%       56.2%       55.6%
qa_definition              far 63.2%       57.9%       57.9%
iot_hue_lighton            far 66.7%       66.7%       66.7%
iot_wemo_on                far 70.0%       70.0%       60.0%
recommendation_movies      far 70.0%       65.0%       65.0%
recommendation_locations   far 67.7%       67.7%       67.7%
iot_hue_lightup            far 63.0%       74.1%       74.1%
qa_currency                far 71.8%       71.8%       74.4%
qa_factoid                 far 70.2%       74.5%       78.0%
social_post                far 80.2%       77.8%       77.8%
calendar_set               far 81.8%       83.7%       83.3%
takeaway_order             far 86.4%       81.8%       81.8%
iot_cleaning               far 84.6%       84.6%       84.6%
takeaway_query             far 85.7%       85.7%       85.7%
iot_hue_lightdim           far 61.9%      100.0%      100.0%
iot_coffee                 far 88.9%       91.7%       91.7%
cooking_recipe             far 91.7%       90.3%       90.3%
calendar_remove            far 92.5%       94.0%       95.5%
qa_maths                   far 96.0%       96.0%       96.0%
iot_hue_lightchange        far 97.2%       97.2%       97.2%
transport_ticket          near 17.1%       17.1%       17.1%
lists_query               near 21.6%       19.6%       19.6%
lists_remove              near 28.8%       21.2%       21.2%
datetime_convert          near 26.7%       26.7%       26.7%
audio_volume_other        near 33.3%       33.3%       33.3%
alarm_query               near 35.3%       50.0%       52.9%
alarm_remove              near 42.9%       52.4%       52.4%
transport_traffic         near 46.7%       53.3%       53.3%
email_query               near 58.0%       62.2%       68.9%
play_podcasts             near 73.0%       60.3%       58.7%
email_sendemail           near 74.6%       82.5%       84.2%
email_querycontact        near 80.8%       84.6%       84.6%
    """
