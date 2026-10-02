# Synthetic utterances for the reserve intents, generated ONCE and reused byte-identically.
#
# The synthetic ablation compares training mechanisms, so the generated data has to be the same in
# every condition: generate_new_intent() calls the LLM on every call, which would give each
# condition different utterances and confound the comparison. This module caches them to
# outputs/datasets/synth_utterances_s{seed}.json, which is therefore an experimental control.

import json
import os

from server.paraphrase import generate_new_intent

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(SRC_DIR, "outputs", "datasets")


def cache_path(seed=42):
    return os.path.join(CACHE_DIR, f"synth_utterances_s{seed}.json")


def escalation_cache_path(escalations_file, seed=42):
    stem = os.path.splitext(os.path.basename(escalations_file))[0]
    return os.path.join(CACHE_DIR, f"{stem}_generated_s{seed}.json")


def ensure_escalated_intents(
    dataclass, config, escalations_file, seed=42, n_generate=130, samples_per_intent=8
):
    """
    One intent per escalated utterance, named by the LLM itself (target_intent=None), as
    run_production_step does. escalations_file is a JSON array of utterances, in the order the
    increments happen. Cached so replay and full train on identical names and utterances.

    Returns {intent name: [utterances]}, in that order.
    """
    path = escalation_cache_path(escalations_file, seed)
    if os.path.exists(path):  # the second condition reuses the first one's data
        with open(path) as f:
            return json.load(f)

    with open(escalations_file) as f:
        escalations = json.load(f)

    cached = {}
    for utterance in escalations:
        name, utterances = generate_new_intent(
            dataclass=dataclass,
            escalated_utts=[utterance],
            target_intent=None,  # production mode: the LLM names the intent
            n_utterances=n_generate,
            config=config,
            samples_per_intent=samples_per_intent,
        )
        cached[name] = list(utterances)
        print(f"'{utterance}' -> '{name}', {len(utterances)} utterances")
        with open(
            path, "w"
        ) as f:  # written per intent so an interrupted generation resumes
            json.dump(cached, f, indent=2)
    return cached


def ensure_synthetic_utterances(
    dataclass, config, intents, seed=42, n_generate=130, samples_per_intent=8
):
    """
    {intent name: [utterances]} for the given intents, from the cache when it exists.

    Missing intents are generated with the real escalated utterance of that intent as the seed
    (the first reserve row), and target_intent pinned to its MASSIVE name so the class matches the
    real-data experiment. The cache is written after every intent, so an interrupted generation
    does not have to start over.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(seed)
    cached = {}
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)

    train_split = dataclass.sets_names[0]
    for intent_name in intents:
        if intent_name in cached:
            print(f"{intent_name}: {len(cached[intent_name])} cached utterances")
            continue

        seed_rows = dataclass.dataset_totrain[train_split].filter(
            lambda ex: ex[dataclass.label_col] == intent_name
        )
        assert len(seed_rows) > 0, f"no reserve rows for '{intent_name}'"
        escalated_utts = [seed_rows[0]["utt"]]

        _, utterances = generate_new_intent(
            dataclass=dataclass,
            escalated_utts=escalated_utts,
            target_intent=intent_name,  # pinned: the class must match the real-data experiment
            n_utterances=n_generate,
            config=config,
            samples_per_intent=samples_per_intent,
        )
        cached[intent_name] = list(utterances)
        with open(path, "w") as f:
            json.dump(cached, f, indent=2)
        print(
            f"{intent_name}: generated {len(utterances)} utterances"
            f" from '{escalated_utts[0]}' -> {path}"
        )

    return {name: cached[name] for name in intents}
