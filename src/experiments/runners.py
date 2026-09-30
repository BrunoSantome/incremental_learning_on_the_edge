import os

from core.dataloader import DataClass
from core.distillation_1 import run_incremental_step
from server.paraphrase import generate_new_intent


def run_incremental_experiment(
    intents_to_add, student_key, config, K, seed=42, exp_name=None
):
    """
    This method is only to be used for the experimental phase.
    It is thought to run on the incremental intents saved on the to-train set to test
    the incremental pipeline before adding the LLM synthetic data generation.
    Every time you run this method it starts from the same V0 version of the model, it can increment
    up to 5 new intents.

    All versions of one run share a wandb group: exp_id.
    Different experiments (different K/alpha/T/seed) stay separable.
    """
    distill_cfg = config[student_key]["distill"]
    exp_id = (
        exp_name  # if exp_name is False use the default constructed
        or f"K{K}_a{distill_cfg['alpha']}_T{distill_cfg['temperature']}_s{seed}"
    )

    registry_path = DataClass._resolve_path(
        config["registry_path"]
    )  # refresh of this file needed at every new experiment
    if os.path.exists(registry_path):
        os.remove(registry_path)

    dataclass = (
        DataClass()
    )  # creates new instance of intent_registry.json at 15 intents
    for version, intent_name in enumerate(intents_to_add, start=1):
        print(f"v{version}: adding intent: {intent_name}")
        run_incremental_step(
            dataclass,
            intent_name,
            student_key,
            config,
            version,
            K,
            seed,
            wandb_group=exp_id,
            wandb_tags=[f"K{K}", f"seed{seed}", intent_name],
        )
    return dataclass  # we return the dataclass to perform a good evaluation on the resulting dataset


def run_synthetic_incremental_experiment(
    intents_to_add,
    student_key,
    config,
    K,
    n_generate,
    seed=42,
    exp_name=None,
    samples_per_intent=8,
):
    """
    Synthetic counterpart of run_incremental_experiment.

    For each reserve intent, the new intent's TRAINING data is LLM-generated instead of
    real MASSIVE data; eval/test stay real. Old intents keep their real replay exemplars,
    so the ONLY variable vs the real-data experiment is the new intent's data source —
    a clean controlled comparison, both evaluated on the real test split.

    """

    distill_cfg = config[student_key]["distill"]
    exp_id = (
        exp_name
        or f"SYNTH_K{K}_g{n_generate}_a{distill_cfg['alpha']}_T{distill_cfg['temperature']}_s{seed}"
    )

    # reset the registry to the 15 pretrain intents so the chain starts from V0's label space
    registry_path = DataClass._resolve_path(config["registry_path"])
    if os.path.exists(registry_path):
        os.remove(registry_path)

    dataclass = DataClass()
    train_split = dataclass.sets_names[0]

    for version, intent_name in enumerate(intents_to_add, start=1):
        print(f"v{version}: generating + adding synthetic intent: {intent_name}")

        # real escalated seed utterance for this intent, taken from the reserve (to-train) data
        seed_rows = dataclass.dataset_totrain[train_split].filter(
            lambda ex: ex[dataclass.label_col] == intent_name
        )
        escalated_utts = [seed_rows[0]["utt"]]

        # LLM-generate the synthetic training utterances;
        # before generating we pass the context of previously know utterances.

        _, utterances = generate_new_intent(
            dataclass=dataclass,
            escalated_utts=escalated_utts,
            target_intent=intent_name,  # pin the name so it matches real MASSIVE for comparison
            n_utterances=n_generate,
            config=config,
            samples_per_intent=samples_per_intent,
        )

        # synthetic train + real eval/test, then the SAME incremental step as the real experiment
        new_utt = dataclass.build_llm_new_utt(intent_name, utterances)
        run_incremental_step(
            dataclass,
            intent_name,
            student_key,
            config,
            version,
            K,
            seed,
            new_utt=new_utt,
            wandb_group=exp_id,
            wandb_tags=[f"K{K}", "synthetic", f"seed{seed}", intent_name],
        )

    # return the dataclass so the same Step-6 tables (real test) can be run for comparison
    return dataclass


def run_production_incremental_experiment(
    intents_to_add,
    student_key,
    config,
    K,
    n_generate,
    seed=42,
    exp_name=None,
    samples_per_intent=8,
    n_eval=20,
    n_test=20,
):
    """
    train/eval/test are all generated and split from the LLM utterances (build_synthetic_splits). Old
    intents keep their real replay exemplars.

    This is to perform the dual evaluation
    (synthetic-test vs real-test) of one and the same model.

    """

    distill_cfg = config[student_key]["distill"]
    exp_id = (
        exp_name
        or f"PROD_K{K}_g{n_generate}_a{distill_cfg['alpha']}_T{distill_cfg['temperature']}_s{seed}"
    )

    # reset the registry to the 15 pretrain intents so the chain starts from V0's label space
    registry_path = DataClass._resolve_path(config["registry_path"])
    if os.path.exists(registry_path):
        os.remove(registry_path)

    dataclass = DataClass()
    train_split = dataclass.sets_names[0]

    for version, intent_name in enumerate(intents_to_add, start=1):
        print(f"v{version}: generating all-synthetic intent: {intent_name}")

        # real escalated seed utterance for this intent, taken from the reserve (to-train) data
        seed_rows = dataclass.dataset_totrain[train_split].filter(
            lambda ex: ex[dataclass.label_col] == intent_name
        )
        escalated_utts = [seed_rows[0]["utt"]]

        _, utterances = generate_new_intent(
            dataclass=dataclass,
            escalated_utts=escalated_utts,
            target_intent=intent_name,  # pinned so real-test dual eval stays possible
            n_utterances=n_generate,
            config=config,
            samples_per_intent=samples_per_intent,
        )

        # fully-synthetic train + eval + test, then the SAME incremental step
        new_utt = dataclass.build_synthetic_splits(
            intent_name, utterances, n_eval, n_test, seed
        )
        run_incremental_step(
            dataclass,
            intent_name,
            student_key,
            config,
            version,
            K,
            seed,
            new_utt=new_utt,
            wandb_group=exp_id,
            wandb_tags=[f"K{K}", "production", f"seed{seed}", intent_name],
        )

    # return the dataclass so intents_report can be run twice (synthetic test + real test)
    return dataclass
