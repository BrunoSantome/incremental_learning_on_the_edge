# This file will be use to run a complete end-to-end incremental loop over the whole system.
# A potential improvement would be to separate the logic of this sequential run on separate isolated
# environements: edge and server, using instead of a mailbox shared folder RESTAPI points. But that will be classified as improvements

# sequence:
# Edge predicts, ood detects, saves the utterance to escalate on a json, the server picks it, calls the LLM, synthetic data generation with replay buffer extension
# Re-training of the edge model on a new version with a new intent, evaluate old and new intents, save new model in folder, edge retrieves it, runs the same prediction and this
# time it will know the new intent added over training.

import os
import json
import shutil

from core.configuration import load_config
from core.dataloader import DataClass
from core.distillation_1 import run_production_step
from core.evaluate import (
    intents_report,
    old_intent_persistance_table,
    new_intent_acquisition_table,
)
from edge.inference import load_edge_model, predict
from edge.ood import is_ood
from shared.mailbox import (
    send_escalation,
    read_escalations,
    publish_release,
    poll_release,
    ESCALATIONS,
    RELEASES,
)
from dotenv import load_dotenv

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(SRC_DIR), ".env")) 


def _checkpoint_dir(config, student_key, version):
    base = os.path.join(SRC_DIR, config[student_key]["distill"]["output_dir"])
    return f"{base}_v{version}" #using specific naming convention for mailbox


def _registry_id2name(config):
    registry_path = os.path.join(SRC_DIR, config["registry_path"]) #retrieving registry_path.json
    with open(registry_path) as f:
        intent2id = json.load(f)["intent2id"]
    return {idx: name for name, idx in intent2id.items()} # the complete dictionary


def _v0_labels(config, n_pretrain=15):
    return {idx: name for idx, name in _registry_id2name(config).items() if idx < n_pretrain}

def _clear_registry_and_mailbox(config):
    # registry back to the 15 pretrain intents (DataClass rebuilds it) so it matches V0's head,
    # and an empty mailbox so no leftover escalation is read.
    registry_path = os.path.join(SRC_DIR, config["registry_path"])
    if os.path.exists(registry_path):
        os.remove(registry_path)

    for mailbox_dir in (ESCALATIONS, RELEASES):
        if os.path.exists(mailbox_dir):
            shutil.rmtree(mailbox_dir)



def run_edge_simulation_test():
    """
    This run experiment aims to simulate the behaviour of the edge device isolated. 
    Using the previous experiment of the end-to-end server isolated with the sepcific intent of
    car_window_control that the LLM generated, also using the checkpoint of the model re-trained on the 
    server side. To test the mailbox logic, of escalating, reading, publishing and polling
    """
    config = load_config()
    student_key = "student1"
    tokenizer_name = config[student_key]["name"]
    utterance = "could you open the window of the front of the car" # same utterance used in the server to generate the new model
    print(utterance)
    # edge starts at V0, only knowing the pretrain intents (NOT car_window_control, even
    # though the registry.json on disk already has 16 (it has in this test 16 already since we using the json from the 
    # server experiment)

    # ---- edge device ----

    v0_labels = _v0_labels(config)
    edge_version = 0
    model, tokenizer = load_edge_model(_checkpoint_dir(config, student_key, edge_version), tokenizer_name)

    name, confidence, probs = predict(model, tokenizer, v0_labels, utterance)
    print(f"[edge v{edge_version}] predicted: '{name}' with (confidence {confidence:.2f})")

    if not is_ood(probs):
        print("the edge device did not flagged as OOD: stopping (nothing to escalate)")
        return

    print("edge device OOD: escalating")
    send_escalation(utterance, edge_version)

    # ---- Server simulation without calling it ----
    # it reads it but does not call the LLM nor the re-training happens
    read_escalations()  # consume the pending escalation (mailbox contract); content unused by the stub
    v1_labels = _registry_id2name(config)  # we load the 16 registrys
    publish_release(
        version=1,
        checkpoint_dir=_checkpoint_dir(config, student_key, 1),
        labels=v1_labels,
    )
    print("The server simulation published v1 (car_window_control)")

    # ---- Edge devive ---- 

    release = poll_release(edge_version)
    edge_version = release["version"]
    model, tokenizer = load_edge_model(release["checkpoint_dir"], tokenizer_name)
    labels = {int(k): v for k, v in release["labels"].items()}  # JSON round trip -> string keys

    utterance2="could you close the rear-window please"
    print(utterance2)
    name, confidence, probs = predict(model, tokenizer, labels, utterance2)
    print(f"[edge v{edge_version}] predicted '{name}' (confidence {confidence:.2f})")


def run_production_iteration_simulation_test():
    """
    This run experiment aims to simulate the behaviour of the whole system end to end. 
    Merging the edge device logic and the server side logic together with the logical mailbox conector.
    On new utterances, that derive in new intents. 

    This function only runs a single iteration end-to-end of an incremental step with edge and server. 
    The objective is to re-use the same utterance "could you open the window of the front of the car" 
    so it is a more controlled test, and can be compared to the results got previously on the server side production test

    This is not the real production mechanism. 

    TODO: Still the ood is bad. 
    """
    config = load_config()
    student_key = "student1"
    tokenizer_name = config[student_key]["name"]
    utterance = "could you open the window of the front of the car" # same utterance used in the server to generate the new model
    print(utterance)

    # ---- edge device ----

    v0_labels = _v0_labels(config)
    edge_version = 0
    model, tokenizer = load_edge_model(_checkpoint_dir(config, student_key, edge_version), tokenizer_name)

    name, confidence, probs = predict(model, tokenizer, v0_labels, utterance)
    print(f"[edge v{edge_version}] predicted: '{name}' with (confidence {confidence:.2f})")

    if not is_ood(probs):
        print("the edge device did not flagged as OOD: stopping (nothing to escalate)")
        return

    print("edge device OOD: escalating")
    send_escalation(utterance, edge_version)

    # ---- server: REAL production step LLM generatpuion + retrain ----
    registry_path = os.path.join(SRC_DIR, config["registry_path"])
    if os.path.exists(registry_path):
        os.remove(registry_path)  # reset to 15 pretrain-only so DataClass() matches V0's head

    dataclass = DataClass()
    version = 1
    esc = read_escalations()[0]  # exactly one
    name, output_dir = run_production_step(dataclass, [esc["utterance"]], student_key, config, version, K=70, n_generate=130)
    if name is None:
        print("LLM named an existing intent: skipping, no new model")
        return
    publish_release(version, output_dir, dataclass.id2intent)
    print(f"server published v{version} ({name})")

    # ---- Edge devive ---- 

    release = poll_release(edge_version)
    edge_version = release["version"]
    model, tokenizer = load_edge_model(release["checkpoint_dir"], tokenizer_name)
    labels = {int(k): v for k, v in release["labels"].items()}  # JSON round trip -> string keys

    utterance2="could you close the rear-window please"
    print(utterance2)
    name, confidence, probs = predict(model, tokenizer, labels, utterance2)
    print(f"[edge v{edge_version}] predicted '{name}' (confidence {confidence:.2f})")


def run_interactive_simulation():
    """
    Final interactive end-to-end simulation of the system. Type an utterance in the terminal, if unknown it will be escalated
    , server generates data and retrains a new version, evaluates it, the edge "device" updates and predicts again.
    Intents accumulate across iterations (V1, V2, ...)

    The _v1+ checkpoints are not deleted: reset them manually before each run

    TODO: save the data of each iteration for in-depth analysis
    TODO: save/reset the state automatically instead of manually 
    TODO: save tokenizer into the checkpoint to not have cloud dependency on the edge model
    TODO: check if returning data for in-depth analysis rather than saving it every run
    """

    config = load_config()
    student_key = "student1"
    tokenizer_name = config[student_key]["name"]

    _clear_registry_and_mailbox(config)
    dataclass = DataClass()  # rebuilds the registry at the 15 pretrain intents
    version = 1
    edge_version = 0
    edge_labels = _v0_labels(config)
    model, tokenizer = load_edge_model(_checkpoint_dir(config, student_key, 0), tokenizer_name)

    while True:
        try:
            utterance = input("\nutterance (exit/quit to stop)> ").strip()
        except EOFError:  # stdin closed
            break
        if utterance.lower() in ("exit", "quit"):
            break
        if not utterance:
            continue

        # edge: inference + OOD 
        name, confidence, probs = predict(model, tokenizer, edge_labels, utterance)
        print(f"[edge v{edge_version}] predicted '{name}' (confidence {confidence:.2f})")
        if not is_ood(probs):
            continue

        print("[edge] OOD: escalating")
        send_escalation(utterance, edge_version)

        # server: generation + retrain
        esc = read_escalations()[0]  # exactly one
        intent_name, output_dir = run_production_step(
            dataclass, [esc["utterance"]], student_key, config, version, K=70, n_generate=130
        )
        if intent_name is None:
            print("server LLM named an existing intent: skipped")
            continue

        # server: cumulative evaluation V0..Vn
        rows, _ = intents_report(dataclass, student_key, config, n_versions=version)
        print(f"\nretention (old intents)\n{old_intent_persistance_table(rows, dataclass.id2intent)}")
        print(f"\nacquisition (new intents)\n{new_intent_acquisition_table(rows, dataclass.id2intent)}")

        publish_release(version, output_dir, dataclass.id2intent)
        print(f"server published v{version} ({intent_name})")

        # edge: update + re-predict the same utterance

        release = poll_release(edge_version)
        edge_version = release["version"]
        model, tokenizer = load_edge_model(release["checkpoint_dir"], tokenizer_name)
        edge_labels = {int(k): v for k, v in release["labels"].items()}  # JSON -> str keys

        name, confidence, probs = predict(model, tokenizer, edge_labels, utterance)
        print(f"[edge v{edge_version}] now predicts '{name}' (confidence {confidence:.2f})")
        version += 1



if __name__ == "__main__":
    # run_edge_simulation_test()

    """
    Results, the ood mechanism is not working properly, very simple it classifies it as weather_query with a confidence of 0.88
    the threshold was changed and set to 0.95 for the test. 
    
    The end-to-end loop is a success. It escalates, reads

    edge v0] predicted: 'weather_query' with (confidence 0.88)
    edge device OOD: escalating
    The server simulation published v1 (car_window_control)
    [edge v1] predicted 'car_window_control' (confidence 1.00)

    Results 2 difference utterance similar to the unknown after re-training

    could you open the window of the front of the car
    [edge v0] predicted: 'weather_query' with (confidence 0.88)
    edge device OOD: escalating
    The server simulation published v1 (car_window_control)
    could you close the rear-window please
    [edge v1] predicted 'car_window_control' (confidence 1.00)

    """

    # run_production_iteration_simulation_test()
    # run_interactive_simulation()
    # Need to run in collab due to lack of GPU in this computerç
    run_interactive_simulation()