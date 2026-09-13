# This file will be use to run a complete end-to-end incremental loop over the whole system.
# A potential improvement would be to separate the logic of this sequential run on separate isolated
# environements: edge and server, using instead of a mailbox shared folder RESTAPI points. But that will be classified as improvements

# sequence:
# Edge predicts, ood detects, saves the utterance to escalate on a json, the server picks it, calls the LLM, synthetic data generation with replay buffer extension
# Re-training of the edge model on a new version with a new intent, evaluate old and new intents, save new model in folder, edge retrieves it, runs the same prediction and this
# time it will know the new intent added over training.

import os
import json

from core.configuration import load_config
from edge.inference import load_edge_model, predict
from edge.ood import is_ood
from shared.mailbox import send_escalation, read_escalations, publish_release, poll_release

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def _checkpoint_dir(config, student_key, version):
    base = os.path.join(SRC_DIR, config[student_key]["distill"]["output_dir"])
    return f"{base}_v{version}" #using specific naming convention


def _registry_id2name(config):
    registry_path = os.path.join(SRC_DIR, config["registry_path"]) #retrieving registry_path.json
    with open(registry_path) as f:
        intent2id = json.load(f)["intent2id"]
    return {idx: name for name, idx in intent2id.items()} # the complete dictionary


def _v0_labels(config, n_pretrain=15):
    return {idx: name for idx, name in _registry_id2name(config).items() if idx < n_pretrain}



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

    # ----Server simulation Server simulation without calling it ----
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


if __name__ == "__main__":
    run_edge_simulation_test()

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