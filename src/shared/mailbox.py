# "mailbox" so the edge and server can talk without importing each
# other. Edge drops escalations here; server drops model releases here. Plain JSON + a
# copied checkpoint dir

import os
import json
import time
import shutil


MAILBOX = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mailbox")
ESCALATIONS = os.path.join(MAILBOX, "escalations")
RELEASES = os.path.join(MAILBOX, "releases")


def _ensure_dirs():
    os.makedirs(ESCALATIONS, exist_ok=True)
    os.makedirs(RELEASES, exist_ok=True)


def send_escalation(utterance, edge_version):
    """
    Edge writes one escalation as escalations/<id>.json. id is the timestamp.
    """
    _ensure_dirs()
    eid = str(time.time_ns())
    payload = {"id": eid, "utterance": utterance, "edge_version": edge_version}
    path = os.path.join(ESCALATIONS, f"{eid}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    return eid


def read_escalations():
    """
    Server reads all pending escalations and consumes them (removes the files) so the
    same escalation is never handled twice. 
    TODO: Handle the case of multiple utterances being escalated, a priority order. 
    """
    _ensure_dirs()
    filenames = [f for f in os.listdir(ESCALATIONS) if f.endswith(".json")]
    escalations = []
    for filename in filenames:
        path = os.path.join(ESCALATIONS, filename)
        with open(path, "r") as f:
            escalations.append(json.load(f))
        os.remove(path)
    return escalations


def publish_release(version, checkpoint_dir, labels):
    """
    Server copies the trained checkpoint into releases/v{version}/ and writes
    releases/latest.json (the pointer the edge polls). labels is the index->name map,
    shipped here so the edge never has to import the registry.
    """
    _ensure_dirs()
    dest = os.path.join(RELEASES, f"v{version}")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(checkpoint_dir, dest)
    latest_path = os.path.join(RELEASES, "latest.json")
    with open(latest_path, "w") as f:
        json.dump({"version": version, "checkpoint_dir": dest, "labels": labels}, f)


def poll_release(current_version):
    """
    Edge checks for a newer release. Returns the release if releases/latest.json names a version newer than
    current_version, else None to prevent using as release the same previous version
    """
    _ensure_dirs()
    latest_path = os.path.join(RELEASES, "latest.json")
    if not os.path.exists(latest_path):
        return None
    with open(latest_path, "r") as f:
        release = json.load(f)
    if release["version"] > current_version:
        return release
    return None
