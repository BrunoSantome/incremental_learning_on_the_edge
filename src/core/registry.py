import json
import os


class IntentRegistry:
    """
    Approach (A) for incremental learning: the pretrain intents occupy the first
    indices (0..N-1), assigned deterministically. Every intent learned later is
    appended to the next free slot

    """

    def __init__(self, intent2id=None, path=None):
        self.intent2id = dict(intent2id) if intent2id else {}
        self.path = path

    @classmethod
    def build_pretrain(cls, intent_names, path=None):
        """Assign deterministic indices 0..N-1 to the pretrain intents."""
        ordered = sorted(intent_names)  # sorted => reproducible across runs
        intent2id = {name: i for i, name in enumerate(ordered)}
        registry = cls(intent2id, path=path)
        if path is not None:
            registry.save()
        return registry

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data["intent2id"], path=path)

    @classmethod
    def get_or_create(cls, pretrain_names, path):
        """
        Load an existing registry (preserving any intents already appended by a
        previous incremental step), or build a fresh one from the pretrain set.
        If a stored registry is missing some pretrain intents they are appended.
        """
        if path and os.path.exists(path):
            registry = cls.load(path)
            for name in sorted(pretrain_names):
                registry.add_intent(name)  # no-op if already present
            return registry
        return cls.build_pretrain(pretrain_names, path=path)

    def save(self, path=None):
        path = path or self.path
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"intent2id": self.intent2id}, f, indent=2)

    def add_intent(self, name):
        """Append a new intent to the next free slot. Idempotent; persists."""
        if name not in self.intent2id:
            self.intent2id[name] = self.num_intents
            if self.path is not None:
                self.save()
        return self.intent2id[name]

    @property
    def id2intent(self):
        return {i: name for name, i in self.intent2id.items()}

    @property
    def num_intents(self):
        return len(self.intent2id)

    def __getitem__(self, name):
        return self.intent2id[name]

    def __contains__(self, name):
        return name in self.intent2id

    def __len__(self):
        return self.num_intents
