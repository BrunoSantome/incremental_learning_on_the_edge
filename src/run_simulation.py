# This file will be use to run a complete end-to-end incremental loop over the whole system. 
# A potential improvement would be to separate the logic of this sequential run on separate isolated
# environements: edge and server, using instead of a mailbox shared folder RESTAPI points. But that will be classified as improvements

# sequence: 
# Edge predicts, ood detects, saves the utterance to escalate on a json, the server picks it, calls the LLM, synthetic data generation with replay buffer extension
# Re-training of the edge model on a new version with a new intent, evaluate old and new intents, save new model in folder, edge retrieves it, runs the same prediction and this
# time it will know the new intent added over training. 