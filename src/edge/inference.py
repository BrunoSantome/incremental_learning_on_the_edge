#load the edge model once and predict an intent for one utterance.
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_edge_model(checkpoint_dir, tokenizer_name, device="cpu"):
    # Function that loads the model and the tokenizer in the edge device
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()  # inference mode: disables dropout, deterministic outputs
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return model, tokenizer

def predict(model, tokenizer, labels, utterance):
    # labels: {int index -> intent name}, matching the model's current head
    device = next(model.parameters()).device
    inputs = tokenizer(
        utterance, truncation=True, max_length=128, padding=True, return_tensors="pt" # important returning a pytorch tensor
    ).to(device)

    with torch.no_grad(): # no gradient tracking to save memory and time
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze(0) # turning the logits into a probability per intent
    pred_idx = int(torch.argmax(probs).item()) # picks the index the index with the highes probability TODO: check Unknow detection ood

    name = labels[pred_idx] # get the intent name of the prediction
    confidence = float(probs[pred_idx].item())
    return name, confidence, probs.tolist()  # ood needs to track the whole probability distribution rather than the single predicted intent