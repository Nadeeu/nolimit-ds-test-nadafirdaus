import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.utils.class_weight import compute_class_weight

LABEL2ID = {'anger': 0, 'fear': 1, 'happy': 2, 'love': 3, 'sadness': 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def compute_class_weights(labels) -> torch.Tensor:
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    return torch.tensor(weights, dtype=torch.float)


class WeightedTrainer(Trainer):
    def __init__(self, weights_tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_tensor = weights_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.weights_tensor.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def predict_emotion(text: str, tokenizer, model, max_length: int = 64) -> dict:
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=max_length
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id = int(probs.argmax())
    emotion = ID2LABEL[pred_id]
    confidence = float(probs[pred_id]) * 100
    all_probs = {ID2LABEL[i]: round(float(probs[i]) * 100, 2) for i in range(5)}

    return {
        'emotion': emotion,
        'confidence': confidence,
        'all_probs': all_probs
    }