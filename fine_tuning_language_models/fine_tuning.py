from datasets import load_dataset
emotions = load_dataset("emotion")
#https://huggingface.co/datasets/emotion
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)


from transformers import AutoModelForSequenceClassification
num_labels = 6
model = (AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device))

print(emotions_encoded["train"].features)

emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
print(emotions_encoded["train"].features)


from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}




from transformers import Trainer, TrainingArguments

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size
training_args = TrainingArguments(output_dir=r"E:\Projects\emo_detector_new\results",
                                  num_train_epochs=8,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="f1",
                                  weight_decay=0.01,
                                  evaluation_strategy="epoch",
                                  disable_tqdm=False,
                                  logging_steps=logging_steps,)


from transformers import Trainer

trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"])
trainer.train();

results = trainer.evaluate()
print(results)
model.save_pretrained('./emo_bert_model')
tokenizer.save_pretrained('./emo_bert_model')
preds_output = trainer.predict(emotions_encoded["validation"])
print(preds_output.metrics)

import numpy as np
from sklearn.metrics import plot_confusion_matrix
y_valid = np.array(emotions_encoded["validation"]["label"])
y_preds = np.argmax(preds_output.predictions, axis=1)
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
plot_confusion_matrix(y_preds, y_valid, labels)



model.save_pretrained('./model')
tokenizer.save_pretrained('./model')