from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/distilbert-base-uncased-emotion")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\results\checkpoint-2000")
texts = ["fuck you bitch", "I like you. I love you"]
tokenized_texts = tokenizer(texts, padding=True)


class SimpleDataset:
    def __init__(self, tokenized_texts):
        self.tokenized_texts = tokenized_texts

    def __len__(self):
        return len(self.tokenized_texts["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tokenized_texts.items()}


test_dataset = SimpleDataset(tokenized_texts)

import numpy as np
import tensorflow as tf
trainer = Trainer(model=model)
preds_output = trainer.predict(test_dataset)
# print(predictions)
y_preds = np.argmax(preds_output.predictions, axis=1)
# tf_prediction = predictions.predictions.argmax(axis=0)
print(y_preds)