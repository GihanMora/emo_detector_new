from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(r"E:\Projects\emo_detector_new\go_model")
model = AutoModel.from_pretrained(r"E:\Projects\emo_detector_new\go_model")
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
predictions = trainer.predict(test_dataset)
print(np.shape(predictions.predictions))
y_preds = np.argmax(predictions.predictions, axis=1)
print(y_preds)
tf_prediction = predictions.predictions.argmax(axis=1)
print(tf_prediction)