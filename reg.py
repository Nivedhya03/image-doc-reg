import os
import sys
import json
import torch
import spacy
import subprocess
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from datasets import Dataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report


def load_annotations(json_path):
    """Load JSON and ensure all entries have 'entities' key."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_data = []
    for item in data:
        text = item.get("text", "")
        entities = item.get("entities", [])
        formatted_data.append({"text": text, "entities": entities})

    return formatted_data


def convert_to_pickle_format(annotated_data, output_file="train_data.pkl"):
    """Convert dataset to spaCy format and save using Pickle."""
    nlp = spacy.blank("en")
    docs = []

    for item in annotated_data:
        text = item["text"]
        entities = item["entities"]

        doc = nlp.make_doc(text)
        ents = []
        seen_spans = []

        for ent in entities:
            start, end, label = ent["start"], ent["end"], ent["label"]
            span = doc.char_span(start, end, label=label)

            if span is not None:
                overlap = any(span.start < s.end and span.end > s.start for s in seen_spans)
                if not overlap:
                    ents.append(span)
                    seen_spans.append(span)

        doc.ents = ents
        docs.append(doc)

    with open(output_file, 'wb') as f:
        pickle.dump(docs, f)

    print(f"Data saved to {output_file}")


def train_spacy_model():
    """Train the spaCy model and return the trained pipeline."""
    subprocess.run([sys.executable, "-m", "spacy", "train", "config.cfg",
                    "--output", "./spacy_output", "--paths.train", "./train.spacy",
                    "--paths.dev", "./train.spacy"], capture_output=True, text=True)

    if os.path.exists("./spacy_output/model-best"):
        return spacy.load("./spacy_output/model-best")
    return None


def train_bert_model(annotated_data):
    """Train a BERT model for token classification with correct label mapping."""
    all_labels = set()
    for item in annotated_data:
        for ent in item["entities"]:
            all_labels.add(ent["label"])

    tag2id = {label: idx for idx, label in enumerate(sorted(all_labels), start=1)}
    tag2id["O"] = 0

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2id))

    return model, tokenizer, tag2id


def fine_tune_bert(model, tokenizer, tag2id, annotated_data, batch_size=16, epochs=3):
    """Fine-tune BERT with correct token-aligned labels."""

    input_ids, attention_masks, label_ids = [], [], []

    for item in annotated_data:
        text, entities = item["text"], item["entities"]
        encoding = tokenizer(text, padding='max_length', truncation=True, max_length=128, return_offsets_mapping=True)

        labels = ["O"] * len(encoding.input_ids)
        offsets = encoding.offset_mapping

        for ent in entities:
            start, end, label = ent["start"], ent["end"], ent["label"]

            for idx, (token_start, token_end) in enumerate(offsets):
                if token_start >= start and token_end <= end:
                    labels[idx] = label

        label_ids.append([tag2id[label] for label in labels])

        input_ids.append(encoding.input_ids)
        attention_masks.append(encoding.attention_mask)

    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)
    label_ids = torch.tensor(label_ids)

    train_dataset = TensorDataset(input_ids, attention_masks, label_ids)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            batch_input_ids, batch_attention_masks, batch_labels = batch
            batch_input_ids = batch_input_ids.to(model.device)
            batch_attention_masks = batch_attention_masks.to(model.device)
            batch_labels = batch_labels.to(model.device)

            outputs = model(input_ids=batch_input_ids, attention_mask=batch_attention_masks, labels=batch_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    model.save_pretrained("fine_tuned_bert")
    tokenizer.save_pretrained("fine_tuned_bert")

    return model, tokenizer


def evaluate_model_from_json(json_file, model, tokenizer, tag2id):
    """Evaluate the fine-tuned BERT model using a classification report."""
    with open(json_file, 'r') as file:
        data = json.load(file)

    id2tag = {v: k for k, v in tag2id.items()}

    y_true, y_pred = [], []

    for item in data:
        text = item["text"]
        true_labels = ["O"] * len(text)

        for ent in item["entities"]:
            start, end, label = ent["start"], ent["end"], ent["label"]
            for i in range(start, end):
                true_labels[i] = label

        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt").to(
            model.device)
        with torch.no_grad():
            outputs = model(**encoding)

        predictions = torch.argmax(outputs.logits, dim=2).squeeze().tolist()
        predicted_labels = [id2tag[idx] for idx in predictions]

        y_true.extend(true_labels)
        y_pred.extend(predicted_labels)

    print("\nBERT Model Performance:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    json_path = r"C:\Users\ACER\Desktop\pro 2\Entities.json"

    if not os.path.exists(json_path):
        print("❌ JSON dataset not found.")
        sys.exit(1)

    annotated_data = load_annotations(json_path)
    convert_to_pickle_format(annotated_data)

    with open("train_data.pkl", 'rb') as f:
        docs = pickle.load(f)

    nlp_spacy = train_spacy_model()
    model, tokenizer, tag2id = train_bert_model(annotated_data)

    model, tokenizer = fine_tune_bert(model, tokenizer, tag2id, annotated_data)

    evaluate_model_from_json(json_path, model, tokenizer, tag2id)
