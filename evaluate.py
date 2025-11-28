import pandas as pd
from sklearn.metrics import classification_report
from pinoybot import tag_language

# 1. Load test data
with open("test_data.txt", "r", encoding="utf8") as f:
    sentences = [line.strip() for line in f]

# 2. Load test labels
with open("test_labels.txt", "r", encoding="utf8") as f:
    gold = [line.strip() for line in f]

# Convert labels from "A|B|C" → ["A","B","C"]
gold_labels = [row.split("|") for row in gold]

all_preds = []
all_gold = []

# 3. Predict each sentence
for sent, gold_row in zip(sentences, gold_labels):
    tokens = sent.split("|")
    preds = tag_language(tokens)

    all_preds.extend(preds)
    all_gold.extend(gold_row)

# 4. Print evaluation
print(classification_report(all_gold, all_preds))
