import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from pinoybot import tag_language

# 1. Load test data
with open("test/test_data.txt", "r", encoding="utf8") as f:
    sentences = [line.strip() for line in f]

# 2. Load test labels
with open("test/test_labels.txt", "r", encoding="utf8") as f:
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

# 4. Print classification report
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(classification_report(all_gold, all_preds, target_names=['ENG', 'FIL', 'OTH'], digits=4))

# 5. Compute TP, FP, FN per category
labels = ['ENG', 'FIL', 'OTH']
cm = confusion_matrix(all_gold, all_preds, labels=labels)

print("\n" + "=" * 60)
print("PER-CATEGORY METRICS (TP, FP, FN)")
print("=" * 60)

for i, label in enumerate(labels):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{label}:")
    print(f"  True Positives (TP):     {TP}")
    print(f"  False Positives (FP):    {FP}")
    print(f"  False Negatives (FN):    {FN}")
    print(f"  Precision:               {precision:.4f}")
    print(f"  Recall:                  {recall:.4f}")
    print(f"  F1-score:                {f1:.4f}")

print("\n" + "=" * 60)
