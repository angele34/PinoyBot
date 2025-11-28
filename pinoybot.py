"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List
from features import extract_features  # Import from our new features.py file

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.

    Args:
        tokens: List of word tokens (strings).

    Returns:
        tags: List of predicted tags ("ENG", "FIL", or "OTH"), one per token.
    """
    # 1. Load the trained model
    with open('language_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # 2. Extract features for each token (using same function as training!)
    X = [extract_features(token) for token in tokens]

    # 3. Predict tags
    predictions = model.predict(X)

    # 4. Convert to list of strings
    tags = [str(tag) for tag in predictions]

    return tags

from pinoybot import tag_language

def label_file(input_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    for i, line in enumerate(lines):
        print(f"\n=== Sentence {i+1} ===")

        # Each sentence is tokenized using "|"
        tokens = line.split("|")

        # Tag each token
        tags = tag_language(tokens)

        # Display result
        for token, tag in zip(tokens, tags):
            print(f"{token}: {tag}")

if __name__ == "__main__":
    label_file("test_data.txt")
