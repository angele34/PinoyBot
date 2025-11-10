# file contains the feature extraction function used by both training (dataprep.py) and prediction (pinoybot.py).

def extract_features(word):
    """
    extracts multiple features from a word to help with identifying its language

    @param word:    word to extract features from
    @return:        list of features
    """
    word_lower = word.lower()

    # count vowels
    vowels = 'aeiou'
    vowel_count = sum(1 for c in word_lower if c in vowels)

    # build feature list
    features = [
        len(word),  # feature 1: word length
        vowel_count,  # feature 2: number of vowels
        1 if len(word) > 0 and word[0].isupper() else 0,  # feature 3: starts with capital
        1 if 'ng' in word_lower else 0,  # feature 4: contains 'ng' (Filipino indicator)
        1 if word_lower.startswith('nag') else 0,  # feature 5: starts with 'nag' (Filipino prefix)
        1 if word_lower.endswith('ing') else 0,  # feature 6: ends with 'ing' (English suffix)
    ]

    return features