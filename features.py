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

    # count consonants
    consonant_count = sum(1 for c in word_lower if c.isalpha() and c not in vowels)

    # build feature list
    features = [
        len(word),  # feature 1: word length
        vowel_count,  # feature 2: number of vowels
        1 if len(word) > 0 and word[0].isupper() else 0,  # feature 3: starts with capital
        1 if 'ng' in word_lower else 0,  # feature 4: contains 'ng' (Filipino indicator)
        1 if word_lower.startswith('nag') else 0,  # feature 5: starts with 'nag' (Filipino prefix)
        1 if word_lower.endswith('ing') else 0,  # feature 6: ends with 'ing' (English suffix)
        1 if word_lower.endswith('an') else 0,  # feature 8: ends with 'an' (Filipino suffix)
        1 if word_lower.startswith('mag') else 0,  # feature 9: starts with 'mag' (Filipino prefix)
        1 if word_lower.startswith('pag') else 0,  # feature 10: starts with 'pag' (Filipino prefix)
        1 if word_lower.startswith('naka') else 0,  # feature 11: starts with 'naka' (Filipino prefix)
        # Filipino-specific features
        1 if word_lower.endswith('ay') else 0,  # feature 12: Filipino particle
        1 if word_lower in ['ng', 'mga', 'na', 'sa', 'ang', 'ko', 'mo', 'ka', 'ko'] else 0,  # feature 13: common Filipino words
        1 if word_lower.startswith('ka') else 0,  # feature 14: Filipino prefix
        # English-specific features
        1 if word_lower.endswith('ed') else 0,  # feature 15: English past tense
        1 if word_lower.endswith('ly') else 0,  # feature 16: English adverb
        1 if word_lower.startswith('un') else 0,  # feature 17: English prefix
        # Symbol detection
        1 if not word.isalpha() else 0,  # feature 18: contains non-alphabetic (punctuation/symbols)
        1 if word in ['.', ',', '!', '?', ';', ':', '-'] else 0,  # feature 19: pure punctuation
    ]

    return features