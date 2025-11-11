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
        len(word),  # word length
        vowel_count,  # number of vowels
        consonant_count, # number of consonants
        vowel_count / len(word) if len(word) > 0 else 0,  # vowel ratio
        1 if len(word) > 0 and word[0].isupper() else 0,  # starts with capital

        # Filipino-specific
        1 if 'ng' in word_lower else 0,  # contains 'ng' 
        word_lower.count('ng'), # frequency of 'ng'

        # Filipino prefix
        1 if word_lower.startswith('nag') else 0,  # starts with 'nag' 
        1 if word_lower.startswith('mag') else 0,  # starts with 'mag' 
        1 if word_lower.startswith('pag') else 0,  # starts with 'pag' 
        1 if word_lower.startswith('naka') else 0,  # starts with 'naka'
        1 if word_lower.startswith('ma') else 0,  # starts with 'ma' 
        1 if word_lower.startswith('ka') else 0,  # starts with 'ka' 
        1 if word_lower.startswith('pa') else 0,  # starts with 'pa' 
        1 if word_lower.startswith('i') and len(word) > 2 else 0,  # starts with 'i'

        # Filipino suffix
        1 if word_lower.endswith('an') else 0,  #  ends with 'an' 
        1 if word_lower.endswith('in') else 0,  # ends with 'in' 
        1 if word_lower.endswith('ay') else 0,  # ends with 'ay' 
        1 if word_lower.endswith('han') else 0,  # ends with 'han' 

        # Common Filipino words
        1 if word_lower in ['ng', 'mga', 'na', 'sa', 'ang', 'ko', 'mo', 'ka', 'po', 'ni', 'si', 'ay'] else 0,  
        1 if word_lower in ['ako', 'ikaw', 'siya', 'kami', 'tayo', 'kayo', 'sila'] else 0,  # filo pronouns
        1 if word_lower in ['at', 'o', 'pero', 'kasi', 'kung', 'dahil'] else 0,  # filo conjunctions

        # English suffix
        1 if word_lower.endswith('ing') else 0,  # ends with 'ing'
        1 if word_lower.endswith('ed') else 0,  # ends with 'ed'
        1 if word_lower.endswith('ly') else 0,  # ends with 'ly'
        1 if word_lower.endswith('tion') else 0,  # ends with 'tion'
        1 if word_lower.endswith('ness') else 0,  # ends with 'ness'
        1 if word_lower.endswith('ment') else 0,  # ends with 'ment'
        1 if word_lower.endswith('er') else 0,  # ends with 'er'
        1 if word_lower.endswith('est') else 0,  # ends with 'est'
        1 if word_lower.endswith('ful') else 0,  # ends with 'ful'
        1 if word_lower.endswith('less') else 0,  # ends with 'less'

        # English prefix
        1 if word_lower.startswith('un') else 0,  # starts with 'un'
        1 if word_lower.startswith('re') else 0,  # starts with 're'
        1 if word_lower.startswith('pre') else 0,  # starts with 'pre'
        1 if word_lower.startswith('dis') else 0,  # starts with 'dis'
        1 if word_lower.startswith('mis') else 0,  # starts with 'mis'
        1 if word_lower.startswith('over') else 0,  # starts with 'over' 
        1 if word_lower.startswith('under') else 0,  # starts with 'under' 
        1 if word_lower.startswith('ex') else 0,  # starts with 'ex' 

        # Common English words
        1 if word_lower in ['the', 'a', 'an', 'this', 'that', 'these', 'those'] else 0,  # articles
        1 if word_lower in ['is', 'am', 'are', 'was', 'were', 'be', 'been', 'being'] else 0,  # copula
        1 if word_lower in ['have', 'has', 'had', 'do', 'does', 'did'] else 0,  # auxiliaries
        1 if word_lower in ['and', 'or', 'but', 'so', 'if', 'when', 'because'] else 0,  # conjunctions
        1 if word_lower in ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'] else 0,  # pronouns
        1 if word_lower in ['my', 'your', 'his', 'her', 'its', 'our', 'their'] else 0,  # possessives
        1 if word_lower in ['can', 'will', 'would', 'should', 'could', 'may', 'might', 'must'] else 0,  # modals
       
        # Symbol detection
        1 if not word.isalpha() else 0,  # contains non-alphabetic characters
        1 if word in ['.', ',', '!', '?', ';', ':', '-', '"', "'", '(', ')'] else 0,  # pure punctuation
        1 if any(c.isdigit() for c in word) else 0,  # contains digits

        # Word shape
        1 if word.isupper() else 0,  # all caps
        1 if word.islower() else 0,  # all lowercase
        1 if word.istitle() else 0,  # title case
    ]

    return features