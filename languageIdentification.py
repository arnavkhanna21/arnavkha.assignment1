import sys
import os
import math

def trainBigramLanguageModel(training_text):
    """Trains a bigram language model with add-one smoothing.

    Args:
        training_text (str): Training text in a given language.

    Returns:
        tuple: (unigram_frequencies, bigram_frequencies)
    """
    unigram_frequencies = {}
    bigram_frequencies = {}
    
    # Count unigrams and bigrams
    for i in range(len(training_text)):
        char = training_text[i]
        unigram_frequencies[char] = unigram_frequencies.get(char, 0) + 1
        
        if i < len(training_text) - 1:
            bigram = training_text[i:i+2]
            bigram_frequencies[bigram] = bigram_frequencies.get(bigram, 0) + 1

    return unigram_frequencies, bigram_frequencies

def identifyLanguage(test_text, languages, unigram_dicts, bigram_dicts):
    """Identifies the most likely language for a given test string.

    Args:
        test_text (str): The test text to be classified.
        languages (list): List of language names.
        unigram_dicts (list): List of unigram frequency dictionaries.
        bigram_dicts (list): List of bigram frequency dictionaries.

    Returns:
        str: The most likely language.
    """
    best_language = None
    best_log_prob = float('-inf')
    
    for lang_index, lang in enumerate(languages):
        unigram_counts = unigram_dicts[lang_index]
        bigram_counts = bigram_dicts[lang_index]
        
        # Compute vocabulary size
        vocab_size = len(unigram_counts)  # Number of unique characters in the language corpus

        # Compute bigram probability with add-one smoothing
        log_prob = 0
        for i in range(len(test_text) - 1):
            bigram = test_text[i:i+2]
            unigram = test_text[i]

            # Apply add-one smoothing
            bigram_freq = bigram_counts.get(bigram, 0) + 1
            unigram_freq = unigram_counts.get(unigram, 0) + vocab_size

            log_prob += math.log(bigram_freq / unigram_freq)  # Log probability to avoid underflow

        # Update best language
        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_language = lang

    return best_language

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 languageIdentification.py [path to training data folder] [test file]")
        sys.exit(1)

    training_folder = sys.argv[1]
    test_file_path = sys.argv[2]

    languages = []
    unigram_dicts = []
    bigram_dicts = []

    # Train language models
    for filename in os.listdir(training_folder):
        training_file_path = os.path.join(training_folder, filename)

        if os.path.isfile(training_file_path):  # Ensure it's a file
            language_name = os.path.splitext(filename)[0]  # Extract language name from file name
            languages.append(language_name)

            with open(training_file_path, 'r', encoding='ISO-8859-1') as f:
                training_text = f.read().replace("\n", " ")  # Replace newlines with spaces

            unigram_freq, bigram_freq = trainBigramLanguageModel(training_text)
            unigram_dicts.append(unigram_freq)
            bigram_dicts.append(bigram_freq)

    # Process test file
    with open(test_file_path, 'r', encoding='ISO-8859-1') as test_file, open("languageIdentification.output", 'w', encoding='ISO-8859-1') as output_file:
        for line_number, line in enumerate(test_file, start=1):
            line = line.strip()
            if line:  # Skip empty lines
                predicted_language = identifyLanguage(line, languages, unigram_dicts, bigram_dicts)
                output_file.write(f"{line_number} {predicted_language}\n")

if __name__ == "__main__":
    main()
