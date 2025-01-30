import sys
import os
import math

def trainBigramLanguageModel(training_text):
    unigram_frequencies = {}
    bigram_frequencies = {}
    for i in range(len(training_text)):
        char = training_text[i]
        unigram_frequencies[char] = unigram_frequencies.get(char, 0) + 1
        
        if i < len(training_text) - 1:
            bigram = training_text[i:i+2]
            bigram_frequencies[bigram] = bigram_frequencies.get(bigram, 0) + 1

    return unigram_frequencies, bigram_frequencies

def identifyLanguage(test_text, languages, unigram_dicts, bigram_dicts):
    best_language = None
    best_log_prob = float('-inf') #floor
    
    for lang_index, lang in enumerate(languages):
        unigram_counts = unigram_dicts[lang_index]
        bigram_counts = bigram_dicts[lang_index]
        vocab_size = len(unigram_counts) 

        log_prob = 0
        for i in range(len(test_text) - 1):
            bigram = test_text[i:i+2]
            unigram = test_text[i]

            bigram_freq = bigram_counts.get(bigram, 0) + 1 #add-one smoothin
            unigram_freq = unigram_counts.get(unigram, 0) + vocab_size
            log_prob += math.log(bigram_freq / unigram_freq) 

        if log_prob > best_log_prob:
            best_log_prob = log_prob
            best_language = lang
            #print(best_language)

    return best_language

if __name__ == "__main__":
    training_folder = sys.argv[1]
    test_file = sys.argv[2]

    languages = []
    unigram_dicts = []
    bigram_dicts = []

    for filename in os.listdir(training_folder):
        training_file_path = os.path.join(training_folder, filename)
        language_name = os.path.splitext(filename)[0] 
        languages.append(language_name)

        with open(training_file_path, 'r', encoding='ISO-8859-1') as f:
            training_text = f.read().replace("\n", " ")

        unigram_freq, bigram_freq = trainBigramLanguageModel(training_text)
        unigram_dicts.append(unigram_freq)
        bigram_dicts.append(bigram_freq)

    with open(test_file, 'r', encoding='ISO-8859-1') as test_file, open("languageIdentification.output", 'w', encoding='ISO-8859-1') as output_file:
        for line_number, line in enumerate(test_file, start=1):
            line = line.strip()
            predicted_language = identifyLanguage(line, languages, unigram_dicts, bigram_dicts)
            output_file.write(f"{line_number} {predicted_language}\n")
    #print("reached")

