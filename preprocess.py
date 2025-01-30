#Arnav Khanna
#arnavkha
import os
import sys
import re
from collections import Counter, defaultdict
from typing import List, Tuple

def removeSGML(text: str) -> str:
    return re.sub(r'<[^>]+>', '', text)

def tokenizeText(text: str) -> List[str]:
    tokenList = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        tokens = ["<start>"]  
        #check contractions 
        line = re.sub(r"([.!?])(\s|$)", r' \1 ', line)
        line = re.sub(r"'m\b", " am", line)
        line = re.sub(r"'s\b", " 's", line)
        line = re.sub(r"'re\b", " are", line)
        line = re.sub(r"'ve\b", " have", line)
        line = re.sub(r"'ll\b", " will", line)
        line = re.sub(r"n't\b", " not", line)
        line = re.sub(r"'d\b", " would", line)

        for word in line.split():
            if not word.strip():
                continue
            if re.match(r'^[\d,./-]+$', word) or re.match(r'^[A-Za-z]+\.[A-Za-z.]+$', word) or '-' in word:
                tokens.append(word)
                continue
            cleaned = re.sub(r'[^\w\s-]', '', word)
            if cleaned:
                tokens.append(cleaned)

        tokenList.extend(tokens)


    return tokenList

#helper func for debugging 
def train_BPE(tokens: List[str], vocabSize: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    vocab = {'<start>'}  
    merge_steps = []
    token_counts = defaultdict(int)

    for token in tokens:
        if token == '<start>':
            token_counts['<start>'] += 1 #treat as a separate char piazza@50
            continue
        split_chars = ' '.join(list(token)) 
        token_counts[split_chars] += 1
        vocab.update(token)

    while len(vocab) < vocabSize:
        #print("Merging", len(vocab))
        pair_counts = defaultdict(int)

        for word, freq in token_counts.items():
            if word == '<start>':
                continue
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pair_counts[pair] += freq
        if not pair_counts:
            break 

        best_pair = None
        max_freq = -1
        for pair, freq in pair_counts.items():
            if freq > max_freq:
                best_pair = pair
                max_freq = freq

        merge_steps.append(best_pair)

        merge_regex = re.escape(' '.join(best_pair))
        merged_form = ''.join(best_pair)
        vocab.add(merged_form)

        new_token_counts = defaultdict(int)
        for word, freq in token_counts.items():
            if word == '<start>':
                new_token_counts[word] += freq
                continue
            updated_word = re.sub(f'(?<!\\S){merge_regex}(?!\\S)', merged_form, word)
            new_token_counts[updated_word] += freq

        token_counts = new_token_counts

    return merge_steps

#helper func
def apply_merges(tokens: List[str], merge_steps: List[Tuple[str, str]]) -> List[str]:
    token_counts = defaultdict(int)

    for token in tokens:
        token_counts[token] += 1

    for best_pair in merge_steps:
        merge_regex = re.escape(' '.join(best_pair))
        merged_form = ''.join(best_pair)

        new_token_counts = defaultdict(int)
        for word, freq in token_counts.items():
            updated_word = re.sub(f'(?<!\\S){merge_regex}(?!\\S)', merged_form, word)
            new_token_counts[updated_word] += freq

        token_counts = new_token_counts

    final_tokens = []
    for word, freq in token_counts.items():
        if word != "<start>":
            tokenized_word = word.split()
        else:
            tokenized_word = [word]
        final_tokens.extend(tokenized_word * freq)

    return final_tokens


def BPE(tokens: List[str], vocabSize: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    merge_steps = train_BPE(tokens, vocabSize)  
    final_tokens = apply_merges(tokens, merge_steps) 
    return final_tokens, merge_steps


if __name__ == "__main__":
    folder_path = sys.argv[1]
    vocabSize = int(sys.argv[2])
    all_tokens = []
    for filename in os.listdir(folder_path):
        with open(os.path.join(folder_path, filename), 'r', encoding='ISO-8859-1') as file: #encoding needed 
            text = removeSGML(file.read())
            all_tokens.extend(tokenizeText(text))

    subword_tokens, merge_rules = BPE(all_tokens, vocabSize)
    token_frequencies = Counter(subword_tokens)

    #encoding needed
    #check formatting
    with open('preprocess.output', 'w', encoding='ISO-8859-1') as output_file:
        output_file.write(f"Tokens: {len(subword_tokens)}\n")
        output_file.write(f"Merge rules: {len(merge_rules)}\n")
        output_file.write("The first 20 merge rules:\n")
        for i, (tok1, tok2) in enumerate(merge_rules[:20]):
            output_file.write(f"({tok1}, {tok2}) -> {tok1+tok2}\n")
        output_file.write("Top 50 tokens:\n")
        for token, frequency in token_frequencies.most_common(50):
            output_file.write(f"{token} [{frequency}]\n")

    # preprocess.answers
    # DELETE BEFORE FINAL SUBMISSION
    # threshold = 0.25 * len(subword_tokens)
    # cumulative_count, min_unique_tokens = 0, 0
    # for token, count in token_frequencies.most_common():
    #     cumulative_count += count
    #     min_unique_tokens += 1
    #     if cumulative_count >= threshold:
    #         break

    # with open("preprocess.answers", "w", encoding="ISO-8859-1") as answer_file:
    #     answer_file.write(f"Total number of BPE tokens: {len(subword_tokens)}\n")
    #     answer_file.write(f"Total number of merge rules: {len(merge_rules)}\n")
    #     answer_file.write(f"Minimum number of unique BPE tokens accounting for 25% of total tokens: {min_unique_tokens}\n")
        
    # print("Reached end")
