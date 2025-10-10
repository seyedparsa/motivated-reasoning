import nltk
import random
import csv
from nltk.corpus import cmudict
from collections import defaultdict
import os

# Ensure you have CMUdict downloaded
nltk.download('cmudict')

# Load CMU Pronouncing Dictionary
pronouncing_dict = cmudict.dict()

def get_rhyming_part(pron):
    """Return the part of the pronunciation from the primary stress onward."""
    for i, p in enumerate(pron):
        if '1' in p:  # Primary stress
            return tuple(pron[i:])
    return tuple(pron)  # fallback if no stress

def build_rhyme_groups(pron_dict):
    """Group words by their rhyming parts."""
    rhyme_groups = defaultdict(list)
    for word, prons in pron_dict.items():
        for pron in prons:
            rhyme = get_rhyming_part(pron)
            rhyme_groups[rhyme].append(word)
    return rhyme_groups

def generate_positive_pairs(rhyme_groups):
    """Generate (word1, word2, label=1) pairs."""
    positive_pairs = []
    for words in rhyme_groups.values():
        if len(words) < 2:
            continue
        # Randomly pair words within each rhyme group
        random.shuffle(words)
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                positive_pairs.append((words[i], words[j], 1))
    return positive_pairs

def generate_negative_pairs(words, rhyme_groups, num_negatives):
    """Generate (word1, word2, label=0) pairs."""
    word_to_rhyme = {}
    for rhyme, rhymers in rhyme_groups.items():
        for w in rhymers:
            word_to_rhyme[w] = rhyme

    negative_pairs = []
    word_list = list(words)
    attempts = 0
    while len(negative_pairs) < num_negatives and attempts < 10 * num_negatives:
        w1, w2 = random.sample(word_list, 2)
        # Only accept if they are not from the same rhyme group
        if w1 in word_to_rhyme and w2 in word_to_rhyme:
            if word_to_rhyme[w1] != word_to_rhyme[w2]:
                negative_pairs.append((w1, w2, 0))
        attempts += 1
    return negative_pairs

def main(output_csv="rhyme_dataset.csv", max_pairs=10000):
    print("Building rhyme groups...")
    rhyme_groups = build_rhyme_groups(pronouncing_dict)

    print("Generating positive pairs...")
    positive_pairs = generate_positive_pairs(rhyme_groups)
    random.shuffle(positive_pairs)

    print("Generating negative pairs...")
    all_words = set(pronouncing_dict.keys())
    negative_pairs = generate_negative_pairs(all_words, rhyme_groups, len(positive_pairs))

    print(f"Positive examples: {len(positive_pairs)}")
    print(f"Negative examples: {len(negative_pairs)}")

    # Limit dataset size if needed
    total_pairs = positive_pairs[:max_pairs//2] + negative_pairs[:max_pairs//2]
    random.shuffle(total_pairs)

    print(f"Saving {len(total_pairs)} pairs to {output_csv}...")
    # os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["word1", "word2", "label"])  # header
        for w1, w2, label in total_pairs:
            writer.writerow([w1, w2, label])

    print("Done!")

if __name__ == "__main__":
    main()