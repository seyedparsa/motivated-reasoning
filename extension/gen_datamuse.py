import requests
import random
import time
import csv
import nltk

from nltk.corpus import words as nltk_words
nltk.download('words')
english_vocab = set(nltk_words.words())

def is_good_word(w):
    return (
        w.isalpha() and
        len(w) > 2 and
        w.lower() in english_vocab
    )

def get_cluster(word, rel, min_freq):
    # print(f"Getting {rel} for '{word}'...")
    if rel == 'rhymes':
        rel_code = 'rel_rhy'
    elif rel == 'means':
        rel_code = 'rel_trg'
    url = f"https://api.datamuse.com/words?{rel_code}={word}&md=f"
    response = requests.get(url)
    results = response.json()
    cluster = []
    for entry in results:
        if not is_good_word(entry['word']):
            continue
        freq = 0
        # print(entry)
        for tag in entry.get('tags', []):
            if tag.startswith('f:'):
                freq = float(tag[2:])
        if freq >= min_freq:
            cluster.append(entry['word'])
            # print(f"Found {rel}: {entry['word']} with frequency {freq}")
    # input()
    return cluster


def build_dataset(anchor_words, size, delay=1.0):
    dataset = []
    rhymes_dics = {}
    means_dics = {}

    for word in anchor_words:
        print(f"Processing '{word}'...")
        rhymes = get_cluster(word, 'rhymes', min_freq=5)
        means = get_cluster(word, 'means', min_freq=5)
        print(f"Found {len(rhymes)} rhymes for '{word}'")
        print(f"Found {len(means)} meanings for '{word}'")
        # print(f"Rhymes: {rhymes}")
        print()
        for other_word in anchor_words:
            if other_word == word:
                continue
            if other_word in rhymes and other_word in means:
                anchor_words.remove(other_word)
                print(f"Found rhyme/meaning in the anchor words: {word} - {other_word}")
        rhymes_dics[word] = rhymes
        means_dics[word] = means

    pairs = set()

    # for _ in range(size):        
    #     found = False
    #     while not found:
    #         label = random.choice([0, 1])
    #         if label == 1:
    #             anchor = random.choice(anchor_words)
    #             word1, word2 = random.choices(rhymes_dics[anchor], k=2)
    #         else:            
    #             anchor1, anchor2 = random.sample(anchor_words, 2)
    #             word1, word2 = random.choice(rhymes_dics[anchor1]), random.choice(rhymes_dics[anchor2])
    #         if word1 != word2 and (word1, word2) not in pairs and (word2, word1) not in pairs:
    #             pairs.add((word1, word2))
    #             found = True
    #     dataset.append((word1, word2, label))

    for _ in range(size):        
        found = False
        while not found:
            label = random.choice([0, 1])
            anchor = random.choice(anchor_words)
            if label == 1:
                word1, word2 = random.choices(rhymes_dics[anchor], k=2)
            else:            
                word1, word2 = random.choices(means_dics[anchor], k=2)                
            if word1 != word2 and (word1, word2) not in pairs and (word2, word1) not in pairs:
                pairs.add((word1, word2))
                found = True
        dataset.append((word1, word2, label))

    return dataset


def save_dataset(dataset, filename="rhyme_dataset_muse.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['word1', 'word2', 'label'])
        writer.writerows(dataset)


def load_anchor_words(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    return words


if __name__ == "__main__":
    # Example words list (you can replace with your own)
    anchor_words = load_anchor_words('anchor_words.txt')
    print(anchor_words)
    dataset = build_dataset(anchor_words, size=10000, delay=1.5)
    save_dataset(dataset, "rhymes_means_dataset.csv")
    print(f"Saved {len(dataset)} pairs to 'rhyme_dataset_muse.csv'")