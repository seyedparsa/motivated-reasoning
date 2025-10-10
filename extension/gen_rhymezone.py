import requests
from bs4 import BeautifulSoup
import random
import time
import csv

# Function to get rhymes for a given word from RhymeZone
def get_rhymes(word):
    url = f"https://www.rhymezone.com/r/rhyme.cgi?Word={word}&typeofrhyme=perfect"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Network error when fetching rhymes for '{word}': {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    
    rhymes = []
    for a in soup.find_all('a'):
        href = a.get('href')
        if href and href.startswith('/r/d='):
            rhyme_word = a.text.strip()
            if rhyme_word.isalpha():  # Keep only alphabetic words
                rhymes.append(rhyme_word.lower())
    return list(set(rhymes))  # remove duplicates

# Function to build dataset
def build_dataset(seed_words, size, delay=1.0):
    dataset = []
    rhymes_dics = {}

    for word in seed_words:
        print(f"Processing '{word}'...")
        rhymes = get_rhymes(word)
        print(f"Found {len(rhymes)} rhymes for '{word}'")
        for other_word in seed_words:
            if other_word == word:
                continue
            if other_word in rhymes:
                seed_words.remove(other_word)
                print(f"Found rhyme in the seed words: {word} - {other_word}")
        rhymes_dics[word] = rhymes
        time.sleep(delay)  # Be polite to RhymeZone

    for _ in range(size):
        label = random.choice([0, 1])
        if label == 1:
            seed = random.choice(seed_words)
            word1, word2 = random.choices(rhymes_dics[seed], k=2)
        else:            
            seed1, seed2 = random.sample(seed_words, 2)
            word1, word2 = random.choice(rhymes_dics[seed1]), random.choice(rhymes_dics[seed2])
        dataset.append((word1, word2, label))

    return dataset

# Save dataset to CSV
def save_dataset(dataset, filename="rhyme_dataset_crawl.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['word1', 'word2', 'label'])
        writer.writerows(dataset)

if __name__ == "__main__":
    # Example words list (you can replace with your own)
    seed_words = ["cat", "dog", "fast", "tree", "love", "light", "sun"]

    dataset = build_dataset(seed_words, size=1000, delay=1.5)
    save_dataset(dataset)
    print(f"Saved {len(dataset)} pairs to 'rhyme_dataset_crawl.csv'")