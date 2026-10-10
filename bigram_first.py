import re
from string import punctuation
from time import altzone
from typing import Counter
from pypdf import PdfReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import bigrams
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk import ConditionalFreqDist
import random

# Copus definition and normalization

reader = PdfReader() #add path to pdf file used as corpus

text = " ".join(page.extract_text() or "" for page in reader.pages)
text = re.sub(r"-\n", "", text)
text = re.sub(r"\s+", " ", text)

tokens = sent_tokenize(text)
tokens[0]=tokens[0][tokens[0].find("This"):]
tokens.insert(0, "1.")
tokens.pop(-1)
words = []
counts = Counter()

for s in tokens:
    w = word_tokenize(s.lower())
    words.append(w)
    counts.update(bigrams(w))

# Bigram collocation finder

finder = BigramCollocationFinder.from_words([w for s in words for w in s])
finder.apply_freq_filter(3)

# Probability distribution of bigrams

cfd = ConditionalFreqDist()
for w in words:
    for a, b in bigrams(w):
        cfd[a][b] += 1

def generate(start, length=20):
    """
    Generates randomly a sequence of words following start word, using the conditional frequency distribution of bigrams.
    """
    out = [start]
    for _ in range(length - 1):
        followers = cfd[out[-1]]
        if not followers:
            break
        nxt = random.choices(list(followers), weights=followers.values())[0]
        out.append(nxt)
    return " ".join(out)


while True:
    print(generate(input("\nAsk something to Nietzsche?\n\n")[-1]))