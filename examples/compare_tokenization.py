#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pyserini",
# ]
# ///
"""Compare tokenization between impact-index and Pyserini/Lucene."""

import impact_index
from pyserini.analysis import Analyzer, get_lucene_analyzer

# Setup
lucene_analyzer = Analyzer(get_lucene_analyzer())

# Use a TextAnalyzer with Porter stemmer and English stop words
# (no vocab needed for tokenization comparison - we just check stems)

test_sentences = [
    "City buses are running on time.",
    "what is paula deen's brother",
    "Androgen receptor define",
    "treatment of varicose veins in legs",
    "what is operating system misconfiguration",
    "what is probability biology",
    "the king's castle was beautiful",
    "U.S.A. is a country",
    "the price is $3.14 per unit",
    "e-mail addresses and co-operation",
    "don't worry about it",
    "children's books are great",
]

stop_words = set(impact_index.get_stop_words("english"))

print(f"{'Sentence':<50} {'Ours':<40} {'Lucene':<40} {'Match'}")
print("=" * 140)

mismatches = 0
for text in test_sentences:
    # Our tokenization: split on !alphanumeric, lowercase, strip possessives, stop words, Porter stem
    # Simplified Python version matching the Rust code:
    import re
    cleaned = text.replace("'s ", " ").replace("'s", "").replace("'S ", " ").replace("'S", "")
    tokens = [t.lower() for t in re.findall(r'\w+', cleaned) if t.lower() not in stop_words]
    our_stems = [impact_index.get_stop_words.__module__ and t for t in tokens]  # placeholder
    # Actually use porter_stemmer for Python side
    try:
        from porter_stemmer import stem
        our_stems = sorted([stem(t) for t in tokens])
    except ImportError:
        # Fallback: just show tokens
        our_stems = sorted(tokens)

    lucene_stems = sorted(lucene_analyzer.analyze(text))

    match = "OK" if our_stems == lucene_stems else "DIFF"
    if match == "DIFF":
        mismatches += 1

    our_str = " ".join(our_stems)
    luc_str = " ".join(lucene_stems)
    print(f"{text:<50} {our_str:<40} {luc_str:<40} {match}")

    if match == "DIFF":
        # Show what's different
        our_set = set(our_stems)
        luc_set = set(lucene_stems)
        only_ours = our_set - luc_set
        only_lucene = luc_set - our_set
        if only_ours:
            print(f"{'':50} Only ours: {only_ours}")
        if only_lucene:
            print(f"{'':50} Only Lucene: {only_lucene}")

print(f"\n{mismatches} mismatches out of {len(test_sentences)} sentences")
