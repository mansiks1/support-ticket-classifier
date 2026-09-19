"""Controlled preprocessing. Normal service uses the selected fitted pipeline."""

import re

from nltk.stem import WordNetLemmatizer


def remove_punctuation(text):
    return re.sub(r"[^\w\s]", " ", text.lower())


def lemmatise(text):
    lemmatiser = WordNetLemmatizer()
    return " ".join(lemmatiser.lemmatize(word) for word in re.findall(r"\b\w\w+\b", text.lower()))
