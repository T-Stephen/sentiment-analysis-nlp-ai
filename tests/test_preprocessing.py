import pytest
from app import clean_and_tokenize

def test_tokenization_and_lowercase():
    tokens = clean_and_tokenize("INCREDIBLE Performance and Direction!")
    words = [w for w, _ in tokens]
    stems = [s for _, s in tokens]
    assert "incredible" in words
    assert "performance" in words
    assert "direction" in words

def test_stopwords_removal():
    tokens = clean_and_tokenize("this is an amazing and truly great film about life")
    words = [w for w, _ in tokens]
    # English stopwords like 'this', 'is', 'an', 'and', 'about' should be filtered
    assert "this" not in words
    assert "is" not in words
    assert "an" not in words
    assert "amazing" in words or "great" in words

def test_stemming_reduction():
    tokens = clean_and_tokenize("connecting connected connects")
    stems = [s for _, s in tokens]
    # Porter stemmer maps connecting, connected, connects to connect
    assert all(s == "connect" for s in stems)

def test_punctuation_and_html_filtering():
    tokens = clean_and_tokenize("<br />Awesome movie! 100% recommended.")
    stems = [s for _, s in tokens]
    assert "awesom" in stems or "recommend" in stems
    assert "100" not in stems
    assert "br" not in stems

def test_empty_string_preprocessing():
    tokens = clean_and_tokenize("")
    assert tokens == []
    tokens_spaces = clean_and_tokenize("   \n\t  ")
    assert tokens_spaces == []
