import pytest

from tokenizer.char_tokenizer import CharTokenizer

SAMPLE = "Hello, World! To be or not to be."


@pytest.fixture
def tok():
    return CharTokenizer(text=SAMPLE)


def test_roundtrip(tok):
    ids = tok.encode(SAMPLE)
    assert tok.decode(ids) == SAMPLE


def test_vocab_size(tok):
    assert tok.vocab_size == len(set(SAMPLE))


def test_encode_type(tok):
    ids = tok.encode("Hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_save_load(tok, tmp_path):
    path = str(tmp_path / "vocab.json")
    tok.save(path)
    tok2 = CharTokenizer(vocab_path=path)
    assert tok2.encode(SAMPLE) == tok.encode(SAMPLE)
    assert tok2.decode(tok.encode(SAMPLE)) == SAMPLE
