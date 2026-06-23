from tokenizer.bpe_tokenizer import BPETokenizer


def test_bpe_roundtrip():
    text = "hello world hello world foo bar"
    tok = BPETokenizer()
    tok.train(text, vocab_size=30, verbose=False)
    decoded = tok.decode(tok.encode("hello world"))
    assert "hello" in decoded and "world" in decoded


def test_bpe_save_load(tmp_path):
    text = "the quick brown fox jumps over the lazy dog"
    tok = BPETokenizer()
    tok.train(text, vocab_size=50, verbose=False)
    path = str(tmp_path / "bpe.json")
    tok.save(path)
    tok2 = BPETokenizer.load(path)
    assert tok.encode("the fox") == tok2.encode("the fox")


def test_bpe_learns_merges():
    # Frequent character pairs should be merged into multi-char tokens.
    text = "the the the then there their them they" * 5
    tok = BPETokenizer()
    tok.train(text, vocab_size=40, verbose=False)
    assert len(tok.merges) > 0
    assert any(len(token) > 1 for token in tok.vocab)
