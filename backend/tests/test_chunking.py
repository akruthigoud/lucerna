from app.chunking import chunk_text, split_sentences


def test_split_sentences():
    assert split_sentences("One here. Two here! Three?") == ["One here.", "Two here!", "Three?"]


def test_split_sentences_empty():
    assert split_sentences("   \n ") == []


def test_short_text_is_one_chunk():
    text = "First sentence. Second sentence."
    chunks = chunk_text(text, size=500)
    assert len(chunks) == 1 and chunks[0].text == text


def test_chunks_stay_near_size():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 230 for c in chunks)


def test_overlap_carries_tail_into_next_chunk():
    text = " ".join(f"This is sentence {i}." for i in range(30))
    chunks = chunk_text(text, size=120, overlap=40)
    last_sentence = "This is sentence" + chunks[0].text.rsplit("This is sentence", 1)[1]
    head = chunks[1].text[: len(chunks[0].text)]
    assert last_sentence in head
    assert chunks[0].text.endswith(chunks[1].text.split(last_sentence)[0] + last_sentence)


def test_indices_are_sequential():
    text = " ".join(f"Point {i} matters in this report." for i in range(40))
    chunks = chunk_text(text, size=100, overlap=20)
    assert [c.index for c in chunks] == list(range(len(chunks)))
