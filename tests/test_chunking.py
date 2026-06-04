from app.rag.chunking import (
    _split_by_structure,
    _merge_small_blocks,
    _split_oversized,
    _add_sentence_overlap,
    proposal_based_chunk_text,
)


def test_split_by_structure():
    text = "# Heading 1\nSome paragraph.\n\n## Heading 2\nAnother paragraph."
    blocks = _split_by_structure(text)
    assert len(blocks) == 2
    assert blocks[0].startswith("# Heading 1")
    assert blocks[1].startswith("## Heading 2")


def test_merge_small_blocks():
    blocks = ["Short text.", "Another short block.", "This is a longer block that shouldn't merge."]
    # min_chars = 30
    merged = _merge_small_blocks(blocks, min_chars=30)
    assert len(merged) == 2
    assert merged[0] == "Short text.\n\nAnother short block."
    assert merged[1] == "This is a longer block that shouldn't merge."


def test_split_oversized():
    blocks = ["This is sentence one. This is sentence two. This is sentence three."]
    # max_chars = 25
    split = _split_oversized(blocks, max_chars=25)
    assert len(split) == 3
    assert split[0] == "This is sentence one."
    assert split[1] == "This is sentence two."
    assert split[2] == "This is sentence three."


def test_add_sentence_overlap():
    chunks = ["Sentence one. Sentence two.", "Sentence three. Sentence four."]
    overlapped = _add_sentence_overlap(chunks, overlap=1)
    assert len(overlapped) == 2
    assert overlapped[0] == "Sentence one. Sentence two."
    assert overlapped[1] == "Sentence two. Sentence three. Sentence four."


def test_proposal_based_chunk_text():
    text = "# Title\nThis is paragraph one. It is short.\n\nThis is paragraph two. It is also short."
    chunks = proposal_based_chunk_text(text)
    assert len(chunks) > 0
