"""The catalog of data/all.jsonl.gz: datasets, frozen test sets, grown batches, prediction keys."""

from selfjev.data.catalog import DATASETS, TEST_DATASETS, datasets, request_sha


def test_test_datasets_are_catalogued():
    names = [d[0] for d in DATASETS]
    assert len(names) == len(set(names)) and set(names) >= TEST_DATASETS


def test_datasets_adds_every_built_batch(tmp_path):
    (tmp_path / "data/batches/b1").mkdir(parents=True)
    (tmp_path / "data/batches/b1.jsonl").write_text("")
    (tmp_path / "data/batches/b1/README.md").write_text("# Negated numbers\n\nmore\n")
    (tmp_path / "data/batches/b2.jsonl").write_text("")  # no README yet
    extra = {d[0]: d for d in datasets(tmp_path)[len(DATASETS) :]}
    assert extra["b1"] == ("b1", "data/batches/b1.jsonl", "batch: Negated numbers", "training", "data/batches/b1/README.md")
    assert extra["b2"][2] == "batch: no README.md yet" and set(extra) == {"b1", "b2"}


def test_request_sha_keys_text_and_question():
    ex = {"state": "t", "question": {"type": "binary", "instruction": "q?"}, "target": True}
    assert request_sha(ex) == request_sha(ex | {"target": False, "id": "other"})  # the label is not part of the request
    assert request_sha(ex) != request_sha(ex | {"state": "t2"})
    assert len(request_sha(ex)) == 16
