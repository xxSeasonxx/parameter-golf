from pathlib import Path

from data.download_hf_docs_and_tokenize import load_specs, tokenizer_kind


ROOT = Path(__file__).resolve().parents[1]


def test_default_tokenizer_config_remains_sp1024_only():
    specs = load_specs(ROOT / "data" / "tokenizer_specs.json")

    assert specs == [
        {
            "name": "sp_bpe_1024",
            "dataset_suffix": "sp1024",
            "vocab_size": 1024,
        }
    ]
    assert tokenizer_kind(specs[0]) == "sentencepiece_bpe"


def test_research_tokenizer_config_is_larger_vocab_only():
    specs = load_specs(ROOT / "data" / "tokenizer_specs_research.json")

    assert specs == [
        {
            "name": "sp_bpe_2048",
            "dataset_suffix": "sp2048",
            "vocab_size": 2048,
            "model_prefix": "fineweb_2048_bpe",
        },
        {
            "name": "sp_bpe_4096",
            "dataset_suffix": "sp4096",
            "vocab_size": 4096,
            "model_prefix": "fineweb_4096_bpe",
        },
    ]
    assert [tokenizer_kind(spec) for spec in specs] == [
        "sentencepiece_bpe",
        "sentencepiece_bpe",
    ]


def test_research_tokenizer_identifiers_are_unique_and_uint16_safe():
    specs = load_specs(ROOT / "data" / "tokenizer_specs_research.json")

    for field in ("name", "dataset_suffix", "model_prefix"):
        values = [spec[field] for spec in specs]
        assert len(values) == len(set(values))

    vocab_sizes = [int(spec["vocab_size"]) for spec in specs]
    assert vocab_sizes == sorted(vocab_sizes)
    assert all(1024 < vocab_size < 2**16 for vocab_size in vocab_sizes)
