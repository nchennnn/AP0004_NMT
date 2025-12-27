import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer
# python filter_vocab_embed.py --data data/train_100k.jsonl data/valid.jsonl data/test.jsonl --src-lang zh --tgt-lang en --output filtered_vocab --tokenizer-name google/mt5-small

def read_jsonl_texts(path: Path, src_lang: str, tgt_lang: str) -> Iterable[str]:
    """Yield source and target texts from a jsonl parallel corpus."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            src = data.get(src_lang)
            tgt = data.get(tgt_lang)
            if src:
                yield src
            if tgt:
                yield tgt


def collect_used_token_ids(
    tokenizer: T5Tokenizer, data_paths: List[Path], src_lang: str, tgt_lang: str
) -> Set[int]:
    """Collect all token ids that appear in the dataset (plus special tokens)."""
    used: Set[int] = set(tokenizer.all_special_ids)
    for path in data_paths:
        for text in read_jsonl_texts(path, src_lang, tgt_lang):
            # Don't truncate, collect complete statistics
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            used.update(ids)
    return used


def build_mappings(tokenizer: T5Tokenizer, kept_ids: Set[int]) -> Tuple[List[int], Dict[str, int], Dict[int, int], bool]:
    """
    Returns:
        kept_sorted: Sorted list of old ids
        new_vocab: token -> new id
        old_to_new: old id -> new id
        has_s_token: Whether the original tokenizer has <s> token
    """
    vocab = tokenizer.get_vocab()
    id_to_token = {idx: token for token, idx in vocab.items()}
    kept_sorted = sorted(kept_ids)

    # Check if original tokenizer has <s> token
    has_s_token = "<s>" in vocab

    new_vocab: Dict[str, int] = {}
    old_to_new: Dict[int, int] = {}
    for new_id, old_id in enumerate(kept_sorted):
        token = id_to_token[old_id]
        new_vocab[token] = new_id
        old_to_new[old_id] = new_id

    # If original tokenizer doesn't have <s>, manually add it
    if not has_s_token:
        # Add <s> to the end of vocab
        new_s_id = len(new_vocab)
        new_vocab["<s>"] = new_s_id
        # Note: don't add to old_to_new because original tokenizer doesn't have this token

    return kept_sorted, new_vocab, old_to_new, has_s_token


def slice_embedding(model_name: str, kept_ordered_ids: List[int], has_s_token: bool, new_vocab: Dict[str, int]) -> torch.Tensor:
    """Load pretrained model and slice its input embedding matrix, and add <s> token embedding if needed."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    emb = model.get_input_embeddings().weight.data
    filtered_emb = emb[kept_ordered_ids].clone()
    
    # If original tokenizer doesn't have <s>, initialize a new embedding vector
    if not has_s_token:
        # Get embedding dimension
        emb_dim = filtered_emb.size(1)
        # Use <pad> token embedding as initialization (if exists), otherwise use random initialization
        vocab = model.get_input_embeddings().weight.data
        pad_token_id = model.config.pad_token_id if hasattr(model.config, 'pad_token_id') else None
        
        if pad_token_id is not None and pad_token_id < vocab.size(0):
            # Use <pad> embedding as initialization
            s_emb = vocab[pad_token_id].clone()
        else:
            # Random initialization (using same distribution as model embedding)
            # Use normal distribution initialization with same std as embedding matrix
            std = filtered_emb.std().item()
            s_emb = torch.randn(emb_dim) * std
        
        # Add <s> embedding to the end of embedding matrix
        filtered_emb = torch.cat([filtered_emb, s_emb.unsqueeze(0)], dim=0)
    
    return filtered_emb


def save_outputs(
    output_dir: Path,
    new_vocab: Dict[str, int],
    old_to_new: Dict[int, int],
    kept_ids: List[int],
    filtered_emb: torch.Tensor,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save new vocab mapping
    (output_dir / "vocab.json").write_text(json.dumps(new_vocab, ensure_ascii=False, indent=2), encoding="utf-8")
    meta = {
        "old_to_new": {str(k): v for k, v in old_to_new.items()},
        "kept_old_ids": kept_ids,
        "size": len(new_vocab),
    }
    (output_dir / "mapping.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # For quick viewing of kept tokens
    with (output_dir / "tokens.txt").open("w", encoding="utf-8") as f:
        for token, new_id in sorted(new_vocab.items(), key=lambda x: x[1]):
            f.write(f"{new_id}\t{token}\n")

    # Save filtered embedding
    torch.save({"embedding": filtered_emb, "old_to_new": old_to_new, "kept_old_ids": kept_ids}, output_dir / "embedding.pt")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter google/mt5-small tokenizer vocab and corresponding embedding based on tokens appearing in dataset."
    )
    parser.add_argument("--tokenizer-name", type=str, default="google/mt5-small", help="HuggingFace tokenizer / model name")
    parser.add_argument(
        "--data",
        type=str,
        nargs="+",
        required=True,
        help="List of jsonl files for vocab statistics (containing src/tgt fields)",
    )
    parser.add_argument("--src-lang", type=str, default="zh", help="Source language field name")
    parser.add_argument("--tgt-lang", type=str, default="en", help="Target language field name")
    parser.add_argument("--output", type=str, default="filtered_vocab", help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    data_paths = [Path(p) for p in args.data]
    for p in data_paths:
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name)

    kept_ids = collect_used_token_ids(tokenizer, data_paths, args.src_lang, args.tgt_lang)
    kept_sorted, new_vocab, old_to_new, has_s_token = build_mappings(tokenizer, kept_ids)
    filtered_emb = slice_embedding(args.tokenizer_name, kept_sorted, has_s_token, new_vocab)

    save_outputs(Path(args.output), new_vocab, old_to_new, kept_sorted, filtered_emb)
    print(f"Filtering completed, kept {len(new_vocab)} tokens, results written to {args.output}")
    if not has_s_token:
        print(f"Added <s> token (BOS) to vocab, ID: {new_vocab['<s>']}")


if __name__ == "__main__":
    main()

